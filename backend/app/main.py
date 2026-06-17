from __future__ import annotations

import json
import os
import re
import time
from json import JSONDecodeError
from collections import Counter
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from psycopg.types.json import Jsonb

from .datasets import source_dataset_inventory
from .db import connect, init_database
from .experiment_runner import create_experiment_job, default_experiment_config
from .experiment_runner import normalize_experiment_config
from .llm_estimation import estimate_experiment_llm_usage, estimate_llm_usage
from .retrieval import (
    ALPACA_METADATA_URL,
    MAX_RETURNED_CANDIDATES,
    MAX_RETRIEVAL_CANDIDATES,
    alpaca_search,
    alpaca_token,
    bounded_candidate_count,
    bounded_returned_candidate_count,
    build_alpaca_query,
    extract_hits,
    normalize_url_slug,
)


app = FastAPI(title="Coverage Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackRequest(BaseModel):
    category: str = Field(default="note", max_length=80)
    note: str = Field(min_length=1, max_length=6000)
    metadata: dict[str, Any] = Field(default_factory=dict)


OPENROUTER_REQUIRED_MODEL = "openai/gpt-oss-120b"
TYPO_CORRECTION_CONFIDENCE_THRESHOLD = 0.85


class LiveAttemptRequest(BaseModel):
    candidate_count: int = Field(default=100, ge=1, le=MAX_RETRIEVAL_CANDIDATES)
    query_text: str | None = None
    human_guidance: str | None = Field(default=None, max_length=3000)
    llm_config: dict[str, Any] = Field(default_factory=dict)


class ExperimentJobRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


class LlmEstimateRequest(BaseModel):
    input: str = Field(default="", max_length=200000)
    model: str | None = None
    max_completion_tokens: int | None = Field(default=None, ge=0, le=200000)
    config: dict[str, Any] = Field(default_factory=dict)


class ExperimentEstimateRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


SENSITIVE_CONFIG_KEYS = {"llm_api_key", "openrouter_api_key", "api_key"}


def _redact_config(config: Any) -> Any:
    if not isinstance(config, dict):
        return config
    redacted = dict(config)
    for key in SENSITIVE_CONFIG_KEYS:
        if redacted.get(key):
            redacted[key] = "__configured__"
    return redacted


def _redact_job(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["config"] = _redact_config(item.get("config"))
    return item


def _row_or_404(row: dict[str, Any] | None, label: str = "Record") -> dict[str, Any]:
    if not row:
        raise HTTPException(status_code=404, detail=f"{label} not found")
    return row


def _parse_buckets(raw: str) -> list[int]:
    buckets: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if 1 <= value <= MAX_RETRIEVAL_CANDIDATES and value not in buckets:
            buckets.append(value)
    return buckets or [1, 5, 10, 20, 50, 100, 250, 500, 1000]


def _default_coverage_buckets(max_rank: int) -> list[int]:
    base = [1, 5, 10, 20, 50, 100, 250, 500, 1000]
    bounded = max(1, min(int(max_rank or 1), MAX_RETRIEVAL_CANDIDATES))
    buckets = [value for value in base if value <= bounded]
    if bounded not in buckets:
        buckets.append(bounded)
    return buckets


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _tokens(value: Any, *, limit: int = 12) -> list[str]:
    text = str(value or "").casefold()
    result: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text):
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    return []


def _entity_text(entity: dict[str, Any]) -> str:
    parts = [
        entity.get("label"),
        entity.get("description"),
        entity.get("context_string"),
        entity.get("coarse_type"),
        entity.get("fine_type"),
    ]
    parts.extend(_string_list(entity.get("aliases")))
    return " ".join(str(part) for part in parts if part)


def _table_context_from_payload(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        return {"header": [], "rows": []}
    table_context = raw_payload.get("table_context")
    if isinstance(table_context, dict):
        return {
            "header": table_context.get("header") if isinstance(table_context.get("header"), list) else [],
            "target_row_id": table_context.get("target_row_id"),
            "target_col_id": table_context.get("target_col_id"),
            "header_cell": table_context.get("header_cell"),
            "rows": table_context.get("rows") if isinstance(table_context.get("rows"), list) else [],
        }
    rows = raw_payload.get("context_rows")
    if isinstance(rows, list):
        return {
            "header": raw_payload.get("header") if isinstance(raw_payload.get("header"), list) else [],
            "target_row_id": raw_payload.get("row_id"),
            "target_col_id": raw_payload.get("col_id"),
            "header_cell": raw_payload.get("header_cell"),
            "rows": rows,
        }
    return {"header": [], "rows": []}


def _is_good_augmentation_term(term: str) -> bool:
    normalized = normalize_query_text(term)
    if not normalized:
        return False
    instruction_words = {
        "add",
        "append",
        "include",
        "instruction",
        "instructions",
        "match",
        "query",
        "search",
        "term",
        "terms",
        "type",
        "types",
        "use",
        "using",
    }
    tokens = _tokens(normalized, limit=10)
    if len(tokens) > 2:
        return False
    if any(token in instruction_words for token in tokens):
        return False
    if re.fullmatch(r"col\d+", normalized):
        return False
    if re.fullmatch(r"[\d,._-]+", normalized):
        return False
    if len(normalized) < 3:
        return False
    return True


def _table_context_terms(table_context: dict[str, Any], query_text: str, *, limit: int = 2) -> list[str]:
    values: list[str] = []
    query_tokens = set(_tokens(query_text, limit=24))
    header_cell = table_context.get("header_cell")
    if header_cell:
        values.append(str(header_cell))
    header = table_context.get("header")
    if isinstance(header, list):
        values.extend(str(item) for item in header if item is not None and not str(item).lower().startswith("col"))
    for row in table_context.get("rows") or []:
        if not isinstance(row, dict):
            continue
        if not row.get("is_target"):
            continue
        cells = row.get("cells")
        if isinstance(cells, list):
            values.extend(str(cell) for cell in cells if cell is not None)

    terms: list[str] = []
    seen: set[str] = set(query_tokens)
    for token in _tokens(" ".join(values), limit=limit * 5):
        if token in seen or not _is_good_augmentation_term(token):
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _target_context_text(table_context: dict[str, Any], *, limit: int = 700) -> str:
    values: list[str] = []
    header = table_context.get("header")
    if isinstance(header, list):
        values.append("headers: " + " | ".join(str(item) for item in header if item is not None))
    for row in table_context.get("rows") or []:
        if not isinstance(row, dict) or not row.get("is_target"):
            continue
        cells = row.get("cells")
        if isinstance(cells, list):
            values.append("target row: " + " | ".join(str(cell) for cell in cells if cell is not None))
            break
    return " ".join(values)[:limit]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(stripped[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except JSONDecodeError:
                return {}
    return {}


def _clean_augmentation_terms(raw_terms: Any, query_text: str, *, limit: int = 2) -> list[str]:
    if isinstance(raw_terms, str):
        raw_terms = re.split(r"[,;\n]", raw_terms)
    if not isinstance(raw_terms, list | tuple):
        return []
    query_norm = normalize_query_text(query_text)
    terms: list[str] = []
    seen: set[str] = set()
    token_budget = 3
    used_tokens = 0
    for raw in raw_terms:
        term = re.sub(r"\s+", " ", str(raw or "")).strip(" \t\r\n,;")
        if not term:
            continue
        term = term[:48]
        normalized = normalize_query_text(term)
        if not normalized or normalized == query_norm or normalized in seen:
            continue
        if not _is_good_augmentation_term(term):
            continue
        term_tokens = max(1, len(_tokens(term, limit=10)))
        if used_tokens + term_tokens > token_budget and terms:
            continue
        seen.add(normalized)
        terms.append(term)
        used_tokens += term_tokens
        if len(terms) >= limit:
            break
    return terms


def normalize_query_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _llm_api_key(config: dict[str, Any]) -> str:
    return str(config.get("llm_api_key") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "").strip()


def _llm_endpoint(config: dict[str, Any]) -> str:
    raw = str(config.get("llm_api_url") or os.environ.get("LLM_API_URL") or os.environ.get("OPENROUTER_CHAT_URL") or "https://openrouter.ai/api/v1/chat/completions").strip()
    trimmed = raw.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return raw
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    if "://" in trimmed and "/" not in trimmed.split("://", 1)[1]:
        return f"{trimmed}/v1/chat/completions"
    return raw


def _llm_label(config: dict[str, Any]) -> str:
    provider = str(config.get("llm_provider_name") or config.get("llm_provider") or "LLM").strip()
    return provider if provider.casefold() != "openai_compatible" else "OpenAI-compatible"


def _openrouter_provider_slug(provider_name: Any) -> str:
    return re.sub(r"\s+", "-", str(provider_name or "").strip().casefold())


def _openrouter_provider_config(config: dict[str, Any]) -> dict[str, Any] | None:
    provider_slug = _openrouter_provider_slug(config.get("llm_provider_name"))
    if not provider_slug:
        return None
    return {
        "order": [provider_slug],
        "allow_fallbacks": bool(config.get("openrouter_allow_fallbacks", True)),
    }


def _llm_request_body(messages: list[dict[str, str]], config: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config["llm_model"],
        "temperature": float(config["llm_temperature"]),
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    if config.get("llm_provider") == "openrouter":
        body["include_reasoning"] = False
        body["reasoning"] = {"effort": config["llm_reasoning_effort"]}
        provider_config = _openrouter_provider_config(config)
        if provider_config:
            body["provider"] = provider_config
    if config.get("llm_max_tokens") is not None:
        body["max_tokens"] = int(config["llm_max_tokens"])
    return body


def _llm_json_request(messages: list[dict[str, str]], llm_config: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = normalize_experiment_config(llm_config or {})
    token = _llm_api_key(config)
    if not token:
        raise RuntimeError("LLM API key is not configured")
    endpoint = _llm_endpoint(config)
    body = _llm_request_body(messages, config)

    timeout = int(config["llm_timeout_seconds"])
    max_retries = max(1, int(config["llm_max_retries"]))
    retry_base_seconds = max(0.5, float(os.environ.get("LLM_RETRY_BASE_SECONDS", os.environ.get("OPENROUTER_RETRY_BASE_SECONDS", "4"))))
    last_error: Exception | None = None
    payload: dict[str, Any] = {}
    for attempt in range(1, max_retries + 1):
        request = Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "HTTP-Referer": config["llm_site_url"],
                "X-Title": config["llm_app_name"],
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read() or b"{}")
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"{_llm_label(config)} HTTP {exc.code}: {detail[:500]}")
            if exc.code == 429 and attempt < max_retries:
                time.sleep(min(90.0, retry_base_seconds * (2 ** (attempt - 1))))
                continue
            raise last_error from exc
        except (URLError, TimeoutError, JSONDecodeError) as exc:
            last_error = RuntimeError(f"{_llm_label(config)} augmentation failed: {exc}")
            if attempt < max_retries:
                time.sleep(min(30.0, retry_base_seconds * attempt))
                continue
            raise last_error from exc
    else:
        raise RuntimeError(f"{_llm_label(config)} augmentation failed: {last_error}")
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _extract_json_object(str(content)), {
        "sent": True,
        "provider": config.get("llm_provider"),
        "endpoint": endpoint,
        "request_body": body,
        "response_usage": payload.get("usage"),
        "response_model": payload.get("model"),
        "response_provider": payload.get("provider"),
        "response_id": payload.get("id"),
        "model_verified": payload.get("model") in (None, config["llm_model"]),
        "fallbacks_allowed": config.get("llm_provider") != "openrouter",
    }


def _normalize_retrieval_signal_plan(raw_plan: dict[str, Any], fallback_query: str) -> dict[str, Any]:
    def optional_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None and value != "" else None
        except (TypeError, ValueError):
            return None

    typo_corrected_mention = str(
        raw_plan.get("typo_corrected_mention") or raw_plan.get("corrected_mention") or ""
    ).strip() or None
    typo_correction_confidence = optional_float(
        raw_plan.get("typo_correction_confidence", raw_plan.get("typo_confidence"))
    )
    use_typo_correction = bool(
        typo_corrected_mention
        and typo_correction_confidence is not None
        and typo_correction_confidence >= TYPO_CORRECTION_CONFIDENCE_THRESHOLD
        and typo_corrected_mention.casefold() != fallback_query.casefold()
    )
    optimized_query = str(raw_plan.get("optimized_query") or raw_plan.get("query") or fallback_query).strip()
    if use_typo_correction:
        optimized_query = typo_corrected_mention
    normalized_mention = str(raw_plan.get("normalized_mention") or "").strip() or None
    aliases = _clean_augmentation_terms(raw_plan.get("aliases"), optimized_query, limit=6)
    context_terms = _clean_augmentation_terms(
        raw_plan.get("context_expansion_terms", raw_plan.get("context_terms")),
        optimized_query,
        limit=3,
    )
    return {
        "optimized_query": optimized_query or fallback_query,
        "normalized_mention": normalized_mention,
        "typo_corrected_mention": typo_corrected_mention,
        "typo_correction_confidence": typo_correction_confidence,
        "typo_correction_applied": use_typo_correction,
        "typo_correction_reason": str(raw_plan.get("typo_correction_reason") or "").strip() or None,
        "coarse_type": str(raw_plan.get("coarse_type") or "").strip().upper() or None,
        "fine_type": str(raw_plan.get("fine_type") or "").strip().upper() or None,
        "wikipedia_url": normalize_url_slug(raw_plan.get("wikipedia_url") or raw_plan.get("wikipedia_title")),
        "dbpedia_url": normalize_url_slug(raw_plan.get("dbpedia_url") or raw_plan.get("dbpedia_title")),
        "aliases": aliases,
        "context_expansion_terms": context_terms,
    }


def _llm_es_query_plan(
    *,
    query_text: str,
    table_context: dict[str, Any],
    human_guidance: str | None,
    llm_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, str | None, dict[str, Any]]:
    guidance = (human_guidance or "").strip()
    config = normalize_experiment_config(llm_config or {})
    if not guidance and (not config.get("llm_enabled") or not _llm_api_key(config)):
        return {
            "optimized_query": query_text,
            "normalized_mention": None,
            "coarse_type": None,
            "fine_type": None,
            "wikipedia_url": None,
            "dbpedia_url": None,
            "aliases": [],
            "context_expansion_terms": _table_context_terms(table_context, query_text, limit=1),
            "typo_corrected_mention": None,
            "typo_correction_confidence": None,
            "typo_correction_applied": False,
            "typo_correction_reason": None,
        }, "heuristic", None, {
            "sent": False,
            "reason": "LLM is disabled or the API key is not configured, and no human guidance was provided",
        }
    messages = [
        {
            "role": "system",
            "content": (
                "Return JSON only. You generate one recall-oriented Elasticsearch entity retrieval plan. "
                "Infer the intended entity from the mention, row context, column context, table context, "
                "metadata, and optional human guidance. Keep the optimized query short: preserve or lightly "
                "normalize the mention, expand abbreviations only when strongly supported, and add only a "
                "small number of highly relevant tokens. If the mention appears to contain a typo and the "
                "table context makes the correction highly confident, return the corrected mention and a "
                f"confidence >= {TYPO_CORRECTION_CONFIDENCE_THRESHOLD}; otherwise leave typo_corrected_mention null. "
                "Predict coarse_type, fine_type, Wikipedia slug, "
                "DBpedia slug, aliases, and context expansion terms when useful. These predictions will be "
                "ranking boosts only, never hard filters."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "mention": query_text,
                    "target_table_context": _target_context_text(table_context),
                    "guidance_for_es_query_generation": guidance,
                    "output_schema": {
                        "optimized_query": "short query text",
                        "normalized_mention": "normalized mention or null",
                        "typo_corrected_mention": "corrected mention only when high-confidence typo correction is justified, else null",
                        "typo_correction_confidence": "0..1 number or null",
                        "typo_correction_reason": "short reason or null",
                        "coarse_type": "PERSON | ORGANIZATION | LOCATION | MISC | CONCEPT | EVENT | PRODUCT | WORK | TYPE | null",
                        "fine_type": "specific type or null",
                        "wikipedia_url": "Wikipedia title/slug or null",
                        "dbpedia_url": "DBpedia title/slug or null",
                        "aliases": ["optional alternative surface forms"],
                        "context_expansion_terms": ["optional short discriminating terms"],
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]
    llm_request = {
        "sent": False,
        "provider": config.get("llm_provider"),
        "endpoint": _llm_endpoint(config),
        "request_body": _llm_request_body(messages, config),
    }
    try:
        parsed, llm_request = _llm_json_request(messages, config)
        plan = _normalize_retrieval_signal_plan(parsed, query_text)
        return plan, f"{config.get('llm_provider') or 'llm'}_query_plan", None, llm_request
    except Exception as exc:
        llm_request["error"] = str(exc)
        if guidance:
            return _normalize_retrieval_signal_plan({}, query_text), "guidance_not_applied_without_llm", str(exc), llm_request
        plan = _normalize_retrieval_signal_plan(
            {"context_expansion_terms": _table_context_terms(table_context, query_text, limit=1)},
            query_text,
        )
        return plan, "heuristic_query_plan_fallback", str(exc), llm_request


def _qid_query_body(qids: list[str]) -> dict[str, Any]:
    return {
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"item_category": ["ENTITY", "TYPE"]}},
                    {"terms": {"_id": qids}},
                ]
            }
        }
    }


def _top_type_distribution(candidates: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for candidate in candidates:
        coarse = str(candidate.get("coarse_type") or "MISC")
        fine = str(candidate.get("fine_type") or "MISC")
        counts[(coarse, fine)] += 1
    return [
        {"coarse_type": coarse, "fine_type": fine, "count": count}
        for (coarse, fine), count in counts.most_common(limit)
    ]


def _build_improvement_diagnostics(
    *,
    mention: dict[str, Any],
    query_text: str,
    human_guidance: str | None,
    gold_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    covered_qids: list[str],
) -> dict[str, Any]:
    gold_entities = [
        {
            "qid": row.get("qid"),
            "label": (row.get("raw_entity") or {}).get("label"),
            "description": (row.get("raw_entity") or {}).get("description"),
            "coarse_type": (row.get("raw_entity") or {}).get("coarse_type"),
            "fine_type": (row.get("raw_entity") or {}).get("fine_type"),
        }
        for row in gold_rows
        if isinstance(row.get("raw_entity"), dict)
    ]
    gold_by_qid = {
        str(row.get("qid")): row.get("raw_entity") or {}
        for row in gold_rows
        if row.get("qid") and isinstance(row.get("raw_entity"), dict)
    }
    retrieved_gold = [candidate for candidate in candidates if str(candidate.get("qid")) in set(covered_qids)]
    gold_text = " ".join(_entity_text(entity) for entity in gold_by_qid.values())
    retrieved_gold_text = " ".join(_entity_text(candidate) for candidate in retrieved_gold)
    observed_text = " ".join([query_text, human_guidance or "", mention.get("mention") or "", mention.get("lookup_text") or ""])
    context_source = retrieved_gold_text or gold_text

    gold_types = [
        {
            "qid": qid,
            "coarse_type": entity.get("coarse_type"),
            "fine_type": entity.get("fine_type"),
        }
        for qid, entity in gold_by_qid.items()
        if entity.get("coarse_type") or entity.get("fine_type")
    ]
    candidate_distribution = _top_type_distribution(candidates)
    top_distribution = candidate_distribution[0] if candidate_distribution else {}
    type_mismatch = bool(
        gold_types
        and top_distribution
        and all(
            item.get("coarse_type") != top_distribution.get("coarse_type")
            and item.get("fine_type") != top_distribution.get("fine_type")
            for item in gold_types
        )
    )

    candidate_context_tokens = [
        token
        for token in _tokens(context_source, limit=24)
        if token not in set(_tokens(observed_text, limit=48))
    ][:10]
    mention_context_tokens = _tokens(observed_text, limit=10)

    recommendations: list[str] = []
    if not covered_qids and gold_types:
        recommendations.append("Check whether the NER fine type is too broad or mapped to MISC for this mention family.")
    if type_mismatch:
        recommendations.append("Candidate pool is dominated by a different type than the gold entity; add or tighten lexical NER clues.")
    if candidate_context_tokens:
        recommendations.append("Consider adding the suggested context tokens to the correct entity context string/indexed aliases.")
    if human_guidance:
        recommendations.append("Human guidance was used as augmentation instruction, not appended directly to the Alpaca query.")

    return {
        "covered_in_retrieval_window": bool(covered_qids),
        "covered_qids": covered_qids,
        "retrieved_count": len(candidates),
        "gold_entities": gold_entities,
        "candidate_type_distribution": candidate_distribution,
        "ner_type_hint": {
            "gold_types": gold_types,
            "type_mismatch_with_top_candidates": type_mismatch,
            "suggested_rule_tokens": _tokens(gold_text, limit=12),
        },
        "context_token_hint": {
            "mention_tokens": mention_context_tokens,
            "candidate_context_tokens_to_consider": candidate_context_tokens,
        },
        "recommendations": recommendations,
    }


@app.on_event("startup")
def startup() -> None:
    init_database()
    with connect() as conn:
        conn.execute(
            """
            UPDATE experiment_jobs
            SET status = 'failed',
                stage = 'failed',
                message = 'API restarted while this job was active',
                error = coalesce(error, 'API restarted while this job was active'),
                finished_at = now()
            WHERE status IN ('queued', 'running')
            """
        )
        conn.commit()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/database-size")
def database_size() -> dict[str, Any]:
    with connect() as conn:
        database_row = conn.execute(
            """
            SELECT
                current_database() AS database_name,
                pg_database_size(current_database()) AS total_bytes,
                pg_size_pretty(pg_database_size(current_database())) AS total_pretty
            """
        ).fetchone()
        table_rows = conn.execute(
            """
            SELECT
                schemaname AS schema_name,
                relname AS table_name,
                pg_total_relation_size(format('%I.%I', schemaname, relname)) AS total_bytes,
                pg_relation_size(format('%I.%I', schemaname, relname)) AS table_bytes,
                pg_indexes_size(format('%I.%I', schemaname, relname)) AS index_bytes,
                n_live_tup AS estimated_rows
            FROM pg_stat_user_tables
            ORDER BY pg_total_relation_size(format('%I.%I', schemaname, relname)) DESC, relname
            """
        ).fetchall()
    return {
        "database_name": database_row["database_name"],
        "total_bytes": database_row["total_bytes"],
        "total_pretty": database_row["total_pretty"],
        "tables": list(table_rows),
    }


@app.get("/api/source-datasets")
def source_datasets() -> list[dict[str, Any]]:
    with connect() as conn:
        return source_dataset_inventory(conn)


@app.get("/api/experiment-defaults")
def experiment_defaults() -> dict[str, Any]:
    return default_experiment_config()


@app.get("/api/config-status")
def config_status() -> dict[str, Any]:
    defaults = default_experiment_config()
    llm_key_configured = bool(os.environ.get("LLM_API_KEY", "").strip() or os.environ.get("OPENROUTER_API_KEY", "").strip())
    return {
        "alpaca_configured": bool(alpaca_token()),
        "llm_configured": llm_key_configured,
        "llm_provider": defaults["llm_provider"],
        "llm_provider_name": defaults["llm_provider_name"],
        "llm_api_url": defaults["llm_api_url"],
        "llm_model": defaults["llm_model"],
        "openrouter_configured": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
        "openrouter_model": defaults["openrouter_model"],
        "openrouter_provider": defaults["openrouter_provider"],
        "openrouter_reasoning_effort": "high",
        "openrouter_allow_fallbacks": defaults["openrouter_allow_fallbacks"],
    }


@app.post("/api/llm/estimate")
def llm_estimate(request: LlmEstimateRequest) -> dict[str, Any]:
    config = normalize_experiment_config({**request.config})
    if request.model:
        config["llm_model"] = request.model.strip()
        config["openrouter_model"] = config["llm_model"]
    try:
        return estimate_llm_usage(
            input_text=request.input,
            config=config,
            max_completion_tokens=request.max_completion_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/experiment-estimate")
def experiment_estimate(request: ExperimentEstimateRequest) -> dict[str, Any]:
    config = normalize_experiment_config(request.config)
    try:
        return estimate_experiment_llm_usage(config)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/experiment-jobs")
def start_experiment_job(request: ExperimentJobRequest) -> dict[str, Any]:
    try:
        return _redact_job(create_experiment_job(request.config))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/experiment-jobs")
def experiment_jobs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT j.*, r.name AS imported_run_name
            FROM experiment_jobs j
            LEFT JOIN runs r ON r.id = j.imported_run_id
            ORDER BY j.created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return [_redact_job(dict(row)) for row in rows]


@app.get("/api/experiment-jobs/{job_id}")
def experiment_job(job_id: int) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT j.*, r.name AS imported_run_name
            FROM experiment_jobs j
            LEFT JOIN runs r ON r.id = j.imported_run_id
            WHERE j.id = %s
            """,
            (job_id,),
        ).fetchone()
    return _redact_job(_row_or_404(row, "Experiment job"))


@app.delete("/api/experiment-jobs/failed")
def clear_failed_experiment_jobs() -> dict[str, int]:
    with connect() as conn:
        result = conn.execute("DELETE FROM experiment_jobs WHERE status = 'failed'")
        conn.commit()
    return {"deleted": result.rowcount or 0}


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, source_path, source_filename, imported_at, table_count,
                   mention_count, candidate_count, covered_count,
                   CASE WHEN mention_count > 0 THEN covered_count::float / mention_count ELSE 0 END AS imported_coverage
            FROM runs
            ORDER BY imported_at DESC
            """
        ).fetchall()
    return list(rows)


@app.get("/api/runs/{run_id}")
def run_detail(run_id: int) -> dict[str, Any]:
    with connect() as conn:
        run = conn.execute(
            """
            SELECT id, name, source_path, source_filename, imported_at, table_count,
                   mention_count, candidate_count, covered_count,
                   CASE WHEN mention_count > 0 THEN covered_count::float / mention_count ELSE 0 END AS imported_coverage,
                   raw_summary, raw_sampling_config
            FROM runs
            WHERE id = %s
            """,
            (run_id,),
        ).fetchone()
    return _row_or_404(run, "Run")


@app.get("/api/runs/{run_id}/filters")
def run_filters(run_id: int) -> dict[str, Any]:
    with connect() as conn:
        datasets = conn.execute(
            """
            SELECT dataset_id, count(*) AS mention_count
            FROM mentions
            WHERE run_id = %s AND dataset_id IS NOT NULL
            GROUP BY dataset_id
            ORDER BY dataset_id
            """,
            (run_id,),
        ).fetchall()
        stages = conn.execute(
            """
            SELECT coalesce(retrieval_stage, 'unknown') AS retrieval_stage, count(*) AS candidate_count
            FROM candidates c
            JOIN mentions m ON m.id = c.mention_id
            WHERE m.run_id = %s
            GROUP BY coalesce(retrieval_stage, 'unknown')
            ORDER BY candidate_count DESC, retrieval_stage
            """,
            (run_id,),
        ).fetchall()
    return {"datasets": list(datasets), "retrieval_stages": list(stages)}


@app.get("/api/runs/{run_id}/coverage")
def coverage_curve(run_id: int, buckets: str | None = Query(default=None)) -> list[dict[str, Any]]:
    with connect() as conn:
        if buckets:
            parsed = _parse_buckets(buckets)
        else:
            max_rank_row = conn.execute(
                """
                SELECT coalesce(max(coalesce(m.best_gt_rank, m.retrieved_count, c.max_candidate_rank)), 1) AS max_rank
                FROM mentions m
                LEFT JOIN LATERAL (
                    SELECT max(rank) AS max_candidate_rank
                    FROM candidates c
                    WHERE c.mention_id = m.id
                ) c ON true
                WHERE m.run_id = %s
                """,
                (run_id,),
            ).fetchone()
            parsed = _default_coverage_buckets(int(max_rank_row["max_rank"] if max_rank_row else 1))

        rows = conn.execute(
            """
            WITH selected_mentions AS (
                SELECT id, best_gt_rank
                FROM mentions
                WHERE run_id = %s
            ),
            gold_hits AS (
                SELECT c.mention_id, min(c.rank) AS gold_best_rank
                FROM candidates c
                JOIN selected_mentions m ON m.id = c.mention_id
                JOIN gold_qids g ON g.mention_id = c.mention_id AND g.qid = c.qid
                GROUP BY c.mention_id
            ),
            scored_mentions AS (
                SELECT
                    m.id,
                    nullif(
                        least(
                            coalesce(m.best_gt_rank, 2147483647),
                            coalesce(gold_hits.gold_best_rank, 2147483647)
                        ),
                        2147483647
                    ) AS imported_best_rank
                FROM selected_mentions m
                LEFT JOIN gold_hits ON gold_hits.mention_id = m.id
            ),
            ks AS (
                SELECT unnest(%s::int[]) AS k
            )
            SELECT
                ks.k,
                count(scored_mentions.id) AS total,
                count(scored_mentions.id) FILTER (WHERE scored_mentions.imported_best_rank <= ks.k) AS covered
            FROM ks
            CROSS JOIN scored_mentions
            GROUP BY ks.k
            ORDER BY ks.k
            """,
            (run_id, parsed),
        ).fetchall()

    result = []
    for row in rows:
        total = int(row["total"] or 0)
        covered = int(row["covered"] or 0)
        result.append(
            {
                "k": int(row["k"]),
                "total": total,
                "covered": covered,
                "coverage": covered / total if total else 0,
            }
        )
    return result


@app.get("/api/runs/{run_id}/mentions")
def run_mentions(
    run_id: int,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    covered: Literal["all", "covered", "missed"] = "all",
    dataset_id: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    base_conditions = ["m.run_id = %s"]
    base_params: list[Any] = [run_id]
    if dataset_id:
        base_conditions.append("m.dataset_id = %s")
        base_params.append(dataset_id)
    if search:
        base_conditions.append("(m.mention ILIKE %s OR m.lookup_text ILIKE %s OR m.cell_key ILIKE %s)")
        like = f"%{search}%"
        base_params.extend([like, like, like])

    base_where = " AND ".join(base_conditions)
    scored_cte = f"""
        WITH gold_hits AS (
            SELECT c.mention_id, min(c.rank) AS gold_best_rank
            FROM candidates c
            JOIN gold_qids g ON g.mention_id = c.mention_id AND g.qid = c.qid
            JOIN mentions m ON m.id = c.mention_id
            WHERE {base_where}
            GROUP BY c.mention_id
        ),
        scored_mentions AS (
            SELECT
                m.id, m.cell_key, m.dataset_id, m.table_id, m.row_id, m.col_id,
                m.mention, m.lookup_text, m.primary_gt_qid, m.selected_qid, m.selected_label,
                m.final_correct, m.coverage_correct, m.hit_at_1, m.hit_at_5, m.hit_at_10,
                m.hit_at_k, m.best_gt_rank, m.retrieved_count, m.candidate_count,
                (
                    coalesce(m.coverage_correct, false)
                    OR m.best_gt_rank IS NOT NULL
                    OR gold_hits.gold_best_rank IS NOT NULL
                ) AS covered_by_imported_candidates,
                nullif(
                    least(
                        coalesce(m.best_gt_rank, 2147483647),
                        coalesce(gold_hits.gold_best_rank, 2147483647)
                    ),
                    2147483647
                ) AS imported_best_rank
            FROM mentions m
            LEFT JOIN gold_hits ON gold_hits.mention_id = m.id
            WHERE {base_where}
        )
    """
    score_where = "TRUE"
    if covered == "covered":
        score_where = "covered_by_imported_candidates"
    elif covered == "missed":
        score_where = "NOT covered_by_imported_candidates"
    scored_params = [*base_params, *base_params]
    with connect() as conn:
        total = conn.execute(
            f"{scored_cte} SELECT count(*) AS total FROM scored_mentions WHERE {score_where}",
            scored_params,
        ).fetchone()
        rows = conn.execute(
            f"""
            {scored_cte}
            SELECT *
            FROM scored_mentions
            WHERE {score_where}
            ORDER BY
                CASE WHEN covered_by_imported_candidates THEN 1 ELSE 0 END ASC,
                imported_best_rank NULLS LAST,
                id ASC
            LIMIT %s OFFSET %s
            """,
            [*scored_params, limit, offset],
        ).fetchall()
    return {"total": int(total["total"] if total else 0), "rows": list(rows)}


@app.get("/api/mentions/{mention_id}")
def mention_detail(mention_id: int) -> dict[str, Any]:
    with connect() as conn:
        mention = conn.execute(
            """
            WITH gold_hits AS (
                SELECT c.mention_id, min(c.rank) AS gold_best_rank
                FROM candidates c
                JOIN gold_qids g ON g.mention_id = c.mention_id AND g.qid = c.qid
                WHERE c.mention_id = %s
                GROUP BY c.mention_id
            )
            SELECT
                m.*,
                r.name AS run_name,
                (
                    coalesce(m.coverage_correct, false)
                    OR m.best_gt_rank IS NOT NULL
                    OR gold_hits.gold_best_rank IS NOT NULL
                ) AS covered_by_imported_candidates,
                nullif(
                    least(
                        coalesce(m.best_gt_rank, 2147483647),
                        coalesce(gold_hits.gold_best_rank, 2147483647)
                    ),
                    2147483647
                ) AS imported_best_rank
            FROM mentions m
            JOIN runs r ON r.id = m.run_id
            LEFT JOIN gold_hits ON gold_hits.mention_id = m.id
            WHERE m.id = %s
            """,
            (mention_id, mention_id),
        ).fetchone()
        _row_or_404(mention, "Mention")
        gold = conn.execute(
            """
            SELECT qid, ordinal, is_primary, raw_entity
            FROM gold_qids
            WHERE mention_id = %s
            ORDER BY ordinal
            """,
            (mention_id,),
        ).fetchall()
        candidates = conn.execute(
            """
            SELECT id, rank, source_rank, qid, label, item_category, coarse_type, fine_type,
                   retrieval_system, retrieval_stage, retrieval_stages, score, es_score,
                   heuristic_score, selected, gold_match, raw_payload
            FROM candidates
            WHERE mention_id = %s
            ORDER BY rank
            """,
            (mention_id,),
        ).fetchall()
        feedback = conn.execute(
            """
            SELECT id, created_at, category, note, metadata
            FROM feedback_notes
            WHERE mention_id = %s
            ORDER BY created_at DESC
            """,
            (mention_id,),
        ).fetchall()
        attempts = conn.execute(
            """
            SELECT id, created_at, candidate_count, query_text, human_guidance, covered,
                   covered_qids, request_payload, response_payload, error
            FROM live_attempts
            WHERE mention_id = %s
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (mention_id,),
        ).fetchall()
        attempt_rows = []
        for attempt in attempts:
            attempt_row = dict(attempt)
            response_payload = attempt_row.get("response_payload")
            if isinstance(response_payload, dict):
                candidates = response_payload.get("candidates")
                if isinstance(candidates, list):
                    covered_qids = {str(qid) for qid in attempt_row.get("covered_qids") or []}
                    attempt_row["candidates"] = [
                        {
                            **candidate,
                            "gold_match": bool(candidate.get("gold_match"))
                            or bool(candidate.get("qid") and str(candidate.get("qid")) in covered_qids),
                        }
                        for candidate in candidates
                        if isinstance(candidate, dict)
                    ]
            attempt_rows.append(attempt_row)
    return {
        "mention": mention,
        "gold_qids": list(gold),
        "candidates": list(candidates),
        "feedback": list(feedback),
        "live_attempts": attempt_rows,
        "table_context": _table_context_from_payload(mention.get("raw_payload")),
    }


@app.get("/api/mentions/{mention_id}/gold-metadata")
def mention_gold_metadata(mention_id: int) -> dict[str, Any]:
    with connect() as conn:
        _row_or_404(conn.execute("SELECT id FROM mentions WHERE id = %s", (mention_id,)).fetchone(), "Mention")
        gold_rows = conn.execute(
            "SELECT qid FROM gold_qids WHERE mention_id = %s ORDER BY ordinal",
            (mention_id,),
        ).fetchall()

    qids = [str(row["qid"]) for row in gold_rows if row.get("qid")]
    if not qids:
        raise HTTPException(status_code=404, detail="No gold QIDs for this mention")

    try:
        response = alpaca_search(_qid_query_body(qids), len(qids))
        hits = extract_hits(response)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    by_qid = {str(candidate.get("qid")): candidate for candidate in hits if candidate.get("qid")}
    for qid in qids:
        if qid in by_qid:
            return {
                "requested_qids": qids,
                "resolved_qid": qid,
                "entity": by_qid[qid],
                "all_found_qids": [str(candidate.get("qid")) for candidate in hits if candidate.get("qid")],
            }

    return {
        "requested_qids": qids,
        "resolved_qid": None,
        "entity": None,
        "all_found_qids": [],
    }


@app.post("/api/mentions/{mention_id}/feedback")
def create_feedback(mention_id: int, request: FeedbackRequest) -> dict[str, Any]:
    with connect() as conn:
        _row_or_404(conn.execute("SELECT id FROM mentions WHERE id = %s", (mention_id,)).fetchone(), "Mention")
        row = conn.execute(
            """
            INSERT INTO feedback_notes (mention_id, category, note, metadata)
            VALUES (%s, %s, %s, %s)
            RETURNING id, created_at, category, note, metadata
            """,
            (mention_id, request.category, request.note, Jsonb(request.metadata)),
        ).fetchone()
        conn.commit()
    return _row_or_404(row, "Feedback")


@app.post("/api/mentions/{mention_id}/live-attempt")
def live_attempt(mention_id: int, request: LiveAttemptRequest) -> dict[str, Any]:
    retrieval_count = bounded_candidate_count(request.candidate_count)
    returned_count = bounded_returned_candidate_count(request.candidate_count)
    with connect() as conn:
        mention = conn.execute(
            """
            SELECT id, mention, lookup_text, raw_payload
            FROM mentions
            WHERE id = %s
            """,
            (mention_id,),
        ).fetchone()
        _row_or_404(mention, "Mention")
        gold_rows = conn.execute(
            "SELECT qid, raw_entity FROM gold_qids WHERE mention_id = %s ORDER BY ordinal",
            (mention_id,),
        ).fetchall()
        gold_qids = {str(row["qid"]) for row in gold_rows}

        query_text = (request.query_text or mention.get("lookup_text") or mention.get("mention") or "").strip()
        if not query_text:
            raise HTTPException(status_code=400, detail="Provide query_text for the live attempt")
        table_context = _table_context_from_payload(mention.get("raw_payload"))
        query_plan, query_plan_source, query_plan_error, llm_inspection = _llm_es_query_plan(
            query_text=query_text,
            table_context=table_context,
            human_guidance=request.human_guidance,
            llm_config=request.llm_config,
        )
        context_text = _target_context_text(table_context)
        query_body = build_alpaca_query(
            query_plan.get("optimized_query") or query_text,
            {
                **query_plan,
                "context_text": context_text,
            },
        )
        alpaca_request_payload = {
            **query_body,
            "size": retrieval_count,
            "_inspection": {
                "endpoint": ALPACA_METADATA_URL,
                "method": "POST",
                "candidate_count": retrieval_count,
                "optimized_query": query_plan.get("optimized_query"),
                "retrieval_strategy": "single_query_recall_soft_boosts",
                "hard_filters": {"item_category": ["ENTITY", "TYPE"]},
                "soft_signals": {
                    "coarse_type": query_plan.get("coarse_type"),
                    "fine_type": query_plan.get("fine_type"),
                    "typo_corrected_mention": query_plan.get("typo_corrected_mention"),
                    "typo_correction_confidence": query_plan.get("typo_correction_confidence"),
                    "typo_correction_applied": query_plan.get("typo_correction_applied"),
                    "typo_correction_reason": query_plan.get("typo_correction_reason"),
                    "wikipedia_url": query_plan.get("wikipedia_url"),
                    "dbpedia_url": query_plan.get("dbpedia_url"),
                    "aliases": query_plan.get("aliases"),
                    "context_expansion_terms": query_plan.get("context_expansion_terms"),
                },
            },
        }

        error: str | None = None
        response_payload: dict[str, Any] = {}
        raw_candidates: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        covered_qids: list[str] = []
        diagnostics: dict[str, Any] = {}
        try:
            response_payload = alpaca_search(query_body, retrieval_count)
            raw_candidates = extract_hits(response_payload)
            candidate_qids = {str(candidate.get("qid")) for candidate in raw_candidates if candidate.get("qid")}
            covered_qids = sorted(gold_qids & candidate_qids)
            for candidate in raw_candidates:
                candidate["gold_match"] = bool(candidate.get("qid") and str(candidate.get("qid")) in gold_qids)
            diagnostics = _build_improvement_diagnostics(
                mention=mention,
                query_text=query_text,
                human_guidance=request.human_guidance,
                gold_rows=list(gold_rows),
                candidates=raw_candidates,
                covered_qids=covered_qids,
            )
            candidates = raw_candidates[:returned_count]
        except Exception as exc:
            error = str(exc)

        row = conn.execute(
            """
            INSERT INTO live_attempts (
                mention_id, candidate_count, query_text, human_guidance, covered, covered_qids,
                request_payload, response_payload, error
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at, candidate_count, query_text, human_guidance, covered,
                      covered_qids, request_payload, response_payload, error
            """,
            (
                mention_id,
                retrieval_count,
                query_text,
                request.human_guidance,
                bool(covered_qids),
                covered_qids,
                Jsonb(
                    {
                        "alpaca_candidate_fetch": alpaca_request_payload,
                        "llm_query_plan": llm_inspection,
                    }
                ),
                Jsonb(
                    {
                        "candidates": candidates,
                        "raw": response_payload,
                        "retrieval_count": retrieval_count,
                        "returned_count": len(candidates),
                        "search_text": query_plan.get("optimized_query") or query_text,
                        "query_plan": query_plan,
                        "query_plan_terms": query_plan.get("context_expansion_terms", []),
                        "query_plan_source": query_plan_source,
                        "query_plan_error": query_plan_error,
                        "augmentation_terms": query_plan.get("context_expansion_terms", []),
                        "augmentation_source": query_plan_source,
                        "augmentation_error": query_plan_error,
                        "llm_query_plan": llm_inspection,
                        "alpaca_candidate_fetch": {
                            "endpoint": ALPACA_METADATA_URL,
                            "request_body": alpaca_request_payload,
                        },
                        "table_context_used": table_context,
                        "retrieval_trace": {
                            "input": {
                                "mention": mention.get("mention"),
                                "lookup_text": mention.get("lookup_text"),
                                "row_context": table_context.get("rows", []),
                                "column_context": table_context.get("header_cell"),
                                "table_context": table_context,
                                "original_coarse_type": None,
                                "original_fine_type": None,
                            },
                            "llm_prompt": (llm_inspection.get("request_body") or {}).get("messages"),
                            "llm_response": query_plan,
                            "elasticsearch_request_body": query_body,
                            "elasticsearch_results": {
                                "retrieved_candidates": [
                                    {
                                        "rank": candidate.get("rank"),
                                        "qid": candidate.get("qid"),
                                        "label": candidate.get("label"),
                                        "es_score": candidate.get("es_score"),
                                        "gold_match": candidate.get("gold_match"),
                                    }
                                    for candidate in raw_candidates
                                ],
                                "selected_candidate": candidates[0] if candidates else None,
                            },
                        },
                        "improvement_diagnostics": diagnostics,
                    }
                ),
                error,
            ),
        ).fetchone()
        conn.commit()

    result = _row_or_404(row, "Live attempt")
    result["candidates"] = candidates
    if error:
        return JSONResponse(status_code=503, content=jsonable_encoder({"detail": result}))
    return result
