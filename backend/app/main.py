from __future__ import annotations

import json
import os
import re
import time
from json import JSONDecodeError
from collections import Counter
from pathlib import Path
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
from .experiment_runner import (
    DEFAULT_CEREBRAS_CHAT_URL,
    DEFAULT_LLM_CHAT_URL,
    cancel_experiment_job,
    create_experiment_job,
    default_experiment_config,
    normalize_experiment_config,
)
from .llm_estimation import estimate_experiment_llm_usage, estimate_llm_usage, llm_usage_cost_from_metadata
from .source_loader import requested_source_datasets, seed_source_data
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
    normalize_query_plan_source,
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


class SourceDiscoveryRequest(BaseModel):
    source_root: str | None = Field(default=None, max_length=2000)
    requested_datasets: list[str] | None = None
    force: bool = False


class LlmEstimateRequest(BaseModel):
    input: str = Field(default="", max_length=200000)
    model: str | None = None
    max_completion_tokens: int | None = Field(default=None, ge=0, le=200000)
    config: dict[str, Any] = Field(default_factory=dict)


class LlmTestRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


class ExperimentEstimateRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


SENSITIVE_CONFIG_KEYS = {"llm_api_key", "openrouter_api_key", "cerebras_api_key", "api_key"}


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


def _extract_llm_response_tasks(metadata: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(metadata, dict):
        return [], None
    tasks = metadata.get("tasks")
    if isinstance(tasks, list) and tasks:
        return [task for task in tasks if isinstance(task, dict)], None
    parsed = metadata.get("parsed_response")
    if isinstance(parsed, dict) and isinstance(parsed.get("tasks"), list) and parsed.get("tasks"):
        return [task for task in parsed["tasks"] if isinstance(task, dict)], None
    raw = metadata.get("response_content")
    if not isinstance(raw, str) or not raw.strip() or raw.strip() == "None":
        return [], None
    try:
        parsed_raw = json.loads(raw)
    except JSONDecodeError as exc:
        return [], f"Stored LLM answer is not valid JSON: {exc.msg} at character {exc.pos}"
    if isinstance(parsed_raw, dict) and isinstance(parsed_raw.get("tasks"), list):
        return [task for task in parsed_raw["tasks"] if isinstance(task, dict)], None
    return [], "Stored LLM answer did not contain a tasks array"


def _compact_llm_task(task: Any) -> dict[str, Any]:
    if not isinstance(task, dict):
        return {}
    return {
        "id": task.get("id"),
        "optimized_query": task.get("optimized_query"),
        "normalized_mention": task.get("normalized_mention"),
        "typo_corrected_mention": task.get("typo_corrected_mention"),
        "coarse_type": task.get("coarse_type"),
        "fine_type": task.get("fine_type"),
        "wikipedia_url": task.get("wikipedia_url"),
        "dbpedia_url": task.get("dbpedia_url"),
        "aliases": task.get("aliases") if isinstance(task.get("aliases"), list) else [],
        "context_expansion_terms": task.get("context_expansion_terms")
        if isinstance(task.get("context_expansion_terms"), list)
        else [],
    }


def _llm_batch_explanation(item: dict[str, Any]) -> str:
    if item.get("error"):
        return "The LLM request failed before producing a usable batch."
    if item.get("response_parse_error"):
        return "The LLM answered, but the stored answer could not be parsed as valid task JSON."
    requested = int(item.get("task_count") or 0)
    returned = int(item.get("returned_task_count") or 0)
    matched = int(item.get("matched_returned_task_count") or 0)
    usable = int(item.get("usable_task_count") or 0)
    missing = int(item.get("missing_task_count") or 0)
    unknown = int(item.get("unknown_returned_task_count") or 0)
    if missing <= 0 and usable >= requested:
        return "All requested tasks produced usable query plans."
    if returned > 0:
        return (
            f"The LLM returned {returned}/{requested} task objects; {matched} matched requested IDs, "
            f"{unknown} used unknown IDs, and {missing} requested tasks did not become usable query plans."
        )
    return f"No usable LLM task objects were stored for this batch; {missing}/{requested} requested tasks need attention."


def _task_troubleshooting_flags(plan: dict[str, Any], mention: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    plan_source = str(plan.get("query_plan_source") or "heuristic")
    retrieval_error = str(mention.get("retrieval_error") or "").strip()
    candidate_count: int | None = None
    if mention.get("candidate_count") is not None:
        candidate_count = int(mention.get("candidate_count") or 0)
    if plan_source == "heuristic":
        flags.append("heuristic_plan")
    if plan.get("query_plan_error"):
        flags.append("llm_plan_error")
    if retrieval_error:
        flags.append("alpaca_error")
    if candidate_count == 0:
        flags.append("zero_candidates")
    if plan_source == "heuristic" and (retrieval_error or candidate_count == 0):
        flags.append("heuristic_retrieval_problem")
    return flags


def _numeric(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


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


def _candidates_from_cache_payload(response_payload: Any, gold_qids: set[str] | None = None) -> list[dict[str, Any]]:
    if not isinstance(response_payload, dict):
        return []
    candidates = extract_hits(response_payload)
    gold_set = gold_qids or set()
    for candidate in candidates:
        qid = candidate.get("qid")
        candidate["gold_match"] = bool(qid and str(qid) in gold_set)
    return candidates


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


def _normalize_fine_type(raw_fine_type: Any) -> tuple[str | None, str | None, dict[str, Any] | None]:
    original = str(raw_fine_type or "").strip().upper() or None
    if original == "US_STATE":
        return original, "REGION", {
            "rule": "fine_type_us_state_to_region",
            "field": "fine_type",
            "original_value": "US_STATE",
            "normalized_value": "REGION",
            "reason": "US_STATE is not used as a distinct retrieval fine type; REGION is the supported broader type.",
        }
    return original, original, None


def normalize_query_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _llm_api_key(config: dict[str, Any]) -> str:
    provider = str(config.get("llm_provider") or "").strip().lower()
    if provider == "cerebras":
        return str(
            config.get("cerebras_api_key")
            or config.get("llm_api_key")
            or os.environ.get("CEREBRAS_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        ).strip()
    if provider == "openrouter":
        return str(
            config.get("openrouter_api_key")
            or config.get("llm_api_key")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        ).strip()
    return str(config.get("llm_api_key") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "").strip()


def _llm_endpoint(config: dict[str, Any]) -> str:
    provider = str(config.get("llm_provider") or "").strip().lower()
    default_url = DEFAULT_CEREBRAS_CHAT_URL if provider == "cerebras" else DEFAULT_LLM_CHAT_URL
    provider_env_url = os.environ.get("CEREBRAS_CHAT_URL" if provider == "cerebras" else "OPENROUTER_CHAT_URL")
    raw = str(config.get("llm_api_url") or os.environ.get("LLM_API_URL") or provider_env_url or default_url).strip()
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
    elif config.get("llm_provider") == "cerebras" and config.get("llm_reasoning_effort"):
        body["reasoning_effort"] = config["llm_reasoning_effort"]
    if config.get("llm_max_tokens") is not None:
        token_key = "max_completion_tokens" if config.get("llm_provider") == "cerebras" else "max_tokens"
        body[token_key] = int(config["llm_max_tokens"])
    return body


def _llm_headers(config: dict[str, Any], token: str) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "coverage-dashboard/1.0",
    }
    if config.get("llm_provider") == "openrouter":
        headers["HTTP-Referer"] = config["llm_site_url"]
        headers["X-Title"] = config["llm_app_name"]
    return headers


def _llm_json_request(messages: list[dict[str, str]], llm_config: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = normalize_experiment_config(llm_config or {})
    token = _llm_api_key(config)
    if not token:
        raise RuntimeError("LLM API key is not configured")
    endpoint = _llm_endpoint(config)
    body = _llm_request_body(messages, config)

    timeout = int(config["llm_timeout_seconds"])
    max_retries = min(5, max(1, int(config["llm_max_retries"])))
    retry_base_seconds = max(0.5, float(os.environ.get("LLM_RETRY_BASE_SECONDS", os.environ.get("OPENROUTER_RETRY_BASE_SECONDS", "4"))))
    last_error: Exception | None = None
    payload: dict[str, Any] = {}
    for attempt in range(1, max_retries + 1):
        request = Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=_llm_headers(config, token),
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read() or b"{}")
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"{_llm_label(config)} HTTP {exc.code}: {detail[:500]}")
            retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
            if retryable and attempt < max_retries:
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
        "usage_cost": llm_usage_cost_from_metadata(
            provider=str(config.get("llm_provider") or ""),
            model=str(payload.get("model") or config.get("llm_model") or ""),
            usage=payload.get("usage"),
            config=config,
            input_text=json.dumps(body.get("messages") or [], ensure_ascii=False),
            output_text=str(content or ""),
        ),
        "response_model": payload.get("model"),
        "response_provider": payload.get("provider"),
        "response_id": payload.get("id"),
        "model_verified": payload.get("model") in (None, config["llm_model"]),
        "fallbacks_allowed": config.get("llm_provider") != "openrouter",
    }


def _llm_plain_request(messages: list[dict[str, str]], llm_config: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    config = normalize_experiment_config(llm_config or {})
    token = _llm_api_key(config)
    if not token:
        raise RuntimeError("LLM API key is not configured")
    endpoint = _llm_endpoint(config)
    body: dict[str, Any] = {
        "model": config["llm_model"],
        "temperature": 0,
        "messages": messages,
    }
    if config.get("llm_provider") == "openrouter":
        provider_config = _openrouter_provider_config(config)
        if provider_config:
            body["provider"] = provider_config
    if config.get("llm_max_tokens") is not None:
        token_key = "max_completion_tokens" if config.get("llm_provider") == "cerebras" else "max_tokens"
        body[token_key] = int(config["llm_max_tokens"])

    request = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=_llm_headers(config, token),
        method="POST",
    )
    try:
        with urlopen(request, timeout=int(config["llm_timeout_seconds"])) as response:
            payload = json.loads(response.read() or b"{}")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{_llm_label(config)} HTTP {exc.code}: {detail[:500]}") from exc
    except (URLError, TimeoutError, JSONDecodeError) as exc:
        raise RuntimeError(f"{_llm_label(config)} test call failed: {exc}") from exc

    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    return str(content or ""), {
        "sent": True,
        "provider": config.get("llm_provider"),
        "endpoint": endpoint,
        "request_body": body,
        "response_usage": payload.get("usage"),
        "usage_cost": llm_usage_cost_from_metadata(
            provider=str(config.get("llm_provider") or ""),
            model=str(payload.get("model") or config.get("llm_model") or ""),
            usage=payload.get("usage"),
            config=config,
            input_text=json.dumps(body.get("messages") or [], ensure_ascii=False),
            output_text=str(content or ""),
        ),
        "response_model": payload.get("model"),
        "response_provider": payload.get("provider"),
        "response_id": payload.get("id"),
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
    original_fine_type, fine_type, fine_type_rule = _normalize_fine_type(raw_plan.get("fine_type"))
    return {
        "optimized_query": optimized_query or fallback_query,
        "normalized_mention": normalized_mention,
        "typo_corrected_mention": typo_corrected_mention,
        "typo_correction_confidence": typo_correction_confidence,
        "typo_correction_applied": use_typo_correction,
        "typo_correction_reason": str(raw_plan.get("typo_correction_reason") or "").strip() or None,
        "coarse_type": str(raw_plan.get("coarse_type") or "").strip().upper() or None,
        "fine_type": fine_type,
        "original_fine_type": original_fine_type,
        "fine_type_rule_applied": bool(fine_type_rule),
        "fine_type_normalization": fine_type_rule,
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
            "original_fine_type": None,
            "fine_type_rule_applied": False,
            "fine_type_normalization": None,
            "wikipedia_url": None,
            "dbpedia_url": None,
            "aliases": [],
            "context_expansion_terms": [],
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
                "Use row context, column context, table context, metadata, and optional human guidance only "
                "to infer type hints, URL slugs, aliases, or typo confidence. Keep the optimized query close "
                "to the mention surface: do not copy column headers, neighboring cells, other mentions from "
                "the same row, or broad table metadata into the query. Context relationships are directional "
                "in the index, so do not add a related entity or work title unless it is itself an alias of "
                "the mention. If the mention appears to contain a typo and the "
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
            {"context_expansion_terms": []},
            query_text,
        )
        return plan, "heuristic_query_plan_fallback", str(exc), llm_request


def _qid_query_body(qids: list[str]) -> dict[str, Any]:
    return {
        "_source": True,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"item_category": ["ENTITY", "TYPE"]}},
                    {"terms": {"_id": qids}},
                ]
            }
        }
    }


def _entity_with_ner_types(candidate: dict[str, Any]) -> dict[str, Any]:
    entity = dict(candidate)
    raw_payload = entity.get("raw_payload")
    if isinstance(raw_payload, dict):
        entity.setdefault("coarse_type", raw_payload.get("coarse_type"))
        entity.setdefault("fine_type", raw_payload.get("fine_type"))
    ner_payload = entity.get("ner")
    if isinstance(ner_payload, dict):
        entity.setdefault("coarse_type", ner_payload.get("coarse_type") or ner_payload.get("coarse"))
        entity.setdefault("fine_type", ner_payload.get("fine_type") or ner_payload.get("fine"))
    type_payload = entity.get("type")
    if isinstance(type_payload, dict):
        entity.setdefault("coarse_type", type_payload.get("coarse_type") or type_payload.get("coarse"))
        entity.setdefault("fine_type", type_payload.get("fine_type") or type_payload.get("fine"))
    entity["coarse_type"] = str(entity.get("coarse_type") or "").strip().upper() or None
    entity["fine_type"] = str(entity.get("fine_type") or "").strip().upper() or None
    entity["ner_types"] = {
        "coarse_type": entity.get("coarse_type"),
        "fine_type": entity.get("fine_type"),
    }
    return entity


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
            SET status = CASE WHEN status = 'cancel_requested' THEN 'cancelled' ELSE 'failed' END,
                stage = CASE WHEN status = 'cancel_requested' THEN 'cancelled' ELSE 'failed' END,
                message = CASE WHEN status = 'cancel_requested' THEN 'Cancelled during API restart' ELSE 'API restarted while this job was active' END,
                error = CASE WHEN status = 'cancel_requested' THEN NULL ELSE coalesce(error, 'API restarted while this job was active') END,
                finished_at = now()
            WHERE status IN ('queued', 'running', 'cancel_requested')
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


@app.post("/api/source-datasets/discover")
def discover_source_datasets(request: SourceDiscoveryRequest) -> dict[str, Any]:
    source_root = Path(request.source_root or os.environ.get("SOURCE_DATA_ROOT", "/source-data"))
    requested = [str(item).strip() for item in request.requested_datasets or [] if str(item).strip()]
    if not requested:
        requested = requested_source_datasets(source_root)
    try:
        result = seed_source_data(source_root=source_root, requested_datasets=requested, force=request.force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    with connect() as conn:
        inventory = source_dataset_inventory(conn)
    return {
        **result,
        "source_root": str(source_root),
        "requested_datasets": requested,
        "inventory": inventory,
    }


@app.get("/api/experiment-defaults")
def experiment_defaults() -> dict[str, Any]:
    return default_experiment_config()


@app.get("/api/config-status")
def config_status() -> dict[str, Any]:
    defaults = default_experiment_config()
    llm_key_configured = bool(
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
        or os.environ.get("CEREBRAS_API_KEY", "").strip()
    )
    return {
        "alpaca_configured": bool(alpaca_token()),
        "llm_configured": llm_key_configured,
        "llm_provider": defaults["llm_provider"],
        "llm_provider_name": defaults["llm_provider_name"],
        "llm_api_url": defaults["llm_api_url"],
        "llm_model": defaults["llm_model"],
        "openrouter_configured": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
        "cerebras_configured": bool(os.environ.get("CEREBRAS_API_KEY", "").strip()),
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


@app.post("/api/llm/test")
def llm_test(request: LlmTestRequest) -> dict[str, Any]:
    config = normalize_experiment_config({**request.config})
    config["llm_enabled"] = True
    if not config.get("llm_max_tokens"):
        config["llm_max_tokens"] = 16
    messages = [
        {
            "role": "user",
            "content": "Reply with one short sentence.",
        },
    ]
    try:
        content, llm_request = _llm_plain_request(messages, config)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "ok": bool(content.strip()),
        "provider": config.get("llm_provider"),
        "provider_name": config.get("llm_provider_name"),
        "endpoint": llm_request.get("endpoint"),
        "model": config.get("llm_model"),
        "response_model": llm_request.get("response_model"),
        "response_provider": llm_request.get("response_provider"),
        "response_usage": llm_request.get("response_usage"),
        "usage_cost": llm_request.get("usage_cost"),
        "response_id": llm_request.get("response_id"),
        "content": content,
    }


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
            WITH task_stats AS (
                SELECT batch_id,
                       count(*) FILTER (WHERE coalesce(plan_payload->>'query_plan_source', 'heuristic') <> 'heuristic') AS usable_task_count,
                       count(*) FILTER (WHERE coalesce(plan_payload->>'query_plan_source', 'heuristic') = 'heuristic') AS missing_task_count
                FROM llm_prompt_tasks
                GROUP BY batch_id
            ),
            batch_stats AS (
                SELECT b.id, b.job_id, b.run_id, b.provider, b.model, b.status, b.error, b.task_count, b.response_metadata,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'returned_task_count', '')::int, jsonb_array_length(coalesce(b.response_metadata->'tasks', '[]'::jsonb))) AS returned_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'usable_task_count', '')::int, t.usable_task_count, 0) AS usable_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'missing_task_count', '')::int, t.missing_task_count, 0) AS missing_task_count,
                       b.response_metadata->>'parse_warning' AS parse_warning
                FROM llm_prompt_batches b
                LEFT JOIN task_stats t ON t.batch_id = b.id
                WHERE b.prompt_template = 'entity_retrieval_query_plan_v1'
            ),
            llm AS (
                SELECT job_id,
                       count(*) AS query_plan_batch_count,
                       count(*) FILTER (WHERE status = 'completed' AND error IS NULL AND parse_warning IS NULL AND missing_task_count = 0) AS query_plan_completed_count,
                       count(*) FILTER (WHERE status = 'failed' OR error IS NOT NULL OR parse_warning IS NOT NULL OR missing_task_count > 0) AS query_plan_failed_count,
                       count(*) FILTER (WHERE missing_task_count > 0) AS query_plan_incomplete_batch_count,
                       coalesce(sum(task_count), 0)::bigint AS query_plan_requested_task_count,
                       coalesce(sum(returned_task_count), 0)::bigint AS query_plan_returned_task_count,
                       coalesce(sum(usable_task_count), 0)::bigint AS query_plan_usable_task_count,
                       coalesce(sum(missing_task_count), 0)::bigint AS query_plan_missing_task_count,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'prompt_tokens', response_metadata->'usage'->>'input_tokens', response_metadata->'usage_cost'->>'input_tokens', response_metadata->'usage_cost'->>'prompt_tokens'), '')::double precision), 0) AS query_plan_prompt_tokens,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'completion_tokens', response_metadata->'usage'->>'output_tokens', response_metadata->'usage_cost'->>'output_tokens', response_metadata->'usage_cost'->>'completion_tokens'), '')::double precision), 0) AS query_plan_completion_tokens,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'total_tokens', response_metadata->'usage_cost'->>'total_tokens'), '')::double precision), 0) AS query_plan_total_tokens,
                       coalesce(sum(coalesce(nullif(coalesce(response_metadata->'usage'->'cost_details'->>'upstream_inference_cost', nullif(response_metadata->'usage_cost'->>'total_cost_usd', '0'), response_metadata->'usage'->>'total_cost_usd', response_metadata->'usage'->>'cost_usd', nullif(response_metadata->'usage'->>'cost', '0')), '')::double precision, CASE WHEN lower(coalesce(provider, '')) = 'cerebras' AND lower(coalesce(model, '')) IN ('gpt-oss-120b', 'openai/gpt-oss-120b', 'gemma-4-31b', 'zai-glm-4.7') THEN (coalesce(nullif(coalesce(response_metadata->'usage'->>'prompt_tokens', response_metadata->'usage'->>'input_tokens', response_metadata->'usage_cost'->>'input_tokens', response_metadata->'usage_cost'->>'prompt_tokens'), '')::double precision, 0) * CASE lower(coalesce(model, '')) WHEN 'gemma-4-31b' THEN 0.99 WHEN 'zai-glm-4.7' THEN 2.25 ELSE 0.35 END / 1000000.0) + (coalesce(nullif(coalesce(response_metadata->'usage'->>'completion_tokens', response_metadata->'usage'->>'output_tokens', response_metadata->'usage_cost'->>'output_tokens', response_metadata->'usage_cost'->>'completion_tokens'), '')::double precision, 0) * CASE lower(coalesce(model, '')) WHEN 'gemma-4-31b' THEN 1.49 WHEN 'zai-glm-4.7' THEN 2.75 ELSE 0.75 END / 1000000.0) END)), 0) AS query_plan_total_cost_usd,
                       count(*) FILTER (WHERE response_metadata->'usage'->'cost_details'->>'upstream_inference_cost' IS NOT NULL OR response_metadata->'usage_cost'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost' IS NOT NULL OR (lower(coalesce(provider, '')) = 'cerebras' AND lower(coalesce(model, '')) IN ('gpt-oss-120b', 'openai/gpt-oss-120b', 'gemma-4-31b', 'zai-glm-4.7'))) AS query_plan_priced_batch_count,
                       count(*) FILTER (WHERE response_metadata->'usage'->'cost_details'->>'upstream_inference_cost' IS NOT NULL OR response_metadata->'usage_cost'->>'cost_kind' = 'response_reported' OR response_metadata->'usage'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost' IS NOT NULL) AS query_plan_response_reported_cost_count
                FROM batch_stats
                GROUP BY job_id
            )
            SELECT j.*, r.name AS imported_run_name,
                   coalesce(llm.query_plan_batch_count, 0) AS llm_query_plan_batch_count,
                   coalesce(llm.query_plan_completed_count, 0) AS llm_query_plan_completed_count,
                   coalesce(llm.query_plan_failed_count, 0) AS llm_query_plan_failed_count,
                   coalesce(llm.query_plan_incomplete_batch_count, 0) AS llm_query_plan_incomplete_batch_count,
                   coalesce(llm.query_plan_requested_task_count, 0) AS llm_query_plan_requested_task_count,
                   coalesce(llm.query_plan_returned_task_count, 0) AS llm_query_plan_returned_task_count,
                   coalesce(llm.query_plan_usable_task_count, 0) AS llm_query_plan_usable_task_count,
                   coalesce(llm.query_plan_missing_task_count, 0) AS llm_query_plan_missing_task_count,
                   coalesce(llm.query_plan_prompt_tokens, 0) AS llm_query_plan_prompt_tokens,
                   coalesce(llm.query_plan_completion_tokens, 0) AS llm_query_plan_completion_tokens,
                   coalesce(llm.query_plan_total_tokens, 0) AS llm_query_plan_total_tokens,
                   coalesce(llm.query_plan_total_cost_usd, 0) AS llm_query_plan_total_cost_usd,
                   coalesce(llm.query_plan_priced_batch_count, 0) AS llm_query_plan_priced_batch_count,
                   coalesce(llm.query_plan_response_reported_cost_count, 0) AS llm_query_plan_response_reported_cost_count
            FROM experiment_jobs j
            LEFT JOIN runs r ON r.id = j.imported_run_id
            LEFT JOIN llm ON llm.job_id = j.id
            ORDER BY j.created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return [_redact_job(dict(row)) for row in rows]


@app.delete("/api/experiment-jobs/failed")
def clear_failed_experiment_jobs() -> dict[str, int]:
    with connect() as conn:
        result = conn.execute("DELETE FROM experiment_jobs WHERE status IN ('failed', 'cancelled')")
        conn.commit()
    return {"deleted": result.rowcount or 0}


@app.delete("/api/experiment-jobs/{job_id}")
def delete_experiment_job(job_id: int) -> dict[str, Any]:
    with connect() as conn:
        with conn.transaction():
            row = conn.execute(
                """
                SELECT id, status, imported_run_id
                FROM experiment_jobs
                WHERE id = %s
                """,
                (job_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Experiment job not found")
            if row["status"] in {"queued", "running", "cancel_requested"}:
                raise HTTPException(status_code=409, detail="Cancel the active job before deleting it")
            conn.execute("DELETE FROM llm_prompt_batches WHERE job_id = %s AND run_id IS NULL", (job_id,))
            conn.execute("UPDATE llm_prompt_batches SET job_id = NULL WHERE job_id = %s", (job_id,))
            deleted = conn.execute(
                """
                DELETE FROM experiment_jobs
                WHERE id = %s
                RETURNING id, status, imported_run_id
                """,
                (job_id,),
            ).fetchone()
    return {"deleted": True, "id": deleted["id"], "status": deleted["status"], "imported_run_id": deleted["imported_run_id"]}


@app.get("/api/experiment-jobs/{job_id}")
def experiment_job(job_id: int) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute(
            """
            WITH task_stats AS (
                SELECT batch_id,
                       count(*) FILTER (WHERE coalesce(plan_payload->>'query_plan_source', 'heuristic') <> 'heuristic') AS usable_task_count,
                       count(*) FILTER (WHERE coalesce(plan_payload->>'query_plan_source', 'heuristic') = 'heuristic') AS missing_task_count
                FROM llm_prompt_tasks
                GROUP BY batch_id
            ),
            batch_stats AS (
                SELECT b.id, b.job_id, b.provider, b.model, b.status, b.error, b.task_count, b.response_metadata,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'returned_task_count', '')::int, jsonb_array_length(coalesce(b.response_metadata->'tasks', '[]'::jsonb))) AS returned_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'usable_task_count', '')::int, t.usable_task_count, 0) AS usable_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'missing_task_count', '')::int, t.missing_task_count, 0) AS missing_task_count,
                       b.response_metadata->>'parse_warning' AS parse_warning
                FROM llm_prompt_batches b
                LEFT JOIN task_stats t ON t.batch_id = b.id
                WHERE b.prompt_template = 'entity_retrieval_query_plan_v1'
            ),
            llm AS (
                SELECT job_id,
                       count(*) AS query_plan_batch_count,
                       count(*) FILTER (WHERE status = 'completed' AND error IS NULL AND parse_warning IS NULL AND missing_task_count = 0) AS query_plan_completed_count,
                       count(*) FILTER (WHERE status = 'failed' OR error IS NOT NULL OR parse_warning IS NOT NULL OR missing_task_count > 0) AS query_plan_failed_count,
                       count(*) FILTER (WHERE missing_task_count > 0) AS query_plan_incomplete_batch_count,
                       coalesce(sum(task_count), 0)::bigint AS query_plan_requested_task_count,
                       coalesce(sum(returned_task_count), 0)::bigint AS query_plan_returned_task_count,
                       coalesce(sum(usable_task_count), 0)::bigint AS query_plan_usable_task_count,
                       coalesce(sum(missing_task_count), 0)::bigint AS query_plan_missing_task_count,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'prompt_tokens', response_metadata->'usage'->>'input_tokens', response_metadata->'usage_cost'->>'input_tokens', response_metadata->'usage_cost'->>'prompt_tokens'), '')::double precision), 0) AS query_plan_prompt_tokens,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'completion_tokens', response_metadata->'usage'->>'output_tokens', response_metadata->'usage_cost'->>'output_tokens', response_metadata->'usage_cost'->>'completion_tokens'), '')::double precision), 0) AS query_plan_completion_tokens,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'total_tokens', response_metadata->'usage_cost'->>'total_tokens'), '')::double precision), 0) AS query_plan_total_tokens,
                       coalesce(sum(coalesce(nullif(coalesce(response_metadata->'usage'->'cost_details'->>'upstream_inference_cost', nullif(response_metadata->'usage_cost'->>'total_cost_usd', '0'), response_metadata->'usage'->>'total_cost_usd', response_metadata->'usage'->>'cost_usd', nullif(response_metadata->'usage'->>'cost', '0')), '')::double precision, CASE WHEN lower(coalesce(provider, '')) = 'cerebras' AND lower(coalesce(model, '')) IN ('gpt-oss-120b', 'openai/gpt-oss-120b', 'gemma-4-31b', 'zai-glm-4.7') THEN (coalesce(nullif(coalesce(response_metadata->'usage'->>'prompt_tokens', response_metadata->'usage'->>'input_tokens', response_metadata->'usage_cost'->>'input_tokens', response_metadata->'usage_cost'->>'prompt_tokens'), '')::double precision, 0) * CASE lower(coalesce(model, '')) WHEN 'gemma-4-31b' THEN 0.99 WHEN 'zai-glm-4.7' THEN 2.25 ELSE 0.35 END / 1000000.0) + (coalesce(nullif(coalesce(response_metadata->'usage'->>'completion_tokens', response_metadata->'usage'->>'output_tokens', response_metadata->'usage_cost'->>'output_tokens', response_metadata->'usage_cost'->>'completion_tokens'), '')::double precision, 0) * CASE lower(coalesce(model, '')) WHEN 'gemma-4-31b' THEN 1.49 WHEN 'zai-glm-4.7' THEN 2.75 ELSE 0.75 END / 1000000.0) END)), 0) AS query_plan_total_cost_usd,
                       count(*) FILTER (WHERE response_metadata->'usage'->'cost_details'->>'upstream_inference_cost' IS NOT NULL OR response_metadata->'usage_cost'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost' IS NOT NULL OR (lower(coalesce(provider, '')) = 'cerebras' AND lower(coalesce(model, '')) IN ('gpt-oss-120b', 'openai/gpt-oss-120b', 'gemma-4-31b', 'zai-glm-4.7'))) AS query_plan_priced_batch_count,
                       count(*) FILTER (WHERE response_metadata->'usage'->'cost_details'->>'upstream_inference_cost' IS NOT NULL OR response_metadata->'usage_cost'->>'cost_kind' = 'response_reported' OR response_metadata->'usage'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost' IS NOT NULL) AS query_plan_response_reported_cost_count
                FROM batch_stats
                GROUP BY job_id
            )
            SELECT j.*, r.name AS imported_run_name,
                   coalesce(llm.query_plan_batch_count, 0) AS llm_query_plan_batch_count,
                   coalesce(llm.query_plan_completed_count, 0) AS llm_query_plan_completed_count,
                   coalesce(llm.query_plan_failed_count, 0) AS llm_query_plan_failed_count,
                   coalesce(llm.query_plan_incomplete_batch_count, 0) AS llm_query_plan_incomplete_batch_count,
                   coalesce(llm.query_plan_requested_task_count, 0) AS llm_query_plan_requested_task_count,
                   coalesce(llm.query_plan_returned_task_count, 0) AS llm_query_plan_returned_task_count,
                   coalesce(llm.query_plan_usable_task_count, 0) AS llm_query_plan_usable_task_count,
                   coalesce(llm.query_plan_missing_task_count, 0) AS llm_query_plan_missing_task_count,
                   coalesce(llm.query_plan_prompt_tokens, 0) AS llm_query_plan_prompt_tokens,
                   coalesce(llm.query_plan_completion_tokens, 0) AS llm_query_plan_completion_tokens,
                   coalesce(llm.query_plan_total_tokens, 0) AS llm_query_plan_total_tokens,
                   coalesce(llm.query_plan_total_cost_usd, 0) AS llm_query_plan_total_cost_usd,
                   coalesce(llm.query_plan_priced_batch_count, 0) AS llm_query_plan_priced_batch_count,
                   coalesce(llm.query_plan_response_reported_cost_count, 0) AS llm_query_plan_response_reported_cost_count
            FROM experiment_jobs j
            LEFT JOIN runs r ON r.id = j.imported_run_id
            LEFT JOIN llm ON llm.job_id = j.id
            WHERE j.id = %s
            """,
            (job_id,),
        ).fetchone()
    return _redact_job(_row_or_404(row, "Experiment job"))


@app.post("/api/experiment-jobs/{job_id}/cancel")
def cancel_job(job_id: int) -> dict[str, Any]:
    try:
        return _redact_job(cancel_experiment_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            WITH task_stats AS (
                SELECT batch_id,
                       count(*) FILTER (WHERE coalesce(plan_payload->>'query_plan_source', 'heuristic') <> 'heuristic') AS usable_task_count,
                       count(*) FILTER (WHERE coalesce(plan_payload->>'query_plan_source', 'heuristic') = 'heuristic') AS missing_task_count
                FROM llm_prompt_tasks
                GROUP BY batch_id
            ),
            batch_stats AS (
                SELECT b.id, b.run_id, b.provider, b.model, b.status, b.error, b.task_count, b.response_metadata,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'returned_task_count', '')::int, jsonb_array_length(coalesce(b.response_metadata->'tasks', '[]'::jsonb))) AS returned_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'usable_task_count', '')::int, t.usable_task_count, 0) AS usable_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'missing_task_count', '')::int, t.missing_task_count, 0) AS missing_task_count,
                       b.response_metadata->>'parse_warning' AS parse_warning
                FROM llm_prompt_batches b
                LEFT JOIN task_stats t ON t.batch_id = b.id
                WHERE b.run_id IS NOT NULL
                  AND b.prompt_template = 'entity_retrieval_query_plan_v1'
            ),
            llm AS (
                SELECT run_id,
                       count(*) AS query_plan_batch_count,
                       count(*) FILTER (WHERE status = 'completed' AND error IS NULL AND parse_warning IS NULL AND missing_task_count = 0) AS query_plan_completed_count,
                       count(*) FILTER (WHERE status = 'failed' OR error IS NOT NULL OR parse_warning IS NOT NULL OR missing_task_count > 0) AS query_plan_failed_count,
                       count(*) FILTER (WHERE missing_task_count > 0) AS query_plan_incomplete_batch_count,
                       coalesce(sum(task_count), 0)::bigint AS query_plan_requested_task_count,
                       coalesce(sum(returned_task_count), 0)::bigint AS query_plan_returned_task_count,
                       coalesce(sum(usable_task_count), 0)::bigint AS query_plan_usable_task_count,
                       coalesce(sum(missing_task_count), 0)::bigint AS query_plan_missing_task_count,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'prompt_tokens', response_metadata->'usage'->>'input_tokens', response_metadata->'usage_cost'->>'input_tokens', response_metadata->'usage_cost'->>'prompt_tokens'), '')::double precision), 0) AS query_plan_prompt_tokens,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'completion_tokens', response_metadata->'usage'->>'output_tokens', response_metadata->'usage_cost'->>'output_tokens', response_metadata->'usage_cost'->>'completion_tokens'), '')::double precision), 0) AS query_plan_completion_tokens,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'total_tokens', response_metadata->'usage_cost'->>'total_tokens'), '')::double precision), 0) AS query_plan_total_tokens,
                       coalesce(sum(coalesce(nullif(coalesce(response_metadata->'usage'->'cost_details'->>'upstream_inference_cost', nullif(response_metadata->'usage_cost'->>'total_cost_usd', '0'), response_metadata->'usage'->>'total_cost_usd', response_metadata->'usage'->>'cost_usd', nullif(response_metadata->'usage'->>'cost', '0')), '')::double precision, CASE WHEN lower(coalesce(provider, '')) = 'cerebras' AND lower(coalesce(model, '')) IN ('gpt-oss-120b', 'openai/gpt-oss-120b', 'gemma-4-31b', 'zai-glm-4.7') THEN (coalesce(nullif(coalesce(response_metadata->'usage'->>'prompt_tokens', response_metadata->'usage'->>'input_tokens', response_metadata->'usage_cost'->>'input_tokens', response_metadata->'usage_cost'->>'prompt_tokens'), '')::double precision, 0) * CASE lower(coalesce(model, '')) WHEN 'gemma-4-31b' THEN 0.99 WHEN 'zai-glm-4.7' THEN 2.25 ELSE 0.35 END / 1000000.0) + (coalesce(nullif(coalesce(response_metadata->'usage'->>'completion_tokens', response_metadata->'usage'->>'output_tokens', response_metadata->'usage_cost'->>'output_tokens', response_metadata->'usage_cost'->>'completion_tokens'), '')::double precision, 0) * CASE lower(coalesce(model, '')) WHEN 'gemma-4-31b' THEN 1.49 WHEN 'zai-glm-4.7' THEN 2.75 ELSE 0.75 END / 1000000.0) END)), 0) AS query_plan_total_cost_usd,
                       count(*) FILTER (WHERE response_metadata->'usage'->'cost_details'->>'upstream_inference_cost' IS NOT NULL OR response_metadata->'usage_cost'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost' IS NOT NULL OR (lower(coalesce(provider, '')) = 'cerebras' AND lower(coalesce(model, '')) IN ('gpt-oss-120b', 'openai/gpt-oss-120b', 'gemma-4-31b', 'zai-glm-4.7'))) AS query_plan_priced_batch_count,
                       count(*) FILTER (WHERE response_metadata->'usage'->'cost_details'->>'upstream_inference_cost' IS NOT NULL OR response_metadata->'usage_cost'->>'cost_kind' = 'response_reported' OR response_metadata->'usage'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost' IS NOT NULL) AS query_plan_response_reported_cost_count
                FROM batch_stats
                GROUP BY run_id
            )
            SELECT r.id, r.name, r.source_path, r.source_filename, r.imported_at, r.table_count,
                   r.mention_count, r.candidate_count, r.covered_count,
                   CASE WHEN r.mention_count > 0 THEN r.covered_count::float / r.mention_count ELSE 0 END AS imported_coverage,
                   coalesce(llm.query_plan_batch_count, 0) AS llm_query_plan_batch_count,
                   coalesce(llm.query_plan_completed_count, 0) AS llm_query_plan_completed_count,
                   coalesce(llm.query_plan_failed_count, 0) AS llm_query_plan_failed_count,
                   coalesce(llm.query_plan_incomplete_batch_count, 0) AS llm_query_plan_incomplete_batch_count,
                   coalesce(llm.query_plan_requested_task_count, 0) AS llm_query_plan_requested_task_count,
                   coalesce(llm.query_plan_returned_task_count, 0) AS llm_query_plan_returned_task_count,
                   coalesce(llm.query_plan_usable_task_count, 0) AS llm_query_plan_usable_task_count,
                   coalesce(llm.query_plan_missing_task_count, 0) AS llm_query_plan_missing_task_count,
                   coalesce(llm.query_plan_prompt_tokens, 0) AS llm_query_plan_prompt_tokens,
                   coalesce(llm.query_plan_completion_tokens, 0) AS llm_query_plan_completion_tokens,
                   coalesce(llm.query_plan_total_tokens, 0) AS llm_query_plan_total_tokens,
                   coalesce(llm.query_plan_total_cost_usd, 0) AS llm_query_plan_total_cost_usd,
                   coalesce(llm.query_plan_priced_batch_count, 0) AS llm_query_plan_priced_batch_count,
                   coalesce(llm.query_plan_response_reported_cost_count, 0) AS llm_query_plan_response_reported_cost_count
            FROM runs r
            LEFT JOIN llm ON llm.run_id = r.id
            ORDER BY r.imported_at DESC
            """
        ).fetchall()
    return list(rows)


@app.get("/api/runs/{run_id}")
def run_detail(run_id: int) -> dict[str, Any]:
    with connect() as conn:
        run = conn.execute(
            """
            WITH task_stats AS (
                SELECT batch_id,
                       count(*) FILTER (WHERE coalesce(plan_payload->>'query_plan_source', 'heuristic') <> 'heuristic') AS usable_task_count,
                       count(*) FILTER (WHERE coalesce(plan_payload->>'query_plan_source', 'heuristic') = 'heuristic') AS missing_task_count
                FROM llm_prompt_tasks
                GROUP BY batch_id
            ),
            batch_stats AS (
                SELECT b.id, b.run_id, b.provider, b.model, b.status, b.error, b.task_count, b.response_metadata,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'returned_task_count', '')::int, jsonb_array_length(coalesce(b.response_metadata->'tasks', '[]'::jsonb))) AS returned_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'usable_task_count', '')::int, t.usable_task_count, 0) AS usable_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'missing_task_count', '')::int, t.missing_task_count, 0) AS missing_task_count,
                       b.response_metadata->>'parse_warning' AS parse_warning
                FROM llm_prompt_batches b
                LEFT JOIN task_stats t ON t.batch_id = b.id
                WHERE b.run_id IS NOT NULL
                  AND b.prompt_template = 'entity_retrieval_query_plan_v1'
            ),
            llm AS (
                SELECT run_id,
                       count(*) AS query_plan_batch_count,
                       count(*) FILTER (WHERE status = 'completed' AND error IS NULL AND parse_warning IS NULL AND missing_task_count = 0) AS query_plan_completed_count,
                       count(*) FILTER (WHERE status = 'failed' OR error IS NOT NULL OR parse_warning IS NOT NULL OR missing_task_count > 0) AS query_plan_failed_count,
                       count(*) FILTER (WHERE missing_task_count > 0) AS query_plan_incomplete_batch_count,
                       coalesce(sum(task_count), 0)::bigint AS query_plan_requested_task_count,
                       coalesce(sum(returned_task_count), 0)::bigint AS query_plan_returned_task_count,
                       coalesce(sum(usable_task_count), 0)::bigint AS query_plan_usable_task_count,
                       coalesce(sum(missing_task_count), 0)::bigint AS query_plan_missing_task_count,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'prompt_tokens', response_metadata->'usage'->>'input_tokens', response_metadata->'usage_cost'->>'input_tokens', response_metadata->'usage_cost'->>'prompt_tokens'), '')::double precision), 0) AS query_plan_prompt_tokens,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'completion_tokens', response_metadata->'usage'->>'output_tokens', response_metadata->'usage_cost'->>'output_tokens', response_metadata->'usage_cost'->>'completion_tokens'), '')::double precision), 0) AS query_plan_completion_tokens,
                       coalesce(sum(nullif(coalesce(response_metadata->'usage'->>'total_tokens', response_metadata->'usage_cost'->>'total_tokens'), '')::double precision), 0) AS query_plan_total_tokens,
                       coalesce(sum(coalesce(nullif(coalesce(response_metadata->'usage'->'cost_details'->>'upstream_inference_cost', nullif(response_metadata->'usage_cost'->>'total_cost_usd', '0'), response_metadata->'usage'->>'total_cost_usd', response_metadata->'usage'->>'cost_usd', nullif(response_metadata->'usage'->>'cost', '0')), '')::double precision, CASE WHEN lower(coalesce(provider, '')) = 'cerebras' AND lower(coalesce(model, '')) IN ('gpt-oss-120b', 'openai/gpt-oss-120b', 'gemma-4-31b', 'zai-glm-4.7') THEN (coalesce(nullif(coalesce(response_metadata->'usage'->>'prompt_tokens', response_metadata->'usage'->>'input_tokens', response_metadata->'usage_cost'->>'input_tokens', response_metadata->'usage_cost'->>'prompt_tokens'), '')::double precision, 0) * CASE lower(coalesce(model, '')) WHEN 'gemma-4-31b' THEN 0.99 WHEN 'zai-glm-4.7' THEN 2.25 ELSE 0.35 END / 1000000.0) + (coalesce(nullif(coalesce(response_metadata->'usage'->>'completion_tokens', response_metadata->'usage'->>'output_tokens', response_metadata->'usage_cost'->>'output_tokens', response_metadata->'usage_cost'->>'completion_tokens'), '')::double precision, 0) * CASE lower(coalesce(model, '')) WHEN 'gemma-4-31b' THEN 1.49 WHEN 'zai-glm-4.7' THEN 2.75 ELSE 0.75 END / 1000000.0) END)), 0) AS query_plan_total_cost_usd,
                       count(*) FILTER (WHERE response_metadata->'usage'->'cost_details'->>'upstream_inference_cost' IS NOT NULL OR response_metadata->'usage_cost'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost' IS NOT NULL OR (lower(coalesce(provider, '')) = 'cerebras' AND lower(coalesce(model, '')) IN ('gpt-oss-120b', 'openai/gpt-oss-120b', 'gemma-4-31b', 'zai-glm-4.7'))) AS query_plan_priced_batch_count,
                       count(*) FILTER (WHERE response_metadata->'usage'->'cost_details'->>'upstream_inference_cost' IS NOT NULL OR response_metadata->'usage_cost'->>'cost_kind' = 'response_reported' OR response_metadata->'usage'->>'total_cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost_usd' IS NOT NULL OR response_metadata->'usage'->>'cost' IS NOT NULL) AS query_plan_response_reported_cost_count
                FROM batch_stats
                GROUP BY run_id
            )
            SELECT r.id, r.name, r.source_path, r.source_filename, r.imported_at, r.table_count,
                   r.mention_count, r.candidate_count, r.covered_count,
                   CASE WHEN r.mention_count > 0 THEN r.covered_count::float / r.mention_count ELSE 0 END AS imported_coverage,
                   coalesce(llm.query_plan_batch_count, 0) AS llm_query_plan_batch_count,
                   coalesce(llm.query_plan_completed_count, 0) AS llm_query_plan_completed_count,
                   coalesce(llm.query_plan_failed_count, 0) AS llm_query_plan_failed_count,
                   coalesce(llm.query_plan_incomplete_batch_count, 0) AS llm_query_plan_incomplete_batch_count,
                   coalesce(llm.query_plan_requested_task_count, 0) AS llm_query_plan_requested_task_count,
                   coalesce(llm.query_plan_returned_task_count, 0) AS llm_query_plan_returned_task_count,
                   coalesce(llm.query_plan_usable_task_count, 0) AS llm_query_plan_usable_task_count,
                   coalesce(llm.query_plan_missing_task_count, 0) AS llm_query_plan_missing_task_count,
                   coalesce(llm.query_plan_prompt_tokens, 0) AS llm_query_plan_prompt_tokens,
                   coalesce(llm.query_plan_completion_tokens, 0) AS llm_query_plan_completion_tokens,
                   coalesce(llm.query_plan_total_tokens, 0) AS llm_query_plan_total_tokens,
                   coalesce(llm.query_plan_total_cost_usd, 0) AS llm_query_plan_total_cost_usd,
                   coalesce(llm.query_plan_priced_batch_count, 0) AS llm_query_plan_priced_batch_count,
                   coalesce(llm.query_plan_response_reported_cost_count, 0) AS llm_query_plan_response_reported_cost_count,
                   r.raw_summary, r.raw_sampling_config
            FROM runs r
            LEFT JOIN llm ON llm.run_id = r.id
            WHERE r.id = %s
            """,
            (run_id,),
        ).fetchone()
    return _row_or_404(run, "Run")


@app.get("/api/llm-query-plan-batches")
def llm_query_plan_batches(
    run_id: int | None = Query(default=None, ge=1),
    job_id: int | None = Query(default=None, ge=1),
    problem_only: bool = Query(default=True),
    limit: int = Query(default=50, ge=1, le=200),
    include_details: bool = Query(default=False),
) -> list[dict[str, Any]]:
    if run_id is None and job_id is None:
        raise HTTPException(status_code=400, detail="Provide run_id or job_id")
    filters = ["b.prompt_template = 'entity_retrieval_query_plan_v1'"]
    params: list[Any] = []
    if run_id is not None:
        filters.append("b.run_id = %s")
        params.append(run_id)
    if job_id is not None:
        filters.append("b.job_id = %s")
        params.append(job_id)
    problem_filter = ""
    if problem_only:
        problem_filter = "WHERE status = 'failed' OR error IS NOT NULL OR parse_warning IS NOT NULL OR missing_task_count > 0"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(
            f"""
            WITH task_stats AS (
                SELECT t.batch_id,
                       jsonb_agg(t.task_id ORDER BY t.id) AS requested_task_ids,
                       jsonb_agg(t.task_id ORDER BY t.task_id) FILTER (WHERE coalesce(t.plan_payload->>'query_plan_source', 'heuristic') = 'heuristic') AS inferred_missing_task_ids,
                       count(*) FILTER (WHERE coalesce(t.plan_payload->>'query_plan_source', 'heuristic') <> 'heuristic') AS usable_task_count,
                       count(*) FILTER (WHERE coalesce(t.plan_payload->>'query_plan_source', 'heuristic') = 'heuristic') AS missing_task_count,
                       count(*) FILTER (WHERE coalesce(t.plan_payload->>'query_plan_source', 'heuristic') = 'heuristic') AS heuristic_plan_count,
                       count(*) FILTER (WHERE m.id IS NOT NULL AND coalesce(m.candidate_count, 0) = 0) AS zero_candidate_count,
                       count(*) FILTER (WHERE coalesce(m.raw_payload->>'retrieval_error', '') <> '') AS retrieval_error_count,
                       count(*) FILTER (
                           WHERE coalesce(t.plan_payload->>'query_plan_source', 'heuristic') = 'heuristic'
                             AND m.id IS NOT NULL
                             AND (coalesce(m.candidate_count, 0) = 0 OR coalesce(m.raw_payload->>'retrieval_error', '') <> '')
                       ) AS heuristic_retrieval_problem_count
                FROM llm_prompt_tasks t
                LEFT JOIN mentions m ON m.id = t.mention_id
                GROUP BY t.batch_id
            ),
            batch_stats AS (
                SELECT b.id, b.run_id, b.job_id, b.provider, b.endpoint, b.model, b.prompt_template,
                       b.task_count, b.status, b.error, b.created_at, b.request_payload,
                       b.response_metadata,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'returned_task_count', '')::int, jsonb_array_length(coalesce(b.response_metadata->'tasks', '[]'::jsonb))) AS returned_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'usable_task_count', '')::int, t.usable_task_count, 0) AS usable_task_count,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'missing_task_count', '')::int, t.missing_task_count, 0) AS missing_task_count,
                       coalesce(b.response_metadata->'task_trace'->'missing_task_ids', t.inferred_missing_task_ids, '[]'::jsonb) AS missing_task_ids,
                       coalesce(b.response_metadata->'task_trace'->'requested_task_ids', t.requested_task_ids, '[]'::jsonb) AS requested_task_ids,
                       coalesce(b.response_metadata->'task_trace'->'unknown_task_ids', '[]'::jsonb) AS unknown_returned_task_ids,
                       coalesce(nullif(b.response_metadata->'task_trace'->>'invalid_task_count', '')::int, 0) AS invalid_task_count,
                       coalesce(t.heuristic_plan_count, 0) AS heuristic_plan_count,
                       coalesce(t.zero_candidate_count, 0) AS zero_candidate_count,
                       coalesce(t.retrieval_error_count, 0) AS retrieval_error_count,
                       coalesce(t.heuristic_retrieval_problem_count, 0) AS heuristic_retrieval_problem_count,
                       b.response_metadata->>'parse_warning' AS parse_warning,
                       b.response_metadata->'attempts' AS attempts,
                       b.response_metadata->'usage' AS usage,
                       b.response_metadata->'usage_cost' AS usage_cost
                FROM llm_prompt_batches b
                LEFT JOIN task_stats t ON t.batch_id = b.id
                WHERE {' AND '.join(filters)}
            )
            SELECT *
            FROM batch_stats
            {problem_filter}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
        batch_ids = [int(row["id"]) for row in rows]
        task_rows: list[dict[str, Any]] = []
        if include_details and batch_ids:
            task_rows = conn.execute(
                """
                SELECT t.batch_id, t.task_id, t.mention_text, t.lookup_text, t.plan_payload,
                       m.id AS mention_id, m.candidate_count, m.retrieved_count, m.best_gt_rank,
                       m.raw_payload->>'retrieval_error' AS retrieval_error,
                       m.raw_payload->'backend_requests' AS backend_requests
                FROM llm_prompt_tasks t
                LEFT JOIN mentions m ON m.id = t.mention_id
                WHERE t.batch_id = ANY(%s)
                ORDER BY t.batch_id, t.id
                """,
                (batch_ids,),
            ).fetchall()
    tasks_by_batch: dict[int, list[dict[str, Any]]] = {}
    for task in task_rows:
        tasks_by_batch.setdefault(int(task["batch_id"]), []).append(task)

    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.pop("request_payload", None)
        metadata = item.pop("response_metadata", {}) or {}
        response_tasks, response_parse_error = _extract_llm_response_tasks(metadata)
        returned_by_id = {
            str(task.get("id")): task
            for task in response_tasks
            if isinstance(task, dict) and task.get("id") is not None
        }
        requested_ids = [str(task_id) for task_id in (item.get("requested_task_ids") or [])]
        requested_id_set = set(requested_ids)
        returned_ids = list(returned_by_id.keys())
        unknown_returned_ids = [task_id for task_id in returned_ids if task_id not in requested_id_set]
        if response_tasks:
            item["returned_task_count"] = len(returned_ids)
            item["matched_returned_task_count"] = len([task_id for task_id in returned_ids if task_id in requested_id_set])
            item["unknown_returned_task_count"] = len(unknown_returned_ids)
            item["unknown_returned_task_ids"] = unknown_returned_ids
        else:
            item["matched_returned_task_count"] = 0
            item["unknown_returned_task_count"] = len(item.get("unknown_returned_task_ids") or [])
        item["response_parse_error"] = response_parse_error
        usage_cost = item.get("usage_cost") if isinstance(item.get("usage_cost"), dict) else None
        usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
        upstream_cost = _numeric(((usage.get("cost_details") or {}) if isinstance(usage.get("cost_details"), dict) else {}).get("upstream_inference_cost"))
        stored_cost = _numeric((usage_cost or {}).get("total_cost_usd"))
        if usage_cost is None or (upstream_cost is not None and upstream_cost > 0 and (stored_cost is None or stored_cost <= 0)):
            usage_cost = llm_usage_cost_from_metadata(
                provider=item.get("provider"),
                model=item.get("model"),
                usage=usage,
                config={},
            )
        item["usage_cost"] = usage_cost
        item["heuristic_analysis"] = {
            "heuristic_plan_count": int(item.get("heuristic_plan_count") or 0),
            "zero_candidate_count": int(item.get("zero_candidate_count") or 0),
            "retrieval_error_count": int(item.get("retrieval_error_count") or 0),
            "heuristic_retrieval_problem_count": int(item.get("heuristic_retrieval_problem_count") or 0),
        }

        if include_details:
            task_details = []
            inferred_missing_ids = set(str(task_id) for task_id in (item.get("missing_task_ids") or []))
            usable_count = 0
            missing_ids: list[str] = []
            for task in tasks_by_batch.get(int(item["id"]), []):
                task_id = str(task["task_id"])
                plan = task.get("plan_payload") if isinstance(task.get("plan_payload"), dict) else {}
                plan_source = str(plan.get("query_plan_source") or "heuristic")
                returned_task = returned_by_id.get(task_id)
                usable = plan_source != "heuristic" and not plan.get("error")
                retrieval_error = str(task.get("retrieval_error") or "").strip() or None
                candidate_count = int(task.get("candidate_count") or 0) if task.get("mention_id") is not None else None
                troubleshooting_flags = _task_troubleshooting_flags(
                    plan,
                    {
                        "retrieval_error": retrieval_error,
                        "candidate_count": candidate_count,
                    },
                )
                if usable:
                    usable_count += 1
                else:
                    missing_ids.append(task_id)
                if returned_task and not usable:
                    state = "returned_not_usable"
                elif returned_task:
                    state = "usable"
                else:
                    state = "missing"
                task_details.append(
                    {
                        "task_id": task_id,
                        "mention_id": task.get("mention_id"),
                        "mention_text": task.get("mention_text"),
                        "lookup_text": task.get("lookup_text"),
                        "state": state,
                        "returned": bool(returned_task),
                        "usable": usable,
                        "plan_source": plan_source,
                        "optimized_query": plan.get("optimized_query") or (returned_task or {}).get("optimized_query"),
                        "normalized_mention": plan.get("normalized_mention") or (returned_task or {}).get("normalized_mention"),
                        "coarse_type": plan.get("coarse_type") or (returned_task or {}).get("coarse_type"),
                        "fine_type": plan.get("fine_type") or (returned_task or {}).get("fine_type"),
                        "wikipedia_url": plan.get("wikipedia_url") or (returned_task or {}).get("wikipedia_url"),
                        "dbpedia_url": plan.get("dbpedia_url") or (returned_task or {}).get("dbpedia_url"),
                        "candidate_count": candidate_count,
                        "retrieved_count": task.get("retrieved_count"),
                        "best_gt_rank": task.get("best_gt_rank"),
                        "retrieval_error": retrieval_error,
                        "troubleshooting_flags": troubleshooting_flags,
                    }
                )
            if task_details:
                item["usable_task_count"] = usable_count
                if not item.get("missing_task_ids") or inferred_missing_ids != set(missing_ids):
                    item["missing_task_ids"] = missing_ids
                item["missing_task_count"] = len(missing_ids)
            item["task_details"] = task_details
            item["returned_tasks"] = [_compact_llm_task(task) for task in response_tasks[:200]]
            item["unknown_returned_tasks"] = [
                _compact_llm_task(returned_by_id[task_id]) for task_id in unknown_returned_ids[:200] if task_id in returned_by_id
            ]
            response_content = metadata.get("response_content")
            item["response_content"] = response_content if isinstance(response_content, str) and response_content.strip() != "None" else None
            item["parsed_response"] = metadata.get("parsed_response") if isinstance(metadata.get("parsed_response"), dict) else None
        item["explanation"] = _llm_batch_explanation(item)
        items.append(item)
    return items


@app.get("/api/heuristic-plan-analysis")
def heuristic_plan_analysis(
    run_id: int = Query(ge=1),
    problem_only: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    problem_filter = ""
    if problem_only:
        problem_filter = """
        AND (
            candidate_count = 0
            OR retrieval_error IS NOT NULL
            OR query_plan_error IS NOT NULL
            OR best_gt_rank IS NULL
        )
        """
    with connect() as conn:
        summary = conn.execute(
            """
            WITH heuristic_mentions AS (
                SELECT
                    m.id,
                    m.candidate_count,
                    m.best_gt_rank,
                    nullif(m.raw_payload->>'retrieval_error', '') AS retrieval_error,
                    nullif(m.raw_payload->'query_plan'->>'query_plan_error', '') AS query_plan_error
                FROM mentions m
                WHERE m.run_id = %s
                  AND (
                      coalesce(m.raw_payload->'query_plan'->>'query_plan_source', '') LIKE 'heuristic%%'
                      OR coalesce(m.query_engine, '') LIKE 'heuristic%%'
                      OR nullif(m.raw_payload->'query_plan'->>'query_plan_error', '') IS NOT NULL
                  )
            )
            SELECT
                count(*) AS heuristic_plan_count,
                count(*) FILTER (WHERE query_plan_error IS NOT NULL) AS llm_fallback_error_count,
                count(*) FILTER (WHERE candidate_count = 0) AS zero_candidate_count,
                count(*) FILTER (WHERE retrieval_error IS NOT NULL) AS retrieval_error_count,
                count(*) FILTER (WHERE candidate_count = 0 OR retrieval_error IS NOT NULL) AS retrieval_problem_count,
                count(*) FILTER (WHERE best_gt_rank IS NULL) AS missed_count,
                count(*) FILTER (WHERE best_gt_rank IS NOT NULL) AS covered_count
            FROM heuristic_mentions
            """,
            (run_id,),
        ).fetchone()
        rows = conn.execute(
            f"""
            WITH heuristic_mentions AS (
                SELECT
                    m.id,
                    m.cell_key,
                    m.dataset_id,
                    m.table_id,
                    m.row_id,
                    m.col_id,
                    m.mention_text,
                    m.lookup_text,
                    m.primary_gt_qid,
                    m.candidate_count,
                    m.retrieved_count,
                    m.best_gt_rank,
                    m.query_engine,
                    m.raw_payload->'query_plan' AS query_plan,
                    coalesce(m.raw_payload->'query_plan'->>'query_plan_source', m.query_engine, 'heuristic') AS query_plan_source,
                    nullif(m.raw_payload->'query_plan'->>'query_plan_error', '') AS query_plan_error,
                    nullif(m.raw_payload->>'retrieval_error', '') AS retrieval_error
                FROM mentions m
                WHERE m.run_id = %s
                  AND (
                      coalesce(m.raw_payload->'query_plan'->>'query_plan_source', '') LIKE 'heuristic%%'
                      OR coalesce(m.query_engine, '') LIKE 'heuristic%%'
                      OR nullif(m.raw_payload->'query_plan'->>'query_plan_error', '') IS NOT NULL
                  )
            )
            SELECT *
            FROM heuristic_mentions
            WHERE TRUE
            {problem_filter}
            ORDER BY
                CASE WHEN retrieval_error IS NOT NULL THEN 0 ELSE 1 END,
                CASE WHEN candidate_count = 0 THEN 0 ELSE 1 END,
                CASE WHEN query_plan_error IS NOT NULL THEN 0 ELSE 1 END,
                best_gt_rank NULLS FIRST,
                id ASC
            LIMIT %s
            """,
            (run_id, limit),
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        plan = item.get("query_plan") if isinstance(item.get("query_plan"), dict) else {}
        item["optimized_query"] = plan.get("optimized_query") or item.get("lookup_text") or item.get("mention_text")
        item["normalized_mention"] = plan.get("normalized_mention")
        item["coarse_type"] = plan.get("coarse_type")
        item["fine_type"] = plan.get("fine_type")
        item["troubleshooting_flags"] = _task_troubleshooting_flags(
            {**plan, "query_plan_source": item.get("query_plan_source"), "query_plan_error": item.get("query_plan_error")},
            {"retrieval_error": item.get("retrieval_error"), "candidate_count": item.get("candidate_count")},
        )
        item.pop("query_plan", None)
        items.append(item)

    summary_item = dict(summary or {})
    total = int(summary_item.get("heuristic_plan_count") or 0)
    covered = int(summary_item.get("covered_count") or 0)
    summary_item["coverage"] = covered / total if total else 0
    return {"summary": summary_item, "rows": items}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: int) -> dict[str, Any]:
    with connect() as conn:
        active_job = conn.execute(
            """
            SELECT id, status
            FROM experiment_jobs
            WHERE imported_run_id = %s
              AND status IN ('queued', 'running', 'cancel_requested')
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if active_job:
            raise HTTPException(
                status_code=409,
                detail=f"Run is attached to active job {active_job['id']}; cancel the job first",
            )
        row = conn.execute(
            """
            DELETE FROM runs
            WHERE id = %s
            RETURNING id, name
            """,
            (run_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        conn.execute(
            """
            UPDATE experiment_jobs
            SET imported_run_id = NULL
            WHERE imported_run_id = %s
            """,
            (run_id,),
        )
        conn.commit()
    return {"deleted": True, "id": row["id"], "name": row["name"]}


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
            SELECT coalesce(candidate_backend, query_engine, 'alpaca') AS retrieval_stage,
                   coalesce(sum(candidate_count), 0) AS candidate_count
            FROM mentions
            WHERE run_id = %s
            GROUP BY coalesce(candidate_backend, query_engine, 'alpaca')
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
                SELECT coalesce(max(coalesce(best_gt_rank, retrieved_count, candidate_count)), 1) AS max_rank
                FROM mentions
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()
            parsed = _default_coverage_buckets(int(max_rank_row["max_rank"] if max_rank_row else 1))

        rows = conn.execute(
            """
            WITH scored_mentions AS (
                SELECT id, best_gt_rank AS imported_best_rank
                FROM mentions
                WHERE run_id = %s
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
        base_conditions.append("(m.mention_text ILIKE %s OR m.lookup_text ILIKE %s OR m.cell_key ILIKE %s)")
        like = f"%{search}%"
        base_params.extend([like, like, like])

    base_where = " AND ".join(base_conditions)
    scored_cte = f"""
        WITH scored_mentions AS (
            SELECT
                m.id, m.cell_key, m.dataset_id, m.table_id, m.row_id, m.col_id,
                m.mention_text AS mention, m.lookup_text, m.primary_gt_qid,
                m.best_gt_rank, m.retrieved_count, m.candidate_count,
                (m.best_gt_rank IS NOT NULL) AS covered_by_imported_candidates,
                m.best_gt_rank AS imported_best_rank
            FROM mentions m
            WHERE {base_where}
        )
    """
    score_where = "TRUE"
    if covered == "covered":
        score_where = "covered_by_imported_candidates"
    elif covered == "missed":
        score_where = "NOT covered_by_imported_candidates"
    scored_params = [*base_params]
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
            SELECT
                m.*,
                m.mention_text AS mention,
                r.name AS run_name,
                (m.best_gt_rank IS NOT NULL) AS covered_by_imported_candidates,
                m.best_gt_rank AS imported_best_rank,
                cache.response_payload AS cached_response_payload
            FROM mentions m
            JOIN runs r ON r.id = m.run_id
            LEFT JOIN candidate_retrieval_cache cache ON cache.cache_key = m.candidate_cache_key
            WHERE m.id = %s
            """,
            (mention_id,),
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
        gold_qid_set = {str(row["qid"]) for row in gold if row.get("qid")}
        candidates = _candidates_from_cache_payload(mention.get("cached_response_payload"), gold_qid_set)
        mention.pop("cached_response_payload", None)
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
        query_plan_batch = None
        if mention.get("query_plan_batch_id"):
            query_plan_batch = conn.execute(
                """
                SELECT
                    b.id, b.run_id, b.job_id, b.provider, b.endpoint, b.model,
                    b.prompt_template, b.task_count, b.status, b.error,
                    b.request_payload, b.response_metadata, b.created_at,
                    t.task_id, t.mention_text, t.lookup_text, t.plan_payload
                FROM llm_prompt_batches b
                LEFT JOIN llm_prompt_tasks t ON t.batch_id = b.id AND t.mention_id = %s
                WHERE b.id = %s
                """,
                (mention_id, mention["query_plan_batch_id"]),
            ).fetchone()
        attempt_rows = []
        for attempt in attempts:
            attempt_row = dict(attempt)
            response_payload = attempt_row.get("response_payload")
            if isinstance(response_payload, dict):
                attempt_candidates = response_payload.get("candidates")
                if isinstance(attempt_candidates, list):
                    covered_qids = {str(qid) for qid in attempt_row.get("covered_qids") or []}
                    attempt_row["candidates"] = [
                        {
                            **candidate,
                            "gold_match": bool(candidate.get("gold_match"))
                            or bool(candidate.get("qid") and str(candidate.get("qid")) in covered_qids),
                        }
                        for candidate in attempt_candidates
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
        "query_plan_batch": query_plan_batch,
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

    by_qid = {
        str(candidate.get("qid")): _entity_with_ner_types(candidate)
        for candidate in hits
        if candidate.get("qid")
    }
    resolved_entities = [by_qid[qid] for qid in qids if qid in by_qid]
    if resolved_entities:
        with connect() as conn:
            conn.cursor().executemany(
                """
                UPDATE gold_qids
                SET raw_entity = %s
                WHERE mention_id = %s
                  AND qid = %s
                """,
                [
                    (Jsonb(entity), mention_id, str(entity["qid"]))
                    for entity in resolved_entities
                    if entity.get("qid")
                ],
            )
            conn.commit()

    for qid in qids:
        if qid in by_qid:
            return {
                "requested_qids": qids,
                "resolved_qid": qid,
                "entity": by_qid[qid],
                "entities": resolved_entities,
                "ner_types": [
                    {
                        "qid": entity.get("qid"),
                        "coarse_type": entity.get("coarse_type"),
                        "fine_type": entity.get("fine_type"),
                    }
                    for entity in resolved_entities
                ],
                "all_found_qids": [str(candidate.get("qid")) for candidate in hits if candidate.get("qid")],
            }

    return {
        "requested_qids": qids,
        "resolved_qid": None,
        "entity": None,
        "entities": [],
        "ner_types": [],
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
            SELECT id, mention_text AS mention, lookup_text, raw_payload
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
        query_plan_generator = normalize_query_plan_source(query_plan_source)
        mention_text = (mention.get("mention") or mention.get("lookup_text") or query_text).strip()
        query_body = build_alpaca_query(
            query_plan.get("optimized_query") or query_text,
            query_plan,
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
                "query_plan_source": query_plan_source,
                "query_plan_generator": query_plan_generator,
                "cache_key_fields": {
                    "mention": mention_text,
                    "query": query_plan.get("optimized_query") or query_text,
                    "query_plan_source": query_plan_generator,
                },
                "hard_filters": {"item_category": ["ENTITY", "TYPE"]},
                "soft_signals": {
                    "coarse_type": query_plan.get("coarse_type"),
                    "fine_type": query_plan.get("fine_type"),
                    "original_fine_type": query_plan.get("original_fine_type"),
                    "fine_type_rule_applied": query_plan.get("fine_type_rule_applied"),
                    "fine_type_normalization": query_plan.get("fine_type_normalization"),
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
            response_payload = alpaca_search(
                query_body,
                retrieval_count,
                mention_text=mention_text,
                query_text=query_plan.get("optimized_query") or query_text,
                query_plan_source=query_plan_generator,
            )
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
                        "query_plan_generator": query_plan_generator,
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
