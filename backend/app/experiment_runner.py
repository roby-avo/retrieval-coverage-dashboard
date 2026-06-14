from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from psycopg.types.json import Jsonb

from .datasets import build_random_sample_bundle_from_db
from .db import connect
from .importer import create_import_run, finalize_import_run, import_experiment_row, update_run_counters, upsert_experiment_table
from .retrieval import (
    ALPACA_METADATA_URL,
    MAX_RETURNED_CANDIDATES,
    MAX_RETRIEVAL_CANDIDATES,
    alpaca_search,
    build_alpaca_query,
    extract_hits,
)


DEFAULT_DATASETS = [
    "2T_2020",
    "2T_2022",
    "HardTablesR2",
    "HardTablesR3",
    "HardTableR1_2022",
    "HardTableR2_2022",
    "Round1_T2D",
    "Round3_2019",
    "Round4_2020",
]
OPENROUTER_REQUIRED_MODEL = "openai/gpt-oss-120b"
OPENROUTER_REQUIRED_PROVIDER = "Cerebras"
DEFAULT_LLM_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
TYPO_CORRECTION_CONFIDENCE_THRESHOLD = 0.85
STOPWORDS = {"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to", "with"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _job_name(job_id: int, config: dict[str, Any]) -> str:
    raw = str(config.get("name") or "").strip()
    if raw:
        return "".join(char if char.isalnum() or char in "-_." else "_" for char in raw)[:120]
    return f"webapp_experiment_job_{job_id}"


def default_experiment_config() -> dict[str, Any]:
    llm_provider = os.environ.get("LLM_PROVIDER", "openrouter").strip() or "openrouter"
    llm_provider_name = os.environ.get("LLM_PROVIDER_NAME", os.environ.get("OPENROUTER_PROVIDER", OPENROUTER_REQUIRED_PROVIDER)).strip()
    llm_model = os.environ.get("LLM_MODEL", os.environ.get("OPENROUTER_MODEL", OPENROUTER_REQUIRED_MODEL)).strip()
    llm_api_url = os.environ.get("LLM_API_URL", os.environ.get("OPENROUTER_CHAT_URL", DEFAULT_LLM_CHAT_URL)).strip()
    llm_parallel_requests = int(os.environ.get("LLM_PARALLEL_REQUESTS", os.environ.get("OPENROUTER_PARALLEL_REQUESTS", "2")))
    return {
        "name": "",
        "requested_datasets": DEFAULT_DATASETS,
        "dataset_sample_size": 8,
        "tables_per_dataset": 5,
        "records_per_table": 10,
        "random_seed": 42,
        "context_rows": 4,
        "table_context_preview_rows": 4,
        "max_candidates": 100,
        "dashboard_candidate_limit": MAX_RETURNED_CANDIDATES,
        "save_full_debug_output": False,
        "enable_recall_query_expansion": False,
        "recall_query_variant_limit": 14,
        "recall_context_term_limit": 24,
        "recall_token_combo_limit": 12,
        "enable_llm_url_hints": True,
        "url_hint_boost": 260,
        "url_hint_confidence_threshold": 0.2,
        "dataset_allowlist": [],
        "table_allowlist_by_dataset": {},
        "max_tasks_per_llm_request": 8,
        "llm_parallel_requests": llm_parallel_requests,
        "max_workers": 8,
        "llm_enabled": True,
        "llm_provider": llm_provider,
        "llm_provider_name": llm_provider_name,
        "llm_api_url": llm_api_url,
        "llm_api_key": "",
        "llm_model": llm_model,
        "llm_reasoning_effort": "high",
        "llm_temperature": float(os.environ.get("LLM_TEMPERATURE", os.environ.get("OPENROUTER_TEMPERATURE", "0.1"))),
        "llm_timeout_seconds": int(os.environ.get("LLM_TIMEOUT_SECONDS", os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "180"))),
        "llm_max_retries": int(os.environ.get("LLM_MAX_RETRIES", os.environ.get("OPENROUTER_MAX_RETRIES", "5"))),
        "llm_site_url": os.environ.get("LLM_SITE_URL", os.environ.get("OPENROUTER_SITE_URL", "http://localhost")),
        "llm_app_name": os.environ.get("LLM_APP_NAME", os.environ.get("OPENROUTER_APP_NAME", "alpaca-random-coverage-sampler")),
        "llm_max_tokens": int(os.environ["LLM_MAX_TOKENS"]) if os.environ.get("LLM_MAX_TOKENS") else int(os.environ["OPENROUTER_MAX_TOKENS"]) if os.environ.get("OPENROUTER_MAX_TOKENS") else None,
        "use_openrouter": True,
        "use_heuristic_fallback_on_llm_failure": True,
        "openrouter_parallel_requests": llm_parallel_requests,
        "openrouter_model": llm_model,
        "openrouter_provider": llm_provider_name,
        "openrouter_reasoning_effort": "high",
        "openrouter_temperature": float(os.environ.get("OPENROUTER_TEMPERATURE", "0.1")),
        "openrouter_timeout_seconds": int(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "180")),
        "openrouter_max_retries": int(os.environ.get("OPENROUTER_MAX_RETRIES", "5")),
        "openrouter_site_url": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
        "openrouter_app_name": os.environ.get("OPENROUTER_APP_NAME", "alpaca-random-coverage-sampler"),
        "openrouter_max_tokens": int(os.environ["OPENROUTER_MAX_TOKENS"]) if os.environ.get("OPENROUTER_MAX_TOKENS") else None,
    }


def normalize_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = default_experiment_config()
    merged = {**defaults, **(config or {})}

    def bounded_int(key: str, minimum: int, maximum: int) -> int:
        try:
            value = int(merged.get(key))
        except (TypeError, ValueError):
            value = int(defaults[key])
        return max(minimum, min(maximum, value))

    merged["dataset_sample_size"] = bounded_int("dataset_sample_size", 0, 1000)
    merged["tables_per_dataset"] = bounded_int("tables_per_dataset", 0, 1000)
    merged["records_per_table"] = bounded_int("records_per_table", 0, 1000)
    merged["random_seed"] = bounded_int("random_seed", 0, 2_147_483_647)
    merged["context_rows"] = bounded_int("context_rows", 0, 20)
    merged["table_context_preview_rows"] = bounded_int("table_context_preview_rows", 0, 20)
    merged["max_candidates"] = bounded_int("max_candidates", 1, MAX_RETRIEVAL_CANDIDATES)
    merged["dashboard_candidate_limit"] = bounded_int("dashboard_candidate_limit", 1, MAX_RETURNED_CANDIDATES)
    merged["max_tasks_per_llm_request"] = bounded_int("max_tasks_per_llm_request", 1, 100)
    if "llm_parallel_requests" not in merged and "openrouter_parallel_requests" in merged:
        merged["llm_parallel_requests"] = merged["openrouter_parallel_requests"]
    merged["llm_parallel_requests"] = bounded_int("llm_parallel_requests", 1, 16)
    merged["openrouter_parallel_requests"] = merged["llm_parallel_requests"]
    merged["max_workers"] = bounded_int("max_workers", 1, 32)
    merged["recall_query_variant_limit"] = bounded_int("recall_query_variant_limit", 1, 100)
    merged["recall_context_term_limit"] = bounded_int("recall_context_term_limit", 1, 200)
    merged["recall_token_combo_limit"] = bounded_int("recall_token_combo_limit", 1, 100)

    requested = merged.get("requested_datasets")
    if not isinstance(requested, list) or not requested:
        requested = DEFAULT_DATASETS
    merged["requested_datasets"] = [str(item) for item in requested if str(item).strip()]
    merged["dataset_allowlist"] = [str(item) for item in merged.get("dataset_allowlist") or []] if isinstance(merged.get("dataset_allowlist"), list) else []
    if not isinstance(merged.get("table_allowlist_by_dataset"), dict):
        merged["table_allowlist_by_dataset"] = {}

    provider = str(merged.get("llm_provider") or ("openrouter" if merged.get("use_openrouter", True) else "none")).strip().lower()
    if provider in {"off", "disabled", "false"}:
        provider = "none"
    merged["llm_provider"] = provider
    merged["llm_enabled"] = bool(merged.get("llm_enabled", merged.get("use_openrouter", True))) and provider != "none"
    merged["use_openrouter"] = merged["llm_enabled"]
    merged["llm_provider_name"] = str(merged.get("llm_provider_name") or merged.get("openrouter_provider") or "").strip()
    merged["llm_api_url"] = str(merged.get("llm_api_url") or os.environ.get("LLM_API_URL") or os.environ.get("OPENROUTER_CHAT_URL") or DEFAULT_LLM_CHAT_URL).strip()
    merged["llm_api_key"] = str(merged.get("llm_api_key") or "").strip()
    merged["llm_model"] = str(merged.get("llm_model") or merged.get("openrouter_model") or OPENROUTER_REQUIRED_MODEL).strip()
    merged["llm_reasoning_effort"] = str(merged.get("llm_reasoning_effort") or merged.get("openrouter_reasoning_effort") or "high").strip()
    merged["openrouter_model"] = merged["llm_model"]
    merged["openrouter_provider"] = merged["llm_provider_name"] or OPENROUTER_REQUIRED_PROVIDER
    merged["openrouter_reasoning_effort"] = merged["llm_reasoning_effort"]
    for next_key, legacy_key in (
        ("llm_temperature", "openrouter_temperature"),
        ("llm_timeout_seconds", "openrouter_timeout_seconds"),
        ("llm_max_tokens", "openrouter_max_tokens"),
    ):
        if next_key not in merged and legacy_key in merged:
            merged[next_key] = merged[legacy_key]
    for next_key, legacy_key in (
        ("llm_site_url", "openrouter_site_url"),
        ("llm_app_name", "openrouter_app_name"),
    ):
        merged[next_key] = str(merged.get(next_key) or merged.get(legacy_key) or defaults[next_key])
    try:
        merged["llm_temperature"] = float(merged.get("llm_temperature", 0.1))
    except (TypeError, ValueError):
        merged["llm_temperature"] = float(defaults["llm_temperature"])
    try:
        merged["llm_timeout_seconds"] = max(1, int(merged.get("llm_timeout_seconds") or defaults["llm_timeout_seconds"]))
    except (TypeError, ValueError):
        merged["llm_timeout_seconds"] = defaults["llm_timeout_seconds"]
    try:
        merged["llm_max_retries"] = max(1, int(merged.get("llm_max_retries") or merged.get("openrouter_max_retries") or 5))
    except (TypeError, ValueError):
        merged["llm_max_retries"] = 5
    try:
        merged["llm_max_tokens"] = int(merged["llm_max_tokens"]) if merged.get("llm_max_tokens") is not None and merged.get("llm_max_tokens") != "" else None
    except (TypeError, ValueError):
        merged["llm_max_tokens"] = None
    merged["openrouter_temperature"] = merged["llm_temperature"]
    merged["openrouter_timeout_seconds"] = merged["llm_timeout_seconds"]
    merged["openrouter_max_retries"] = merged["llm_max_retries"]
    merged["openrouter_site_url"] = merged["llm_site_url"]
    merged["openrouter_app_name"] = merged["llm_app_name"]
    merged["openrouter_max_tokens"] = merged["llm_max_tokens"]

    for key in (
        "save_full_debug_output",
        "enable_recall_query_expansion",
        "enable_llm_url_hints",
        "llm_enabled",
        "use_openrouter",
        "use_heuristic_fallback_on_llm_failure",
    ):
        merged[key] = bool(merged.get(key))
    return merged


def create_experiment_job(config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_experiment_config(config)
    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO experiment_jobs (config, message)
            VALUES (%s, %s)
            RETURNING *
            """,
            (Jsonb(normalized), "Queued"),
        ).fetchone()
        conn.commit()
    if not row:
        raise RuntimeError("Could not create experiment job")
    thread = threading.Thread(target=run_experiment_job, args=(int(row["id"]),), daemon=True)
    thread.start()
    return row


def update_job(job_id: int, **fields: Any) -> None:
    if not fields:
        return
    assignments = []
    values = []
    for key, value in fields.items():
        assignments.append(f"{key} = %s")
        values.append(Jsonb(value) if key in {"config", "stage_progress"} else value)
    values.append(job_id)
    with connect() as conn:
        conn.execute(f"UPDATE experiment_jobs SET {', '.join(assignments)} WHERE id = %s", values)
        conn.commit()


def _llm_api_key(config: dict[str, Any]) -> str:
    return str(
        config.get("llm_api_key")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or ""
    ).strip()


def _llm_endpoint(config: dict[str, Any]) -> str:
    raw = str(config.get("llm_api_url") or os.environ.get("LLM_API_URL") or os.environ.get("OPENROUTER_CHAT_URL") or DEFAULT_LLM_CHAT_URL).strip()
    if not raw:
        return DEFAULT_LLM_CHAT_URL
    trimmed = raw.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return raw
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    if "://" in trimmed and "/" not in trimmed.split("://", 1)[1]:
        return f"{trimmed}/v1/chat/completions"
    return raw


def _llm_label(config: dict[str, Any]) -> str:
    if not config.get("llm_enabled", config.get("use_openrouter")):
        return "Heuristic"
    provider = str(config.get("llm_provider_name") or config.get("llm_provider") or "LLM").strip()
    return provider if provider.casefold() != "openai_compatible" else "OpenAI-compatible"


def _sampling_config(config: dict[str, Any], run_started_at: datetime) -> dict[str, Any]:
    return {
        "dataset_sample_size": config["dataset_sample_size"],
        "tables_per_dataset": config["tables_per_dataset"],
        "records_per_table": config["records_per_table"],
        "random_seed": config["random_seed"],
        "dataset_allowlist": config["dataset_allowlist"],
        "table_allowlist_by_dataset": config["table_allowlist_by_dataset"],
        "max_candidates": config["max_candidates"],
        "context_rows": config["context_rows"],
        "table_context_preview_rows": config["table_context_preview_rows"],
        "llm_provider": config["llm_provider"],
        "llm_provider_name": config["llm_provider_name"],
        "llm_api_url": _llm_endpoint(config),
        "llm_model": config["llm_model"],
        "llm_parallel_requests": config["llm_parallel_requests"],
        "openrouter_model": config["openrouter_model"],
        "openrouter_provider": config["openrouter_provider"],
        "openrouter_parallel_requests": config["openrouter_parallel_requests"],
        "alpaca_search_endpoint": os.environ.get("ALPACA_SEARCH_ENDPOINT", ALPACA_METADATA_URL),
        "created_at": run_started_at.isoformat(),
        "source_storage": "filesystem_lazy",
        "run_storage": "postgres_incremental",
    }


def _stage_entry(
    *,
    label: str,
    current: int,
    total: int,
    started_wall: datetime,
    started_monotonic: float,
    status: str = "running",
) -> dict[str, Any]:
    bounded_current = max(0, min(current, total)) if total else max(0, current)
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    eta = None
    if status == "running" and total > 0 and bounded_current > 0:
        rate = bounded_current / elapsed if elapsed > 0 else 0
        if rate > 0:
            eta = max(0.0, (total - bounded_current) / rate)
    return {
        "label": label,
        "current": bounded_current,
        "total": total,
        "status": status,
        "started_at": started_wall.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
        "finished_at": _now().isoformat() if status in {"completed", "failed"} else None,
    }


def _tokens(value: Any, *, limit: int = 12) -> list[str]:
    text = str(value or "").casefold()
    result: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def _clean_terms(raw_terms: Any, query_text: str, *, limit: int) -> list[str]:
    if isinstance(raw_terms, str):
        raw_terms = re.split(r"[,;\n]", raw_terms)
    if not isinstance(raw_terms, list | tuple):
        return []
    query_norm = str(query_text or "").casefold().strip()
    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        term = re.sub(r"\s+", " ", str(raw or "")).strip(" \t\r\n,;")[:48]
        normalized = term.casefold()
        if not term or normalized == query_norm or normalized in seen:
            continue
        if re.fullmatch(r"col\d+|[\d,._-]+", normalized):
            continue
        seen.add(normalized)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _target_context_text(sample: dict[str, Any], *, limit: int = 700) -> str:
    values: list[str] = []
    header = sample.get("header")
    if isinstance(header, list):
        values.append("headers: " + " | ".join(str(item) for item in header if item is not None))
    target_row = sample.get("target_row")
    if isinstance(target_row, list):
        values.append("target row: " + " | ".join(str(cell) for cell in target_row if cell is not None))
    return " ".join(values)[:limit]


def _table_context_preview(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    preview_rows = int(config["table_context_preview_rows"])
    before = list(sample.get("rows_before") or [])[-preview_rows:]
    after = list(sample.get("rows_after") or [])[:preview_rows]
    rows = []
    before_start = int(sample["row_id"]) - len(before)
    for index, row in enumerate(before):
        rows.append(
            {
                "row_id": before_start + index,
                "relative_position": index - len(before),
                "is_target": False,
                "cells": row,
                "mention_cell": row[sample["col_id"]] if sample["col_id"] < len(row) else "",
            }
        )
    target_row = list(sample.get("target_row") or [])
    rows.append(
        {
            "row_id": sample["row_id"],
            "relative_position": 0,
            "is_target": True,
            "cells": target_row,
            "mention_cell": target_row[sample["col_id"]] if sample["col_id"] < len(target_row) else "",
        }
    )
    for index, row in enumerate(after, start=1):
        rows.append(
            {
                "row_id": int(sample["row_id"]) + index,
                "relative_position": index,
                "is_target": False,
                "cells": row,
                "mention_cell": row[sample["col_id"]] if sample["col_id"] < len(row) else "",
            }
        )
    return {
        "header": sample.get("header") or [],
        "target_row_id": sample.get("row_id"),
        "target_col_id": sample.get("col_id"),
        "header_cell": sample.get("header_cell"),
        "rows": rows,
    }


def _heuristic_plan(sample: dict[str, Any]) -> dict[str, Any]:
    lookup_text = str(sample.get("lookup_text") or sample.get("mention_text") or "").strip()
    context_terms = []
    query_tokens = set(_tokens(lookup_text, limit=24))
    for token in _tokens(" ".join(str(item) for item in sample.get("lookup_context") or []), limit=20):
        if token in query_tokens:
            continue
        context_terms.append(token)
        if len(context_terms) >= 2:
            break
    return {
        "optimized_query": lookup_text,
        "normalized_mention": None,
        "typo_corrected_mention": None,
        "typo_correction_confidence": None,
        "typo_correction_applied": False,
        "typo_correction_reason": None,
        "coarse_type": None,
        "fine_type": None,
        "wikipedia_url": None,
        "dbpedia_url": None,
        "aliases": [],
        "context_expansion_terms": context_terms,
        "query_plan_source": "heuristic",
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(stripped[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _normalize_plan(raw_plan: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    fallback_query = str(sample.get("lookup_text") or sample.get("mention_text") or "").strip()

    def optional_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None and value != "" else None
        except (TypeError, ValueError):
            return None

    typo_corrected_mention = str(raw_plan.get("typo_corrected_mention") or raw_plan.get("corrected_mention") or "").strip() or None
    typo_correction_confidence = optional_float(raw_plan.get("typo_correction_confidence", raw_plan.get("typo_confidence")))
    typo_correction_applied = bool(
        typo_corrected_mention
        and typo_correction_confidence is not None
        and typo_correction_confidence >= TYPO_CORRECTION_CONFIDENCE_THRESHOLD
        and typo_corrected_mention.casefold() != fallback_query.casefold()
    )
    optimized_query = str(raw_plan.get("optimized_query") or raw_plan.get("query") or fallback_query).strip()
    if typo_correction_applied:
        optimized_query = typo_corrected_mention or optimized_query
    return {
        "optimized_query": optimized_query or fallback_query,
        "normalized_mention": str(raw_plan.get("normalized_mention") or "").strip() or None,
        "typo_corrected_mention": typo_corrected_mention,
        "typo_correction_confidence": typo_correction_confidence,
        "typo_correction_applied": typo_correction_applied,
        "typo_correction_reason": str(raw_plan.get("typo_correction_reason") or "").strip() or None,
        "coarse_type": str(raw_plan.get("coarse_type") or "").strip().upper() or None,
        "fine_type": str(raw_plan.get("fine_type") or "").strip().upper() or None,
        "wikipedia_url": raw_plan.get("wikipedia_url") or raw_plan.get("wikipedia_title"),
        "dbpedia_url": raw_plan.get("dbpedia_url") or raw_plan.get("dbpedia_title"),
        "aliases": _clean_terms(raw_plan.get("aliases"), optimized_query, limit=6),
        "context_expansion_terms": _clean_terms(raw_plan.get("context_expansion_terms", raw_plan.get("context_terms")), optimized_query, limit=3),
    }


def _llm_batch_body(samples: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    tasks = [
        {
            "id": sample["sample_id"],
            "mention": sample.get("mention_text"),
            "lookup_text": sample.get("lookup_text"),
            "dataset": sample.get("dataset"),
            "table_id": sample.get("table_id"),
            "row_id": sample.get("row_id"),
            "col_id": sample.get("col_id"),
            "header": sample.get("header"),
            "target_row": sample.get("target_row"),
            "rows_before": sample.get("rows_before"),
            "rows_after": sample.get("rows_after"),
            "context": sample.get("lookup_context"),
        }
        for sample in samples
    ]
    body: dict[str, Any] = {
        "model": config["llm_model"],
        "temperature": float(config["llm_temperature"]),
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return JSON only. Generate recall-oriented Elasticsearch entity retrieval plans for table mentions. "
                    "Keep optimized_query short, preserve the mention unless the table context strongly supports a correction, "
                    "and use type or URL predictions only as soft ranking signals."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "tasks": tasks,
                        "output_schema": {
                            "tasks": [
                                {
                                    "id": "same id",
                                    "optimized_query": "short query",
                                    "normalized_mention": "normalized mention or null",
                                    "typo_corrected_mention": "high-confidence correction or null",
                                    "typo_correction_confidence": 0.0,
                                    "typo_correction_reason": "short reason or null",
                                    "coarse_type": "PERSON | ORGANIZATION | LOCATION | MISC | CONCEPT | EVENT | PRODUCT | WORK | TYPE | null",
                                    "fine_type": "specific type or null",
                                    "wikipedia_url": "page slug or null",
                                    "dbpedia_url": "resource slug or null",
                                    "aliases": ["short alternatives"],
                                    "context_expansion_terms": ["short discriminating terms"],
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    if config.get("llm_provider") == "openrouter":
        body["include_reasoning"] = False
        body["reasoning"] = {"effort": config["llm_reasoning_effort"]}
        if config.get("llm_provider_name"):
            body["provider"] = {"order": [config["llm_provider_name"]], "allow_fallbacks": False}
    return body


def _call_llm_batch(samples: Sequence[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    token = _llm_api_key(config)
    if not token:
        raise RuntimeError("LLM API key is not configured")
    endpoint = _llm_endpoint(config)
    body = _llm_batch_body(samples, config)
    if config.get("llm_max_tokens") is not None:
        body["max_tokens"] = config["llm_max_tokens"]

    timeout = int(config["llm_timeout_seconds"])
    max_retries = max(1, int(config["llm_max_retries"]))
    retry_base_seconds = max(0.5, float(os.environ.get("LLM_RETRY_BASE_SECONDS", os.environ.get("OPENROUTER_RETRY_BASE_SECONDS", "4"))))
    payload: dict[str, Any] = {}
    last_error: Exception | None = None
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
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = RuntimeError(f"{_llm_label(config)} query planning failed: {exc}")
            if attempt < max_retries:
                time.sleep(min(30.0, retry_base_seconds * attempt))
                continue
            raise last_error from exc
    else:
        raise RuntimeError(f"{_llm_label(config)} query planning failed: {last_error}")

    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _extract_json_object(str(content))
    raw_tasks = parsed.get("tasks") if isinstance(parsed.get("tasks"), list) else []
    plans: dict[str, dict[str, Any]] = {}
    by_id = {sample["sample_id"]: sample for sample in samples}
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict) or not raw_task.get("id"):
            continue
        sample_id = str(raw_task["id"])
        sample = by_id.get(sample_id)
        if not sample:
            continue
        plans[sample_id] = _normalize_plan(raw_task, sample)

    log = {
        "error": None,
        "sample_count": len(samples),
        "provider": config.get("llm_provider"),
        "endpoint": endpoint,
        "request_body": body,
        "response_usage": payload.get("usage"),
        "response_model": payload.get("model"),
        "response_provider": payload.get("provider"),
        "response_id": payload.get("id"),
    }
    return plans, log


def _batched(items: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    size = max(1, int(batch_size))
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def _generate_query_plans(
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    plans = {sample["sample_id"]: _heuristic_plan(sample) for sample in samples}
    logs: list[dict[str, Any]] = []
    if not config["llm_enabled"]:
        if progress_callback:
            progress_callback(len(samples))
        return plans, logs

    def apply_batch_plans(batch: list[dict[str, Any]], batch_plans: dict[str, dict[str, Any]], log: dict[str, Any]) -> None:
        for sample in batch:
            plan = batch_plans.get(sample["sample_id"])
            if plan:
                plan["query_plan_source"] = str(config.get("llm_provider") or "llm")
                plan["llm_inspection"] = {
                    "sent": True,
                    "provider": log.get("provider"),
                    "endpoint": log.get("endpoint"),
                    "batch_size": len(batch),
                    "task_ids": [item["sample_id"] for item in batch],
                    "request_body": log.get("request_body"),
                    "response_usage": log.get("response_usage"),
                    "response_model": log.get("response_model"),
                    "response_provider": log.get("response_provider"),
                    "response_id": log.get("response_id"),
                }
                plans[sample["sample_id"]] = plan

    def mark_batch_errors(batch: list[dict[str, Any]], exc: Exception) -> None:
        for sample in batch:
            plans[sample["sample_id"]]["query_plan_error"] = str(exc)

    batches = list(_batched(samples, config["max_tasks_per_llm_request"]))
    parallel_requests = min(len(batches), max(1, int(config.get("llm_parallel_requests") or 1)))
    completed = 0
    if parallel_requests <= 1:
        for batch in batches:
            try:
                batch_plans, log = _call_llm_batch(batch, config)
                apply_batch_plans(batch, batch_plans, log)
                logs.append(log)
            except Exception as exc:
                logs.append({"error": str(exc), "sample_count": len(batch)})
                if not config["use_heuristic_fallback_on_llm_failure"]:
                    raise
                mark_batch_errors(batch, exc)
            completed += len(batch)
            if progress_callback:
                progress_callback(completed)
        return plans, logs

    with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
        futures = {executor.submit(_call_llm_batch, batch, config): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                batch_plans, log = future.result()
                apply_batch_plans(batch, batch_plans, log)
                logs.append(log)
            except Exception as exc:
                logs.append({"error": str(exc), "sample_count": len(batch)})
                if not config["use_heuristic_fallback_on_llm_failure"]:
                    raise
                mark_batch_errors(batch, exc)
            completed += len(batch)
            if progress_callback:
                progress_callback(completed)
    return plans, logs


def _best_gold_rank(candidates: Sequence[dict[str, Any]], gold_qids: Sequence[str], rank_field: str = "rank") -> int | None:
    gold_set = set(gold_qids or [])
    ranks = [candidate.get(rank_field) for candidate in candidates if candidate.get("qid") in gold_set]
    ranks = [rank for rank in ranks if isinstance(rank, int)]
    return min(ranks) if ranks else None


def _gt_match_details(candidates: Sequence[dict[str, Any]], gold_qids: Sequence[str], rank_field: str = "rank") -> list[dict[str, Any]]:
    gold_set = set(gold_qids or [])
    details = []
    for candidate in candidates:
        qid = candidate.get("qid")
        if qid in gold_set:
            details.append(
                {
                    "qid": qid,
                    "rank": candidate.get(rank_field) or candidate.get("rank"),
                    "es_rank": candidate.get("es_rank"),
                    "label": candidate.get("label"),
                    "description": candidate.get("description"),
                }
            )
    return details


def _mark_candidates(candidates: list[dict[str, Any]], gold_qids: Sequence[str], selected_qid: str | None) -> list[dict[str, Any]]:
    gold_set = set(gold_qids or [])
    marked = []
    for index, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        item["rank"] = index
        item["alpaca_rank"] = index
        item["ranking_basis"] = "alpaca_return_order"
        item["isGold"] = bool(item.get("qid") in gold_set)
        item["isSelected"] = bool(selected_qid and item.get("qid") == selected_qid)
        item.setdefault("retrieval_stage", "alpaca_search")
        item.setdefault("retrieval_stages", ["alpaca_search"])
        marked.append(item)
    return marked


def _gold_entities_from_candidates(candidates: Sequence[dict[str, Any]], gold_qids: Sequence[str]) -> list[dict[str, Any]]:
    gold_set = set(gold_qids or [])
    seen: set[str] = set()
    entities: list[dict[str, Any]] = []
    for candidate in candidates:
        qid = candidate.get("qid")
        if not qid or qid not in gold_set or qid in seen:
            continue
        seen.add(str(qid))
        entities.append({key: candidate.get(key) for key in ("qid", "label", "description", "coarse_type", "fine_type", "item_category", "wikipedia_url", "dbpedia_url") if candidate.get(key)})
    return entities


def _process_sample(sample: dict[str, Any], query_plan: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    lookup_text = str(query_plan.get("optimized_query") or sample.get("lookup_text") or sample.get("mention_text") or "").strip()
    context_text = _target_context_text(sample)
    query_body = build_alpaca_query(lookup_text, {**query_plan, "context_text": context_text})
    request_payload = {
        **query_body,
        "size": config["max_candidates"],
        "_inspection": {
            "endpoint": os.environ.get("ALPACA_SEARCH_ENDPOINT", ALPACA_METADATA_URL),
            "method": "POST",
            "candidate_count": config["max_candidates"],
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
    backend_request = {"status": "ok", "request": request_payload}
    response_payload: dict[str, Any] = {}
    raw_candidates: list[dict[str, Any]] = []
    error: str | None = None
    try:
        response_payload = alpaca_search(query_body, config["max_candidates"])
        raw_candidates = extract_hits(response_payload)
        backend_request["response_total"] = response_payload.get("hits", {}).get("total")
    except Exception as exc:
        error = str(exc)
        backend_request["status"] = "error"
        backend_request["error"] = error

    selected_qid = str(raw_candidates[0]["qid"]) if raw_candidates and raw_candidates[0].get("qid") else None
    selected_label = str(raw_candidates[0]["label"]) if raw_candidates and raw_candidates[0].get("label") else None
    candidates = _mark_candidates(raw_candidates, sample.get("gt_qids") or [], selected_qid)
    best_rank = _best_gold_rank(candidates, sample.get("gt_qids") or [])
    final_correct = bool(selected_qid and selected_qid in set(sample.get("gt_qids") or []))
    limited_candidates = candidates[: int(config["dashboard_candidate_limit"])]
    gold_qids = list(sample.get("gt_qids") or [])
    return {
        "cell_key": sample["sample_id"],
        "dataset_id": sample["dataset"],
        "table_id": sample["table_id"],
        "row_id": sample["row_id"],
        "col_id": sample["col_id"],
        "mention": sample.get("mention_text"),
        "mention_text": sample.get("mention_text"),
        "lookup_text": sample.get("lookup_text"),
        "gold_qids": gold_qids,
        "gt_qids": gold_qids,
        "primary_gt_qid": sample.get("primary_gt_qid") or (gold_qids[0] if gold_qids else None),
        "gt_raw_value": sample.get("gt_raw_value"),
        "gt_source_file": sample.get("gt_source_file"),
        "gt_entities": _gold_entities_from_candidates(candidates, gold_qids),
        "selected_qid": selected_qid,
        "selected_label": selected_label,
        "selected_correct": final_correct,
        "final_correct": final_correct,
        "coverage_correct": best_rank is not None,
        "hit_at_1": best_rank == 1,
        "hit_at_5": bool(best_rank and best_rank <= 5),
        "hit_at_10": bool(best_rank and best_rank <= 10),
        "hit_at_k": best_rank is not None,
        "best_gt_rank": best_rank,
        "gold_rank": best_rank,
        "gt_match_details": _gt_match_details(candidates, gold_qids),
        "retrieved_count": len(candidates),
        "candidate_count": len(candidates),
        "candidate_backend": "alpaca",
        "query_engine": query_plan.get("query_plan_source") or "heuristic",
        "query_plan": query_plan,
        "query_text": lookup_text,
        "candidates": limited_candidates,
        "backend_requests": [backend_request],
        "backend_response": response_payload if config["save_full_debug_output"] else {},
        "retrieval_error": error,
        "table_context": _table_context_preview(sample, config),
        "header": sample.get("header") or [],
        "header_cell": sample.get("header_cell"),
        "target_row": sample.get("target_row") or [],
        "context_rows": {
            "before": sample.get("rows_before") or [],
            "after": sample.get("rows_after") or [],
        },
        "lookup_context": sample.get("lookup_context") or [],
        "original_table_name": sample.get("original_table_name"),
        "table_num_rows": sample.get("table_num_rows"),
        "table_num_cols": sample.get("table_num_cols"),
    }


def _column_stats(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_col: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_col[int(sample["col_id"])].append(sample)
    stats = []
    for col_id, col_samples in sorted(by_col.items()):
        headers: dict[str, int] = defaultdict(int)
        for sample in col_samples:
            headers[str(sample.get("header_cell") or "")] += 1
        mentions = [str(sample.get("mention_text") or "") for sample in col_samples]
        stats.append(
            {
                "col_id": col_id,
                "header": max(headers.items(), key=lambda item: item[1])[0] if headers else "",
                "sample_count": len(col_samples),
                "unique_mentions": len(set(mentions)),
            }
        )
    return stats


def _build_table_profile(samples: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if not samples:
        return {}
    first = samples[0]
    return {
        "dataset_id": first["dataset"],
        "table_id": first["table_id"],
        "original_table_name": first.get("original_table_name"),
        "num_rows": first.get("table_num_rows"),
        "num_cols": first.get("table_num_cols"),
        "header": first.get("header") or [],
        "sampled_records": len(samples),
        "context_rows_per_record": config["context_rows"],
        "column_stats": _column_stats(samples),
        "source_storage": "filesystem_lazy",
    }


def _summarize_llm_logs(llm_logs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    call_count = 0
    error_count = 0
    for log in llm_logs:
        if log.get("error"):
            error_count += 1
            continue
        call_count += 1
        usage = log.get("response_usage") or {}
        total_prompt += int(usage.get("prompt_tokens") or 0)
        total_completion += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
    return {
        "llm_call_count": call_count,
        "llm_error_count": error_count,
        "llm_prompt_tokens": total_prompt,
        "llm_completion_tokens": total_completion,
        "llm_total_tokens": total_tokens,
    }


def run_experiment_job(job_id: int) -> None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM experiment_jobs WHERE id = %s", (job_id,)).fetchone()
    if not row:
        return
    config = normalize_experiment_config(dict(row["config"] or {}))
    stage_progress: dict[str, Any] = {}
    stage_timers: dict[str, tuple[datetime, float]] = {}

    def set_stage_progress(
        key: str,
        *,
        label: str,
        current: int,
        total: int,
        status: str = "running",
        message: str | None = None,
        stage: str | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
    ) -> None:
        if key not in stage_timers:
            stage_timers[key] = (_now(), time.monotonic())
        started_wall, started_monotonic = stage_timers[key]
        stage_progress[key] = _stage_entry(label=label, current=current, total=total, started_wall=started_wall, started_monotonic=started_monotonic, status=status)
        update_fields: dict[str, Any] = {"stage_progress": stage_progress}
        if message is not None:
            update_fields["message"] = message
        if stage is not None:
            update_fields["stage"] = stage
        if progress_current is not None:
            update_fields["progress_current"] = progress_current
        if progress_total is not None:
            update_fields["progress_total"] = progress_total
        update_job(job_id, **update_fields)

    try:
        run_started_at = _now()
        run_name = _job_name(job_id, config)
        update_job(
            job_id,
            status="running",
            stage="sampling",
            started_at=run_started_at,
            config=config,
            output_path=None,
            query_plan_output_path=None,
            message="Discovering source metadata and sampling filesystem-backed mentions",
        )

        sample_bundle = build_random_sample_bundle_from_db(config)
        samples = list(sample_bundle.get("samples") or [])
        if not samples:
            raise RuntimeError("No source metadata found in Postgres. Run ./scripts/seed_source_data.sh to populate source_datasets and source_tables.")
        update_job(job_id, stage="sampled", progress_current=0, progress_total=len(samples), message=f"Sampled {len(samples)} mentions from filesystem-backed source metadata")

        set_stage_progress(
            "query_plans",
            label=f"{_llm_label(config)} query plans" if config["llm_enabled"] else "Heuristic query plans",
            current=0,
            total=len(samples),
            stage="query_plans",
            progress_current=0,
            progress_total=len(samples),
            message=f"Generating query plans for {len(samples)} sampled mentions",
        )
        query_plan_completed = 0

        def update_query_plan_progress(completed: int) -> None:
            nonlocal query_plan_completed
            query_plan_completed = completed
            set_stage_progress(
                "query_plans",
                label=f"{_llm_label(config)} query plans" if config["llm_enabled"] else "Heuristic query plans",
                current=completed,
                total=len(samples),
                stage="query_plans",
                progress_current=completed,
                progress_total=len(samples),
                message=f"Generated query plans for {completed}/{len(samples)} mentions",
            )

        try:
            query_plans, llm_logs = _generate_query_plans(samples, config, progress_callback=update_query_plan_progress)
        except Exception:
            set_stage_progress(
                "query_plans",
                label=f"{_llm_label(config)} query plans" if config["llm_enabled"] else "Heuristic query plans",
                current=query_plan_completed,
                total=len(samples),
                status="failed",
                stage="query_plans",
                progress_current=query_plan_completed,
                progress_total=len(samples),
                message="Query planning failed",
            )
            raise
        set_stage_progress(
            "query_plans",
            label=f"{_llm_label(config)} query plans" if config["llm_enabled"] else "Heuristic query plans",
            current=len(samples),
            total=len(samples),
            status="completed",
            stage="query_plans",
            progress_current=len(samples),
            progress_total=len(samples),
            message=f"Generated query plans for {len(samples)} mentions",
        )

        samples_by_table: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            samples_by_table[(sample["dataset"], sample["table_id"])].append(sample)
        table_profiles = {key: _build_table_profile(table_samples, config) for key, table_samples in samples_by_table.items()}

        source_path = f"experiment-job://{job_id}"
        with connect() as conn:
            with conn.transaction():
                run_id = create_import_run(
                    conn,
                    name=run_name,
                    source_path=source_path,
                    source_filename=f"{run_name} (database)",
                    raw_summary={"status": "running", "evaluated": 0},
                    raw_sampling_config=_sampling_config(config, run_started_at),
                    replace_existing=True,
                )
                table_ids = {
                    key: upsert_experiment_table(
                        conn,
                        run_id=run_id,
                        dataset_id=key[0],
                        table_id=key[1],
                        sample_limit=config["records_per_table"],
                        raw_profile=profile,
                        raw_payload=profile,
                    )
                    for key, profile in table_profiles.items()
                }
                update_run_counters(conn, run_id=run_id, table_count=len(table_ids))
        update_job(job_id, imported_run_id=run_id)

        lock = threading.Lock()
        completed = 0
        imported_mentions = 0
        imported_candidates = 0
        imported_covered = 0
        hard_failed_mentions = 0
        started = time.time()

        def process_and_import_sample(sample: dict[str, Any]) -> None:
            nonlocal completed, imported_mentions, imported_candidates, imported_covered, hard_failed_mentions
            result = _process_sample(sample, query_plans[sample["sample_id"]], config)
            key = (result["dataset_id"], result["table_id"])
            result["table_profile"] = table_profiles.get(key, {})
            backend_request = (result.get("backend_requests") or [{}])[0]
            hard_failed = backend_request.get("status") == "error" and not result.get("candidates")
            with connect() as conn:
                with conn.transaction():
                    row_stats = import_experiment_row(conn, run_id=run_id, table_db_id=table_ids.get(key), row=result)
                    update_run_counters(
                        conn,
                        run_id=run_id,
                        mention_delta=row_stats.mention_count,
                        candidate_delta=row_stats.candidate_count,
                        covered_delta=row_stats.covered_count,
                    )
            with lock:
                completed += 1
                imported_mentions += row_stats.mention_count
                imported_candidates += row_stats.candidate_count
                imported_covered += row_stats.covered_count
                hard_failed_mentions += 1 if hard_failed else 0
                set_stage_progress(
                    "alpaca_search",
                    label="Alpaca candidate search",
                    current=completed,
                    total=len(samples),
                    stage="alpaca_search",
                    progress_current=completed,
                    progress_total=len(samples),
                    message=f"Fetched and stored candidates for {completed}/{len(samples)} mentions",
                )

        effective_workers = config["max_workers"]
        if config["max_candidates"] > 500:
            effective_workers = min(effective_workers, 2)
        set_stage_progress(
            "alpaca_search",
            label="Alpaca candidate search",
            current=0,
            total=len(samples),
            stage="alpaca_search",
            progress_current=0,
            progress_total=len(samples),
            message=f"Running Alpaca searches with max_candidates={config['max_candidates']} and workers={effective_workers}",
        )
        if effective_workers <= 1:
            for sample in samples:
                process_and_import_sample(sample)
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = [executor.submit(process_and_import_sample, sample) for sample in samples]
                for future in as_completed(futures):
                    future.result()
        runtime_ms = round((time.time() - started) * 1000, 1)
        set_stage_progress(
            "alpaca_search",
            label="Alpaca candidate search",
            current=len(samples),
            total=len(samples),
            status="completed",
            stage="alpaca_search",
            progress_current=len(samples),
            progress_total=len(samples),
            message=f"Stored candidates for {len(samples)}/{len(samples)} mentions",
        )

        allowed_failures = max(5, int(len(samples) * 0.05))
        if hard_failed_mentions > allowed_failures:
            with connect() as conn:
                with conn.transaction():
                    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
            update_job(job_id, imported_run_id=None)
            raise RuntimeError(
                f"Alpaca retrieval failed for {hard_failed_mentions}/{len(samples)} mentions. Partial run was discarded."
            )

        update_job(job_id, stage="finalizing", message="Finalizing Postgres run")
        overall_summary = {
            "evaluated": imported_mentions,
            "gold_retrieved": imported_covered,
            "recall_at_retrieval_top_k": imported_covered / imported_mentions if imported_mentions else None,
            "retrieval_top_k": config["max_candidates"],
            "effective_workers": effective_workers,
            "hard_failed_mentions": hard_failed_mentions,
            "runtime_ms": runtime_ms,
            "datasets_evaluated": len({sample["dataset"] for sample in samples}),
            "tables_evaluated": len(table_ids),
            "candidate_count": imported_candidates,
            "source_storage": "filesystem_lazy",
            "run_storage": "postgres_incremental",
            **_summarize_llm_logs(llm_logs),
        }
        with connect() as conn:
            with conn.transaction():
                stats = finalize_import_run(conn, run_id=run_id, raw_summary=overall_summary)

        update_job(
            job_id,
            status="completed",
            stage="completed",
            progress_current=len(samples),
            progress_total=len(samples),
            message=f"Completed and imported run {stats.run_id}",
            imported_run_id=stats.run_id,
            finished_at=_now(),
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            stage="failed",
            error=f"{exc}\n{traceback.format_exc()}",
            message=str(exc),
            finished_at=_now(),
        )
