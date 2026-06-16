from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


OPENROUTER_API_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").rstrip("/")
_MODEL_CACHE_TTL_SECONDS = int(os.environ.get("OPENROUTER_MODEL_CACHE_TTL_SECONDS", "3600"))
_MODEL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _api_key(config: dict[str, Any]) -> str:
    return str(
        config.get("llm_api_key")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or ""
    ).strip()


def _headers(config: dict[str, Any]) -> dict[str, str]:
    headers = {"accept": "application/json", "User-Agent": "coverage-dashboard/1.0"}
    token = _api_key(config)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json(url: str, config: dict[str, Any], *, timeout: int = 20) -> dict[str, Any]:
    request = Request(url, headers=_headers(config), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read() or b"{}")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter pricing lookup HTTP {exc.code}: {detail[:500]}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenRouter pricing lookup failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OpenRouter pricing lookup returned a non-object response")
    return payload


def _cached_model(model: str) -> dict[str, Any] | None:
    cached = _MODEL_CACHE.get(model)
    if not cached:
        return None
    cached_at, payload = cached
    if time.time() - cached_at > _MODEL_CACHE_TTL_SECONDS:
        _MODEL_CACHE.pop(model, None)
        return None
    return payload


def _cache_model(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    _MODEL_CACHE[model] = (time.time(), payload)
    return payload


def openrouter_model_info(model: str, config: dict[str, Any]) -> dict[str, Any] | None:
    model_id = str(model or "").strip()
    if not model_id:
        return None
    cached = _cached_model(model_id)
    if cached is not None:
        return cached

    model_path = "/".join(quote(part, safe=":") for part in model_id.split("/"))
    try:
        payload = _get_json(f"{OPENROUTER_API_BASE}/model/{model_path}", config)
        data = payload.get("data")
        if isinstance(data, dict):
            return _cache_model(model_id, data)
    except RuntimeError:
        pass

    query = urlencode({"q": model_id, "output_modalities": "text"})
    payload = _get_json(f"{OPENROUTER_API_BASE}/models?{query}", config)
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("id") == model_id:
                return _cache_model(model_id, item)
    return None


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"\S+", text)
    by_chars = math.ceil(len(text) / 4)
    by_words = math.ceil(len(words) * 1.35)
    return max(1, by_chars, by_words)


def _estimate_chat_overhead_tokens(input_text: str) -> int:
    lines = [line for line in input_text.splitlines() if line.strip()]
    return 8 + min(64, len(lines) * 2)


def _as_price(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pricing_cost(
    pricing: dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
    *,
    request_count: int = 1,
) -> dict[str, Any]:
    prompt_price = _as_price(pricing.get("prompt"))
    completion_price = _as_price(pricing.get("completion"))
    request_price = (_as_price(pricing.get("request")) or 0.0) * max(1, request_count)
    prompt_cost = prompt_tokens * prompt_price if prompt_price is not None else None
    completion_cost = completion_tokens * completion_price if completion_price is not None else None
    total_cost = None
    if prompt_cost is not None and completion_cost is not None:
        total_cost = prompt_cost + completion_cost + request_price
    return {
        "prompt_per_token": prompt_price,
        "completion_per_token": completion_price,
        "request": request_price,
        "prompt_per_million": prompt_price * 1_000_000 if prompt_price is not None else None,
        "completion_per_million": completion_price * 1_000_000 if completion_price is not None else None,
        "estimated_prompt_cost_usd": prompt_cost,
        "estimated_completion_cost_usd": completion_cost,
        "estimated_total_cost_usd": total_cost,
    }


def estimate_llm_usage(
    *,
    input_text: str,
    config: dict[str, Any],
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    model = str(config.get("llm_model") or config.get("openrouter_model") or "").strip()
    prompt_tokens = _estimate_text_tokens(input_text) + _estimate_chat_overhead_tokens(input_text)
    completion_tokens = int(max_completion_tokens if max_completion_tokens is not None else config.get("llm_max_tokens") or 0)
    completion_tokens = max(0, completion_tokens)
    provider = str(config.get("llm_provider") or "").strip().lower()

    model_info = openrouter_model_info(model, config) if provider == "openrouter" and model else None
    pricing = model_info.get("pricing") if isinstance(model_info, dict) and isinstance(model_info.get("pricing"), dict) else {}
    cost = _pricing_cost(pricing, prompt_tokens, completion_tokens) if pricing else {}

    return {
        "model": model,
        "provider": provider or None,
        "route_provider": str(config.get("llm_provider_name") or "").strip() or None,
        "estimation_method": "local_char_word_heuristic",
        "token_estimate": {
            "prompt_tokens": prompt_tokens,
            "max_completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "pricing_source": "openrouter_models_api" if pricing else None,
        "pricing": cost or None,
        "model_info": {
            "id": model_info.get("id"),
            "name": model_info.get("name"),
            "context_length": model_info.get("context_length"),
            "top_provider": model_info.get("top_provider"),
        }
        if isinstance(model_info, dict)
        else None,
        "notes": [
            "Prompt tokens are estimated before generation; exact native usage is returned by OpenRouter after a real request.",
            "Prices are fetched from OpenRouter's model catalog when available and are USD per token/request.",
        ],
    }


def estimate_experiment_llm_usage(config: dict[str, Any]) -> dict[str, Any]:
    from .datasets import build_random_sample_bundle_from_db
    from .experiment_runner import _batched, _llm_batch_body

    batch_size = max(1, int(config.get("max_tasks_per_llm_request") or 1))
    sample_bundle = build_random_sample_bundle_from_db(config)
    samples = list(sample_bundle.get("samples") or [])
    batches = list(_batched(samples, batch_size))

    prompt_tokens = 0
    request_estimates: list[dict[str, Any]] = []
    for index, batch in enumerate(batches, start=1):
        body = _llm_batch_body(batch, config)
        serialized_messages = json.dumps(body.get("messages") or [], ensure_ascii=False)
        batch_prompt_tokens = _estimate_text_tokens(serialized_messages) + _estimate_chat_overhead_tokens(serialized_messages)
        prompt_tokens += batch_prompt_tokens
        request_estimates.append(
            {
                "batch_index": index,
                "task_count": len(batch),
                "prompt_tokens": batch_prompt_tokens,
            }
        )

    if config.get("llm_max_tokens") is not None:
        completion_per_request = max(0, int(config.get("llm_max_tokens") or 0))
        completion_source = "configured_llm_max_tokens"
    else:
        completion_per_request = max(180, 90 * min(batch_size, max(1, len(samples))))
        completion_source = "json_plan_heuristic_per_batch"
    completion_tokens = completion_per_request * len(batches)

    provider = str(config.get("llm_provider") or "").strip().lower()
    model = str(config.get("llm_model") or config.get("openrouter_model") or "").strip()
    model_info = openrouter_model_info(model, config) if provider == "openrouter" and model else None
    pricing = model_info.get("pricing") if isinstance(model_info, dict) and isinstance(model_info.get("pricing"), dict) else {}
    cost = _pricing_cost(pricing, prompt_tokens, completion_tokens, request_count=len(batches)) if pricing else {}

    return {
        "model": model,
        "provider": provider or None,
        "route_provider": str(config.get("llm_provider_name") or "").strip() or None,
        "estimation_method": "sampled_experiment_llm_batches",
        "target": {
            "sampled_mentions": len(samples),
            "llm_request_count": len(batches),
            "max_tasks_per_llm_request": batch_size,
            "dataset_inventory": sample_bundle.get("dataset_inventory") or [],
            "sampling_manifest": sample_bundle.get("sampling_manifest") or [],
            "warnings": sample_bundle.get("warnings") or [],
        },
        "token_estimate": {
            "prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": completion_tokens,
            "completion_tokens_per_request": completion_per_request,
            "completion_token_source": completion_source,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "pricing_source": "openrouter_models_api" if pricing else None,
        "pricing": cost or None,
        "model_info": {
            "id": model_info.get("id"),
            "name": model_info.get("name"),
            "context_length": model_info.get("context_length"),
            "top_provider": model_info.get("top_provider"),
        }
        if isinstance(model_info, dict)
        else None,
        "request_estimates": request_estimates[:50],
        "notes": [
            "Prompt tokens are estimated from the same sampled LLM batch messages used by the experiment runner.",
            "Completion tokens are an estimate unless llm_max_tokens is configured.",
            "Prices are fetched from OpenRouter's model catalog when available and are USD per token/request.",
        ],
    }
