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
CEREBRAS_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-oss-120b": (0.35, 0.75),
    "openai/gpt-oss-120b": (0.35, 0.75),
    "gemma-4-31b": (0.99, 1.49),
    "zai-glm-4.7": (2.25, 2.75),
}


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
    try:
        import tiktoken  # type: ignore[import-not-found]

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.encoding_for_model("gpt-4")
        return max(1, len(encoding.encode(text)))
    except Exception:
        pass
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


def _per_million_to_per_token(value: float | None) -> float | None:
    return value / 1_000_000 if value is not None else None


def _provider_pricing(model: str, provider: str, config: dict[str, Any]) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    provider_name = provider.strip().lower()
    model_id = model.strip()
    if provider_name == "openrouter" and model_id:
        model_info = openrouter_model_info(model_id, config)
        pricing = model_info.get("pricing") if isinstance(model_info, dict) and isinstance(model_info.get("pricing"), dict) else {}
        return pricing, "openrouter_models_api" if pricing else None, model_info if isinstance(model_info, dict) else None

    input_per_million = _as_price(
        config.get("llm_input_price_per_million")
        or config.get("input_price_per_million")
        or os.environ.get("LLM_INPUT_PRICE_PER_MILLION")
        or os.environ.get(f"{provider_name.upper()}_INPUT_PRICE_PER_MILLION")
    )
    output_per_million = _as_price(
        config.get("llm_output_price_per_million")
        or config.get("output_price_per_million")
        or os.environ.get("LLM_OUTPUT_PRICE_PER_MILLION")
        or os.environ.get(f"{provider_name.upper()}_OUTPUT_PRICE_PER_MILLION")
    )
    pricing_source = "configured_provider_pricing" if input_per_million is not None and output_per_million is not None else None

    if provider_name == "cerebras" and (input_per_million is None or output_per_million is None):
        default_prices = CEREBRAS_PRICING_PER_MILLION.get(model_id.casefold())
        if default_prices:
            input_per_million, output_per_million = default_prices
            pricing_source = "cerebras_public_pricing"

    if input_per_million is None or output_per_million is None:
        return {}, None, None

    return {
        "input": _per_million_to_per_token(input_per_million),
        "output": _per_million_to_per_token(output_per_million),
    }, pricing_source, {
        "id": model_id,
        "name": model_id,
        "top_provider": {"pricing_source": pricing_source},
    }


def _safe_estimate_pricing(pricing: dict[str, Any], prompt_tokens: int, completion_tokens: int, *, request_count: int = 1) -> dict[str, Any]:
    cost = _pricing_cost(pricing, prompt_tokens, completion_tokens, request_count=request_count, cost_multiplier=2.0)
    cost.pop("estimated_subtotal_cost_usd", None)
    cost.pop("cost_safety_multiplier", None)
    return cost


def _usage_number(usage: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = usage.get(key)
        if value is None and "." in key:
            current: Any = usage
            for part in key.split("."):
                current = current.get(part) if isinstance(current, dict) else None
                if current is None:
                    break
            value = current
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _pricing_cost(
    pricing: dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
    *,
    request_count: int = 1,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    prompt_price = _as_price(pricing.get("prompt"))
    if prompt_price is None:
        prompt_price = _as_price(pricing.get("input"))
    completion_price = _as_price(pricing.get("completion"))
    if completion_price is None:
        completion_price = _as_price(pricing.get("output"))
    request_price = (_as_price(pricing.get("request")) or 0.0) * max(1, request_count)
    prompt_cost = prompt_tokens * prompt_price if prompt_price is not None else None
    completion_cost = completion_tokens * completion_price if completion_price is not None else None
    total_cost = None
    subtotal_cost = None
    if prompt_cost is not None and completion_cost is not None:
        subtotal_cost = prompt_cost + completion_cost + request_price
        total_cost = subtotal_cost * max(0.0, float(cost_multiplier))
    return {
        "input_per_token": prompt_price,
        "output_per_token": completion_price,
        "prompt_per_token": prompt_price,
        "completion_per_token": completion_price,
        "request": request_price,
        "input_per_million": prompt_price * 1_000_000 if prompt_price is not None else None,
        "output_per_million": completion_price * 1_000_000 if completion_price is not None else None,
        "prompt_per_million": prompt_price * 1_000_000 if prompt_price is not None else None,
        "completion_per_million": completion_price * 1_000_000 if completion_price is not None else None,
        "estimated_input_cost_usd": prompt_cost,
        "estimated_output_cost_usd": completion_cost,
        "estimated_prompt_cost_usd": prompt_cost,
        "estimated_completion_cost_usd": completion_cost,
        "estimated_subtotal_cost_usd": subtotal_cost,
        "estimated_total_cost_usd": total_cost,
        "cost_safety_multiplier": max(0.0, float(cost_multiplier)),
    }


def _usage_tokens_or_estimate(
    *,
    usage: dict[str, Any],
    input_text: str | None = None,
    output_text: str | None = None,
) -> tuple[int, int, int, str]:
    prompt_tokens = _usage_number(usage, "prompt_tokens", "input_tokens", "promptTokens", "inputTokens")
    completion_tokens = _usage_number(
        usage,
        "completion_tokens",
        "output_tokens",
        "completionTokens",
        "outputTokens",
    )
    token_source = "llm_response_usage"
    if prompt_tokens is None:
        prompt_tokens = _estimate_text_tokens(input_text or "") if input_text else 0
        if prompt_tokens:
            token_source = "tokenizer_estimate_from_payload"
    if completion_tokens is None:
        completion_tokens = _estimate_text_tokens(output_text or "") if output_text else 0
        if completion_tokens:
            token_source = "tokenizer_estimate_from_payload" if token_source != "llm_response_usage" else "mixed_response_usage_and_tokenizer_estimate"
    prompt_int = int(prompt_tokens or 0)
    completion_int = int(completion_tokens or 0)
    total_tokens = int(_usage_number(usage, "total_tokens", "totalTokens") or (prompt_int + completion_int))
    return prompt_int, completion_int, total_tokens, token_source


def _response_reported_cost(usage: dict[str, Any]) -> dict[str, Any] | None:
    upstream_total = _usage_number(usage, "cost_details.upstream_inference_cost")
    upstream_prompt = _usage_number(usage, "cost_details.upstream_inference_prompt_cost")
    upstream_completion = _usage_number(usage, "cost_details.upstream_inference_completions_cost")
    if upstream_total is not None and upstream_total > 0:
        return {
            "total_cost_usd": upstream_total,
            "input_cost_usd": upstream_prompt,
            "output_cost_usd": upstream_completion,
            "pricing_source": "llm_response_usage.cost_details",
            "notes": ["Cost uses upstream inference cost details reported by the LLM service response."],
        }

    direct_total = _usage_number(
        usage,
        "total_cost_usd",
        "cost_usd",
        "total_cost",
        "cost",
        "billing.cost_usd",
        "billing.cost",
    )
    if direct_total is None:
        return None
    return {
        "total_cost_usd": direct_total,
        "input_cost_usd": None,
        "output_cost_usd": None,
        "pricing_source": "llm_response_usage",
        "notes": ["Cost was reported by the LLM service response usage metadata."],
    }


def llm_usage_cost_from_metadata(
    *,
    provider: str | None,
    model: str | None,
    usage: Any,
    config: dict[str, Any] | None = None,
    request_count: int = 1,
    input_text: str | None = None,
    output_text: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        usage = {}
    if not usage and not (input_text or output_text):
        return None

    prompt_tokens, completion_tokens, total_tokens, token_source = _usage_tokens_or_estimate(
        usage=usage,
        input_text=input_text,
        output_text=output_text,
    )
    response_reported = _response_reported_cost(usage)
    if response_reported is not None and response_reported["total_cost_usd"] > 0:
        return {
            "cost_kind": "response_reported",
            "pricing_source": response_reported["pricing_source"],
            "total_cost_usd": response_reported["total_cost_usd"],
            "input_cost_usd": response_reported["input_cost_usd"],
            "output_cost_usd": response_reported["output_cost_usd"],
            "prompt_cost_usd": response_reported["input_cost_usd"],
            "completion_cost_usd": response_reported["output_cost_usd"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "token_source": token_source,
            "request_count": max(1, int(request_count)),
            "notes": response_reported["notes"],
        }

    provider_name = str(provider or "").strip().lower()
    model_id = str(model or "").strip()
    if not model_id or not (prompt_tokens or completion_tokens):
        return {
            "cost_kind": "actual_tokens_no_price",
            "pricing_source": None,
            "total_cost_usd": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "token_source": token_source,
            "request_count": max(1, int(request_count)),
            "notes": ["Input/output tokens were measured or estimated, but no endpoint-reported USD cost or model pricing was available."],
        }

    try:
        pricing, pricing_source, model_info = _provider_pricing(model_id, provider_name, config or {})
    except RuntimeError as exc:
        return {
            "cost_kind": "actual_tokens_price_lookup_failed",
            "pricing_source": None,
            "total_cost_usd": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "token_source": token_source,
            "request_count": max(1, int(request_count)),
            "notes": [f"Token usage was reported, but provider pricing lookup failed: {exc}"],
        }

    if not pricing:
        return {
            "cost_kind": "actual_tokens_no_price",
            "pricing_source": None,
            "total_cost_usd": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "token_source": token_source,
            "request_count": max(1, int(request_count)),
            "notes": ["Token usage was reported, but no provider pricing is configured for this model."],
        }

    cost = _pricing_cost(pricing, prompt_tokens, completion_tokens, request_count=request_count)
    return {
        "cost_kind": "actual_tokens_provider_pricing",
        "pricing_source": pricing_source,
        "total_cost_usd": cost.get("estimated_total_cost_usd"),
        "input_cost_usd": cost.get("estimated_input_cost_usd"),
        "output_cost_usd": cost.get("estimated_output_cost_usd"),
        "prompt_cost_usd": cost.get("estimated_prompt_cost_usd"),
        "completion_cost_usd": cost.get("estimated_completion_cost_usd"),
        "request_cost_usd": cost.get("request"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_source": token_source,
        "request_count": max(1, int(request_count)),
        "pricing": {
            "input_per_token": cost.get("input_per_token"),
            "output_per_token": cost.get("output_per_token"),
            "prompt_per_token": cost.get("prompt_per_token"),
            "completion_per_token": cost.get("completion_per_token"),
            "input_per_million": cost.get("input_per_million"),
            "output_per_million": cost.get("output_per_million"),
            "prompt_per_million": cost.get("prompt_per_million"),
            "completion_per_million": cost.get("completion_per_million"),
        },
        "model_info": {
            "id": model_info.get("id"),
            "name": model_info.get("name"),
            "top_provider": model_info.get("top_provider"),
        }
        if isinstance(model_info, dict)
        else None,
        "notes": ["Cost uses separate input and output token counts with provider pricing."],
    }


def estimate_llm_usage(
    *,
    input_text: str,
    config: dict[str, Any],
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    model = str(config.get("llm_model") or config.get("openrouter_model") or "").strip()
    prompt_tokens = _estimate_text_tokens(input_text) + _estimate_chat_overhead_tokens(input_text)
    if max_completion_tokens is not None:
        completion_tokens = max(0, int(max_completion_tokens))
        completion_source = "requested_max_completion_tokens"
    elif config.get("llm_max_tokens") is not None:
        completion_tokens = max(0, int(config.get("llm_max_tokens") or 0))
        completion_source = "configured_llm_max_tokens"
    else:
        completion_tokens = max(64, min(4096, math.ceil(prompt_tokens * 0.35)))
        completion_source = "tokenizer_output_heuristic"
    provider = str(config.get("llm_provider") or "").strip().lower()

    pricing, pricing_source, model_info = _provider_pricing(model, provider, config) if provider and model else ({}, None, None)
    cost = _safe_estimate_pricing(pricing, prompt_tokens, completion_tokens) if pricing else {}

    return {
        "model": model,
        "provider": provider or None,
        "route_provider": str(config.get("llm_provider_name") or "").strip() or None,
        "estimation_method": "local_char_word_heuristic",
        "token_estimate": {
            "prompt_tokens": prompt_tokens,
            "input_tokens": prompt_tokens,
            "max_completion_tokens": completion_tokens,
            "estimated_completion_tokens": completion_tokens,
            "output_tokens": completion_tokens,
            "completion_token_source": completion_source,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "pricing_source": pricing_source,
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
            "Input tokens are estimated locally before generation; exact native usage is returned by the LLM service after a real request when available.",
            "Output tokens are estimated when no max completion token limit is supplied.",
            "Prices are fetched from provider pricing when available and are USD per token/request.",
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
    pricing, pricing_source, model_info = _provider_pricing(model, provider, config) if provider and model else ({}, None, None)
    cost = _safe_estimate_pricing(pricing, prompt_tokens, completion_tokens, request_count=len(batches)) if pricing else {}

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
            "input_tokens": prompt_tokens,
            "estimated_completion_tokens": completion_tokens,
            "output_tokens": completion_tokens,
            "completion_tokens_per_request": completion_per_request,
            "completion_token_source": completion_source,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "pricing_source": pricing_source,
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
            "Prices are fetched from provider pricing when available and are USD per token/request.",
        ],
    }
