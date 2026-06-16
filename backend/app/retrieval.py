from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .env import load_dotenv


load_dotenv()

ALPACA_METADATA_URL = os.environ.get(
    "ALPACA_METADATA_URL",
    "https://alpaca.zooverse.dev/debug/elasticsearch/alpaca-entities/_search",
)

DEFAULT_RETRIEVAL_CANDIDATES = 100
MAX_RETRIEVAL_CANDIDATES = 1000
MAX_RETURNED_CANDIDATES = 1000
TYPO_CORRECTION_CONFIDENCE_THRESHOLD = 0.85
ALPACA_SEARCH_CACHE_MAX_ENTRIES = int(os.environ.get("ALPACA_SEARCH_CACHE_MAX_ENTRIES", "256"))

TEXT_MATCH_FIELDS = [
    "label^6",
    "labels^4",
    "aliases^4",
    "context_string^1.5",
    "description^1.25",
    "search_text",
]

SOFT_KEYWORD_BOOSTS = {
    "coarse_type": 8.0,
    "fine_type": 10.0,
    "wikipedia_url": 25.0,
    "dbpedia_url": 25.0,
}

_ALPACA_SEARCH_CACHE: dict[str, tuple[int, dict[str, Any]]] = {}
_ALPACA_SEARCH_CACHE_LOCK = threading.Lock()


def alpaca_token() -> str:
    return os.environ.get("ALPACA_TOKEN", "").strip()


def _bounded_int(value: int | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def bounded_candidate_count(value: int | None, default: int = DEFAULT_RETRIEVAL_CANDIDATES) -> int:
    return _bounded_int(value, default=default, minimum=1, maximum=MAX_RETRIEVAL_CANDIDATES)


def bounded_returned_candidate_count(value: int | None, default: int = MAX_RETURNED_CANDIDATES) -> int:
    return _bounded_int(value, default=default, minimum=1, maximum=MAX_RETURNED_CANDIDATES)


def _clean_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _clean_string_list(value: Any, *, limit: int = 8) -> list[str]:
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = []
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        text = _clean_string(raw)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def normalize_url_slug(value: Any) -> str | None:
    text = _clean_string(value)
    if not text:
        return None
    text = text.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = text.strip().replace(" ", "_")
    return text or None


def build_alpaca_query(query_text: str, retrieval_signals: dict[str, Any] | None = None) -> dict[str, Any]:
    signals = retrieval_signals or {}
    optimized_query = _clean_string(signals.get("optimized_query")) or query_text
    typo_corrected_mention = _clean_string(signals.get("typo_corrected_mention"))
    try:
        typo_confidence = (
            float(signals["typo_correction_confidence"])
            if signals.get("typo_correction_confidence") is not None
            else None
        )
    except (TypeError, ValueError):
        typo_confidence = None
    if (
        typo_corrected_mention
        and typo_confidence is not None
        and typo_confidence >= TYPO_CORRECTION_CONFIDENCE_THRESHOLD
    ):
        optimized_query = typo_corrected_mention
    normalized_mention = _clean_string(signals.get("normalized_mention"))
    context_text = _clean_string(signals.get("context_text"))
    aliases = _clean_string_list(signals.get("aliases"), limit=10)
    expansion_terms = _clean_string_list(signals.get("context_expansion_terms"), limit=6)
    should: list[dict[str, Any]] = [
        {
            "multi_match": {
                "query": optimized_query,
                "fields": TEXT_MATCH_FIELDS,
                "type": "best_fields",
                "operator": "or",
                "boost": 4.0,
            }
        }
    ]

    if normalized_mention and normalized_mention.casefold() != optimized_query.casefold():
        should.append(
            {
                "multi_match": {
                    "query": normalized_mention,
                    "fields": ["label^5", "labels^3", "aliases^3", "search_text"],
                    "type": "best_fields",
                    "operator": "or",
                    "boost": 2.5,
                }
            }
        )

    for alias in aliases:
        should.append(
            {
                "multi_match": {
                    "query": alias,
                    "fields": ["aliases^6", "labels^4", "label^4", "search_text"],
                    "type": "phrase",
                    "boost": 3.0,
                }
            }
        )

    if context_text:
        should.append(
            {
                "multi_match": {
                    "query": context_text,
                    "fields": ["context_string^2", "description^1.5", "search_text"],
                    "operator": "or",
                    "boost": 1.0,
                }
            }
        )

    for term in expansion_terms:
        should.append(
            {
                "multi_match": {
                    "query": term,
                    "fields": ["context_string^2", "description^1.5", "aliases", "search_text"],
                    "operator": "or",
                    "boost": 0.75,
                }
            }
        )

    for field, boost in SOFT_KEYWORD_BOOSTS.items():
        value = signals.get(field)
        if field in {"wikipedia_url", "dbpedia_url"}:
            value = normalize_url_slug(value)
        else:
            value = _clean_string(value)
        if value:
            should.append({"term": {field: {"value": value, "boost": boost}}})

    return {
        "track_total_hits": False,
        "_source": True,
        "query": {
            "bool": {
                "filter": [{"terms": {"item_category": ["ENTITY", "TYPE"]}}],
                "should": should,
                "minimum_should_match": 1,
            }
        },
        "sort": [
            {"_score": {"order": "desc"}},
            {"popularity": {"order": "desc"}},
        ],
    }


def _alpaca_cache_key(query_body: dict[str, Any]) -> str:
    payload = dict(query_body)
    payload.pop("size", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _slice_alpaca_response(response: dict[str, Any], size: int) -> dict[str, Any]:
    bounded_size = bounded_candidate_count(size)
    result = copy.deepcopy(response)
    hits = result.get("hits")
    if isinstance(hits, dict) and isinstance(hits.get("hits"), list):
        hits["hits"] = hits["hits"][:bounded_size]
    return result


def _cached_alpaca_response(cache_key: str, size: int) -> dict[str, Any] | None:
    with _ALPACA_SEARCH_CACHE_LOCK:
        cached = _ALPACA_SEARCH_CACHE.get(cache_key)
    if not cached:
        return None
    cached_size, response = cached
    if cached_size < bounded_candidate_count(size):
        return None
    return _slice_alpaca_response(response, size)


def _store_alpaca_response(cache_key: str, size: int, response: dict[str, Any]) -> None:
    if ALPACA_SEARCH_CACHE_MAX_ENTRIES <= 0:
        return
    bounded_size = bounded_candidate_count(size)
    with _ALPACA_SEARCH_CACHE_LOCK:
        existing = _ALPACA_SEARCH_CACHE.get(cache_key)
        if existing and existing[0] >= bounded_size:
            return
        if len(_ALPACA_SEARCH_CACHE) >= ALPACA_SEARCH_CACHE_MAX_ENTRIES:
            oldest_key = next(iter(_ALPACA_SEARCH_CACHE))
            _ALPACA_SEARCH_CACHE.pop(oldest_key, None)
        _ALPACA_SEARCH_CACHE[cache_key] = (bounded_size, copy.deepcopy(response))


def alpaca_search(query_body: dict[str, Any], size: int) -> dict[str, Any]:
    token = alpaca_token()
    if not token:
        raise RuntimeError("ALPACA_TOKEN is not configured")

    cache_key = _alpaca_cache_key(query_body)
    cached_response = _cached_alpaca_response(cache_key, size)
    if cached_response is not None:
        return cached_response

    payload = dict(query_body)
    payload["size"] = bounded_candidate_count(size)
    request = Request(
        ALPACA_METADATA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "coverage-dashboard/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            response_payload = json.loads(response.read() or b"{}")
            _store_alpaca_response(cache_key, payload["size"], response_payload)
            return response_payload
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Alpaca HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Alpaca request failed: {exc}") from exc


def extract_hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    hits = response.get("hits", {}).get("hits", [])
    candidates: list[dict[str, Any]] = []
    if not isinstance(hits, list):
        return candidates

    for index, hit in enumerate(hits, start=1):
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source") if isinstance(hit.get("_source"), dict) else hit
        candidate = dict(source)
        candidate["rank"] = index
        candidate["es_rank"] = index
        candidate["es_score"] = hit.get("_score", candidate.get("es_score"))
        candidates.append(candidate)
    return candidates
