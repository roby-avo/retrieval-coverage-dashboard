from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from psycopg.types.json import Jsonb

from .db import connect
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
ALPACA_SEARCH_CACHE_ENABLED = os.environ.get("ALPACA_SEARCH_CACHE_ENABLED", "true").strip().casefold() not in {
    "0",
    "false",
    "no",
    "off",
}

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


def normalize_cache_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def normalize_query_plan_source(value: Any) -> str:
    normalized = normalize_cache_text(value)
    if normalized == "heuristic" or "heuristic" in normalized or normalized in {"guidance_not_applied_without_llm"}:
        return "heuristic"
    if normalized in {"", "none", "off", "disabled", "false", "legacy"}:
        return "legacy"
    return "llm"


def candidate_retrieval_cache_key(mention_text: Any, query_text: Any, query_plan_source: Any = "legacy") -> str | None:
    normalized_mention = normalize_cache_text(mention_text)
    normalized_query = normalize_cache_text(query_text)
    normalized_source = normalize_query_plan_source(query_plan_source)
    if not normalized_mention or not normalized_query:
        return None
    canonical = json.dumps(
        {"mention": normalized_mention, "query": normalized_query, "query_plan_source": normalized_source},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    aliases = _clean_string_list(signals.get("aliases"), limit=10)
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


def _canonical_alpaca_query_payload(query_body: dict[str, Any]) -> dict[str, Any]:
    payload = dict(query_body)
    payload.pop("size", None)
    payload.pop("_inspection", None)
    return payload


def alpaca_query_fingerprint(query_body: dict[str, Any]) -> str:
    canonical = json.dumps(
        _canonical_alpaca_query_payload(query_body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _alpaca_query_fingerprint(query_body: dict[str, Any]) -> str:
    return alpaca_query_fingerprint(query_body)


def _slice_alpaca_response(response: dict[str, Any], size: int) -> dict[str, Any]:
    bounded_size = bounded_candidate_count(size)
    result = copy.deepcopy(response)
    hits = result.get("hits")
    if isinstance(hits, dict) and isinstance(hits.get("hits"), list):
        hits["hits"] = hits["hits"][:bounded_size]
    return result


def _cached_candidate_retrieval_response(mention_text: Any, query_text: Any, query_plan_source: Any, size: int) -> dict[str, Any] | None:
    cache_key = candidate_retrieval_cache_key(mention_text, query_text, query_plan_source)
    if not cache_key:
        return None
    requested_size = bounded_candidate_count(size)
    with connect() as conn:
        row = conn.execute(
            """
            SELECT response_payload, cached_candidate_count
            FROM candidate_retrieval_cache
            WHERE cache_key = %s
              AND query_plan_source = %s
              AND cached_candidate_count >= %s
            """,
            (cache_key, normalize_query_plan_source(query_plan_source), requested_size),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            """
            UPDATE candidate_retrieval_cache
            SET hit_count = hit_count + 1,
                last_hit_at = now()
            WHERE cache_key = %s
            """,
            (cache_key,),
        )
        conn.commit()

    response = row.get("response_payload")
    if not isinstance(response, dict):
        return None
    return _slice_alpaca_response(response, requested_size)


def _store_candidate_retrieval_response(
    *,
    mention_text: Any,
    query_text: Any,
    query_plan_source: Any,
    query_body: dict[str, Any],
    size: int,
    response: dict[str, Any],
) -> None:
    cache_key = candidate_retrieval_cache_key(mention_text, query_text, query_plan_source)
    if not cache_key:
        return
    requested_size = bounded_candidate_count(size)
    normalized_mention = normalize_cache_text(mention_text)
    normalized_query = normalize_cache_text(query_text)
    normalized_source = normalize_query_plan_source(query_plan_source)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO candidate_retrieval_cache (
                cache_key, mention_text, normalized_mention, query_text, normalized_query,
                query_plan_source, query_payload, response_payload, cached_candidate_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cache_key) DO UPDATE
            SET response_payload = CASE
                    WHEN candidate_retrieval_cache.cached_candidate_count < EXCLUDED.cached_candidate_count
                    THEN EXCLUDED.response_payload
                    ELSE candidate_retrieval_cache.response_payload
                END,
                cached_candidate_count = greatest(
                    candidate_retrieval_cache.cached_candidate_count,
                    EXCLUDED.cached_candidate_count
                ),
                query_payload = CASE
                    WHEN candidate_retrieval_cache.cached_candidate_count < EXCLUDED.cached_candidate_count
                    THEN EXCLUDED.query_payload
                    ELSE candidate_retrieval_cache.query_payload
                END,
                mention_text = EXCLUDED.mention_text,
                query_text = EXCLUDED.query_text,
                query_plan_source = EXCLUDED.query_plan_source,
                last_cached_at = now()
            """,
            (
                cache_key,
                str(mention_text or "").strip(),
                normalized_mention,
                str(query_text or "").strip(),
                normalized_query,
                normalized_source,
                Jsonb(_canonical_alpaca_query_payload(query_body)),
                Jsonb(response),
                requested_size,
            ),
        )
        conn.commit()


def alpaca_search(
    query_body: dict[str, Any],
    size: int,
    *,
    mention_text: Any | None = None,
    query_text: Any | None = None,
    query_plan_source: Any | None = None,
) -> dict[str, Any]:
    use_candidate_cache = mention_text is not None and query_text is not None and query_plan_source is not None
    if use_candidate_cache:
        cached_candidate_response = _cached_candidate_retrieval_response(mention_text, query_text, query_plan_source, size)
        if cached_candidate_response is not None:
            return cached_candidate_response

    token = alpaca_token()
    if not token:
        raise RuntimeError("ALPACA_TOKEN is not configured")

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
            if mention_text is not None and query_text is not None:
                _store_candidate_retrieval_response(
                    mention_text=mention_text,
                    query_text=query_text,
                    query_plan_source=query_plan_source,
                    query_body=query_body,
                    size=payload["size"],
                    response=response_payload,
                )
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
        candidate.setdefault("qid", hit.get("_id"))
        candidate["rank"] = index
        candidate["es_rank"] = index
        candidate["es_score"] = hit.get("_score", candidate.get("es_score"))
        candidates.append(candidate)
    return candidates
