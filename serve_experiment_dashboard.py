#!/usr/bin/env python3
"""Serve the experiment dashboard and proxy GT metadata requests."""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.app.env import load_dotenv


load_dotenv()

ALPACA_METADATA_URL = os.environ.get(
    "ALPACA_METADATA_URL",
    "https://alpaca.zooverse.dev/debug/elasticsearch/alpaca-entities/_search",
)
OPENROUTER_CHAT_URL = os.environ.get("OPENROUTER_CHAT_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b")
OPENROUTER_PROVIDER = os.environ.get("OPENROUTER_PROVIDER", "Cerebras")
OPENROUTER_MAX_TOKENS = os.environ.get("OPENROUTER_MAX_TOKENS")
NER_TYPE_PAIRS = (
    ("PERSON", "PERSON"),
    ("PERSON", "FICTIONAL_CHARACTER"),
    ("ORGANIZATION", "COMPANY"),
    ("ORGANIZATION", "NONPROFIT_ORG"),
    ("ORGANIZATION", "GOVERNMENT_ORG"),
    ("ORGANIZATION", "EDUCATIONAL_ORG"),
    ("ORGANIZATION", "SPORTS_TEAM"),
    ("LOCATION", "COUNTRY"),
    ("LOCATION", "CITY"),
    ("LOCATION", "REGION"),
    ("LOCATION", "LANDMARK"),
    ("LOCATION", "CELESTIAL_BODY"),
    ("EVENT", "CONFLICT"),
    ("EVENT", "SPORT_EVENT"),
    ("EVENT", "EVENT_GENERIC"),
    ("WORK", "FILM"),
    ("WORK", "BOOK"),
    ("WORK", "MUSIC_WORK"),
    ("WORK", "SOFTWARE"),
    ("WORK", "INTERNET_MEME"),
    ("PRODUCT", "DEVICE"),
    ("PRODUCT", "MEDICATION"),
    ("PRODUCT", "FOOD_BEVERAGE"),
    ("PRODUCT", "PRODUCT_GENERIC"),
    ("CONCEPT", "LANGUAGE"),
    ("CONCEPT", "LAW"),
    ("CONCEPT", "SCIENTIFIC_THEORY"),
    ("CONCEPT", "BIOLOGICAL_TAXON"),
    ("CONCEPT", "ANATOMY"),
    ("RELATION", "PROPERTY"),
    ("MISC", "MISC"),
)
ALLOWED_COARSE_TYPES = {coarse for coarse, _ in NER_TYPE_PAIRS}
ALLOWED_FINE_TYPES = {fine for _, fine in NER_TYPE_PAIRS}
TEXT_SEARCH_FIELDS = {"label", "labels", "aliases", "context_string", "description"}
URL_FILTER_FIELDS = {"wikipedia_url", "dbpedia_url"}
EXACT_FILTER_FIELDS = {"coarse_type", "fine_type", "item_category", "types"}
SOFT_TYPE_FIELDS = {"coarse_type", "fine_type", "types"}
ALLOWED_ITEM_CATEGORIES = ["ENTITY", "TYPE"]
NULL_URL_HINT_VALUES = {"", "n/a", "na", "none", "null", "unknown"}


class DashboardHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_POST(self) -> None:
        route = self.path.rstrip("/")
        if route == "/api/gt-metadata":
            self._handle_gt_metadata()
            return
        if route == "/api/live-attempt":
            self._handle_live_attempt()
            return
        self.send_error(404, "Unknown API route")

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b"{}"
        payload = json.loads(body or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _handle_gt_metadata(self) -> None:
        try:
            body = self._read_json_body()
            qids = _clean_qids(body.get("qids") or body.get("qid") or [])
            if not qids:
                self._send_json({"hits": {"hits": []}})
                return

            response = _alpaca_search({"query": {"ids": {"values": qids}}}, len(qids))
            self._send_json(response)
        except HTTPError as error:
            self._send_json({"error": error.reason, "status": error.code}, status=error.code)
        except (json.JSONDecodeError, ValueError, URLError, TimeoutError, OSError) as error:
            self._send_json({"error": str(error)}, status=502)

    def _handle_live_attempt(self) -> None:
        try:
            body = self._read_json_body()
            row = body.get("row") if isinstance(body.get("row"), dict) else {}
            gold_qids = set(_clean_qids(body.get("gold_qids") or row.get("gold_qids") or []))
            candidate_count = _bounded_int(body.get("candidate_count"), default=50, minimum=1, maximum=300)
            attempt_count = _bounded_int(body.get("attempt_count"), default=2, minimum=1, maximum=3)
            human_guidance = str(body.get("human_guidance") or "").strip()[:2000]
            if not row:
                self._send_json({"error": "Missing row payload"}, status=400)
                return

            llm_plan = _generate_live_attempt_plan(row, human_guidance, attempt_count)
            plan_metadata = llm_plan.get("metadata", {})
            attempts = []
            for index, plan in enumerate(llm_plan.get("attempts", [])[:attempt_count], start=1):
                query_body = _sanitize_query_body(plan.get("body") or plan.get("query_body") or plan)
                try:
                    response = _alpaca_search(query_body, candidate_count)
                    candidates = _extract_hits(response)
                    candidate_qids = {str(candidate.get("qid") or "") for candidate in candidates}
                    covered_qids = sorted(gold_qids & candidate_qids)
                    attempts.append(
                        {
                            "attempt": index,
                            "title": plan.get("title") or f"Attempt {index}",
                            "rationale": plan.get("rationale") or "",
                            "query_body": {**query_body, "size": candidate_count},
                            "covered": bool(covered_qids),
                            "covered_qids": covered_qids,
                            "candidate_count": len(candidates),
                            "candidates": candidates[:candidate_count],
                        }
                    )
                except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                    attempts.append(
                        {
                            "attempt": index,
                            "title": plan.get("title") or f"Attempt {index}",
                            "rationale": plan.get("rationale") or "",
                            "query_body": {**query_body, "size": candidate_count},
                            "covered": False,
                            "covered_qids": [],
                            "candidate_count": 0,
                            "candidates": [],
                            "error": str(error),
                        }
                    )

            self._send_json(
                {
                    "model": OPENROUTER_MODEL,
                    "provider": OPENROUTER_PROVIDER,
                    "candidate_count": candidate_count,
                    "attempt_count": len(attempts),
                    "requested_attempt_count": attempt_count,
                    "human_guidance": human_guidance,
                    "model_understanding": plan_metadata.get("model_understanding") or "",
                    "query_strategy": plan_metadata.get("query_strategy") or "",
                    "gold_qids": sorted(gold_qids),
                    "covered": any(attempt.get("covered") for attempt in attempts),
                    "covered_qids": sorted({qid for attempt in attempts for qid in attempt.get("covered_qids", [])}),
                    "attempts": attempts,
                }
            )
        except HTTPError as error:
            self._send_json({"error": error.reason, "status": error.code}, status=error.code)
        except (json.JSONDecodeError, ValueError, URLError, TimeoutError, OSError) as error:
            self._send_json({"error": str(error)}, status=502)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        response_body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def _clean_qids(values: object) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list | tuple | set):
        return []
    seen: set[str] = set()
    qids: list[str] = []
    for value in values:
        qid = str(value or "").strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)
        qids.append(qid)
    return qids


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _post_json(url: str, payload: dict, headers: dict[str, str], *, timeout: int = 60) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.7.1",
            **headers,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {error.code}: {detail}") from error


def _alpaca_token() -> str:
    return (
        os.environ.get("ALPACA_TOKEN")
        or os.environ.get("ALPACA_AUTH_TOKEN")
        or os.environ.get("ALPACA_TOKEN_AUTH")
        or ""
    ).strip()


def _alpaca_search(query_body: dict, size: int) -> dict:
    token = _alpaca_token()
    if not token:
        raise ValueError("ALPACA_TOKEN is not configured")
    payload = dict(query_body)
    payload["size"] = size
    return _post_json(
        ALPACA_METADATA_URL,
        payload,
        {"Authorization": f"Bearer {token}"},
        timeout=45,
    )


def _extract_hits(response: dict) -> list[dict]:
    hits = response.get("hits", {}).get("hits", [])
    candidates: list[dict] = []
    for index, hit in enumerate(hits, start=1):
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source") if isinstance(hit.get("_source"), dict) else hit
        candidate = dict(source)
        candidate["es_score"] = hit.get("_score", candidate.get("es_score"))
        candidate["es_rank"] = index
        candidates.append(candidate)
    return candidates


def _truncate(value: object, limit: int = 1200) -> object:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return [_truncate(item, limit) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _truncate(item, limit) for key, item in list(value.items())[:30]}
    return value


def _compact_row_for_llm(row: dict) -> dict:
    profile = row.get("__table_profile") if isinstance(row.get("__table_profile"), dict) else {}
    preprocessing = row.get("preprocessing") if isinstance(row.get("preprocessing"), dict) else {}
    return {
        "mention": row.get("mention"),
        "dataset_id": row.get("dataset_id"),
        "table_id": row.get("table_id"),
        "row_id": row.get("row_id"),
        "col_id": row.get("col_id"),
        "source_record": row.get("__source_record") or row.get("source_record"),
        "table_semantic_family": profile.get("table_semantic_family"),
        "row_template": profile.get("row_template"),
        "column_roles": profile.get("column_roles"),
        "table_hypotheses": profile.get("table_hypotheses"),
        "semantic_hint_vocabulary": profile.get("semantic_hint_vocabulary"),
        "preprocessing": {
            "canonical_mention": preprocessing.get("canonical_mention"),
            "mention_variants": preprocessing.get("mention_variants"),
            "query_variants": preprocessing.get("query_variants"),
            "entity_hypotheses": preprocessing.get("entity_hypotheses"),
            "hard_filters": preprocessing.get("hard_filters"),
            "soft_context_terms": preprocessing.get("soft_context_terms"),
            "wikipedia_title_hints": preprocessing.get("wikipedia_title_hints"),
            "url_hints": preprocessing.get("url_hints"),
            "expected_disambiguators": preprocessing.get("expected_disambiguators"),
            "forbidden_disambiguators": preprocessing.get("forbidden_disambiguators"),
            "weakness_reasons": preprocessing.get("weakness_reasons"),
        },
    }


def _ner_type_summary() -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {}
    for coarse, fine in NER_TYPE_PAIRS:
        summary.setdefault(coarse, [])
        if fine not in summary[coarse]:
            summary[coarse].append(fine)
    return summary


def _generate_live_attempt_plan(row: dict, human_guidance: str, attempt_count: int) -> dict:
    row_context = _truncate(_compact_row_for_llm(row))
    messages = [
        {
            "role": "system",
            "content": (
                "You generate Elasticsearch DSL query bodies for the alpaca-entities index. "
                "Return compact minified JSON only. Do not include markdown or prose. Do not use QID filters or any ground-truth IDs. "
                "Never decide the result size; the server injects size externally. "
                "Create the requested number of diverse query attempts. Use only ordinary search clauses such as bool, multi_match, match, match_phrase, term, terms, and should clauses. "
                "Keep DSL conservative: avoid cross_fields, fuzziness, function_score, script_score, rescore, aggregations, nested queries, and custom analyzers. "
                "NER/type clues are soft boosts only. Do not put coarse_type, fine_type, or types in filter or must clauses. Put type clues in should clauses only. "
                "The server filters all queries to item_category ENTITY or TYPE. "
                "Only use fine_type values listed in allowed_ner_types. "
                "Searchable text fields: label, labels, aliases, context_string, description. "
                "Exact filter fields: coarse_type, fine_type, item_category, types. "
                "The wikipedia_url and dbpedia_url fields are returned as final page/resource slugs, not full URLs, but are not directly queryable in this index. "
                "When using URL hints, extract only the final slug, e.g. Paris from https://en.wikipedia.org/wiki/Paris or https://dbpedia.org/resource/Paris, and search that slug text in label, labels, aliases, context_string, or description. "
                "Entity docs have fields qid, label, labels, aliases, types, context_string, coarse_type, fine_type, item_category, popularity, prior, wikipedia_url, dbpedia_url, description. "
                "Prefer queries that could recover the correct KG item from the mention and table context, including disambiguating row context and NER type filters when helpful."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Create Elasticsearch query attempts to retrieve the correct entity for this table cell.",
                    "attempt_count": attempt_count,
                    "human_guidance": human_guidance,
                    "output_schema": {
                        "model_understanding": "plain explanation of what the table cell likely denotes and which context matters",
                        "query_strategy": "short explanation of the retrieval strategy and risk tradeoffs",
                        "attempts": [
                            {
                                "title": "short name",
                                "rationale": "why this query should help",
                                "body": {"query": "Elasticsearch query object; no size/from/aggs"},
                            }
                        ]
                    },
                    "allowed_ner_types": _ner_type_summary(),
                    "row_without_gt_qids": row_context,
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = _openrouter_chat(messages)
    parsed = _extract_json_object(response)
    attempts = parsed.get("attempts") if isinstance(parsed, dict) else None
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("OpenRouter response did not contain attempts[]")
    valid_attempts = [attempt for attempt in attempts if isinstance(attempt, dict)]
    if not valid_attempts:
        raise ValueError("OpenRouter attempts[] did not contain JSON objects")
    return {
        "metadata": {
            "model_understanding": parsed.get("model_understanding") if isinstance(parsed.get("model_understanding"), str) else "",
            "query_strategy": parsed.get("query_strategy") if isinstance(parsed.get("query_strategy"), str) else "",
        },
        "attempts": valid_attempts,
    }


def _openrouter_chat(messages: list[dict]) -> str:
    token = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not token:
        raise ValueError("OPENROUTER_API_KEY is not configured")
    payload = {
        "model": OPENROUTER_MODEL,
        "provider": {"order": [OPENROUTER_PROVIDER], "allow_fallbacks": False},
        "messages": messages,
        "temperature": 0.35,
        "include_reasoning": False,
        "response_format": {"type": "json_object"},
    }
    if OPENROUTER_MAX_TOKENS:
        payload["max_tokens"] = _bounded_int(OPENROUTER_MAX_TOKENS, default=1200, minimum=256, maximum=8192)
    response = _post_json(
        OPENROUTER_CHAT_URL,
        payload,
        {
            "Authorization": f"Bearer {token}",
            "HTTP-Referer": "http://127.0.0.1:8765/experiment_results_dashboard.html",
            "X-Title": "Experiment Results Dashboard",
        },
        timeout=90,
    )
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise ValueError(f"OpenRouter returned no choices: {response}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter returned an empty message")
    return content


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _sanitize_query_body(body: object) -> dict:
    if not isinstance(body, dict):
        raise ValueError("LLM attempt body must be a JSON object")
    if "query" not in body:
        body = {"query": body}
    cleaned = {
        key: value
        for key, value in body.items()
        if key not in {"size", "from", "aggs", "aggregations", "scroll", "pit", "search_after"}
    }
    if "query" not in cleaned or not isinstance(cleaned["query"], dict):
        raise ValueError("LLM attempt must contain query object")
    cleaned["query"] = _with_allowed_item_category_filter(_sanitize_query_clause(cleaned["query"]) or {"match_all": {}})
    return cleaned


def _sanitize_query_clause(value: object) -> object:
    if isinstance(value, list):
        return [item for item in (_sanitize_query_clause(item) for item in value) if item is not None]
    if not isinstance(value, dict):
        return value

    if "bool" in value and isinstance(value["bool"], dict):
        bool_query = value["bool"]
        sanitized_bool: dict = {}
        soft_should: list[object] = []

        for key in ("must", "filter", "should", "must_not"):
            raw_clauses = bool_query.get(key)
            if raw_clauses is None:
                continue
            clause_list = raw_clauses if isinstance(raw_clauses, list) else [raw_clauses]
            cleaned_clauses: list[object] = []
            for clause in clause_list:
                cleaned = _sanitize_query_clause(clause)
                if cleaned in (None, [], {}):
                    continue
                if key in {"must", "filter"} and _is_soft_type_clause(cleaned):
                    soft_should.append(cleaned)
                else:
                    cleaned_clauses.append(cleaned)
            if cleaned_clauses:
                sanitized_bool[key] = cleaned_clauses if isinstance(raw_clauses, list) else cleaned_clauses[0]

        existing_should = sanitized_bool.get("should")
        if existing_should is None:
            sanitized_bool["should"] = soft_should
        elif isinstance(existing_should, list):
            sanitized_bool["should"] = existing_should + soft_should
        else:
            sanitized_bool["should"] = [existing_should, *soft_should]

        if not sanitized_bool.get("should"):
            sanitized_bool.pop("should", None)
        for key in ("minimum_should_match", "boost"):
            if key in bool_query and key not in sanitized_bool:
                sanitized_bool[key] = bool_query[key]
        if sanitized_bool.get("must") or sanitized_bool.get("filter"):
            sanitized_bool.pop("minimum_should_match", None)
        return {"bool": sanitized_bool}

    if "multi_match" in value and isinstance(value["multi_match"], dict):
        fields = []
        for field in value["multi_match"].get("fields", []):
            base = str(field).split("^", 1)[0]
            if base in TEXT_SEARCH_FIELDS:
                fields.append(field)
        if not fields:
            fields = ["label^4", "labels^2", "aliases^2", "context_string", "description"]
        multi_match = {
            key: item
            for key, item in value["multi_match"].items()
            if key not in {"fuzziness", "analyzer", "minimum_should_match"}
        }
        multi_match["fields"] = fields
        if multi_match.get("type") not in {None, "best_fields", "most_fields", "phrase", "phrase_prefix", "bool_prefix"}:
            multi_match.pop("type", None)
        return {"multi_match": multi_match}

    if "term" in value and isinstance(value["term"], dict):
        field, raw = next(iter(value["term"].items()))
        if field in URL_FILTER_FIELDS:
            return _url_slug_text_clause(raw)
        if field not in EXACT_FILTER_FIELDS:
            return None
        item = raw.get("value") if isinstance(raw, dict) and "value" in raw else raw
        if field == "fine_type" and item not in ALLOWED_FINE_TYPES:
            return None
        if field == "coarse_type" and item not in ALLOWED_COARSE_TYPES:
            return None
        return {"term": {field: raw}}

    if "terms" in value and isinstance(value["terms"], dict):
        field, raw_values = next(iter(value["terms"].items()))
        if field in URL_FILTER_FIELDS:
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            clauses = [clause for clause in (_url_slug_text_clause(item) for item in values) if clause]
            if not clauses:
                return None
            return {"bool": {"should": clauses, "minimum_should_match": 1}}
        if field not in EXACT_FILTER_FIELDS:
            return None
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        if field == "fine_type":
            values = [item for item in values if item in ALLOWED_FINE_TYPES]
        if field == "coarse_type":
            values = [item for item in values if item in ALLOWED_COARSE_TYPES]
        if not values:
            return None
        return {"terms": {field: values}}

    for text_query in ("match", "match_phrase"):
        if text_query in value and isinstance(value[text_query], dict):
            field, raw = next(iter(value[text_query].items()))
            if field in URL_FILTER_FIELDS:
                return _url_slug_text_clause(raw)
            if field not in TEXT_SEARCH_FIELDS:
                return None
            return {text_query: {field: raw}}

    sanitized: dict = {}
    for key, item in value.items():
        cleaned = _sanitize_query_clause(item)
        if cleaned in (None, [], {}):
            continue
        sanitized[key] = cleaned
    return sanitized


def _normalize_url_slug(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.split("#", 1)[0].split("?", 1)[0].strip().strip("/")
    for marker in ("/wiki/", "/resource/", "/page/"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
            break
    else:
        text = text.rsplit("/", 1)[-1]
    text = text.strip().replace(" ", "_")
    if not text or text.casefold() in NULL_URL_HINT_VALUES:
        return None
    return text


def _with_allowed_item_category_filter(query: object) -> dict:
    return {
        "bool": {
            "filter": [{"terms": {"item_category": ALLOWED_ITEM_CATEGORIES}}],
            "must": [query if isinstance(query, dict) and query else {"match_all": {}}],
        }
    }


def _url_slug_text_clause(value: object) -> dict | None:
    raw = value.get("value", value.get("query")) if isinstance(value, dict) else value
    slug = _normalize_url_slug(raw)
    if not slug:
        return None
    return {
        "multi_match": {
            "query": slug.replace("_", " "),
            "fields": ["label^8", "labels^6", "aliases^6", "description^1.2", "context_string^0.4"],
            "type": "best_fields",
            "operator": "and",
        }
    }


def _is_soft_type_clause(clause: object) -> bool:
    if not isinstance(clause, dict):
        return False
    for query_type in ("term", "terms"):
        payload = clause.get(query_type)
        if isinstance(payload, dict) and payload:
            field = next(iter(payload.keys()))
            return field in SOFT_TYPE_FIELDS
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve experiment_results_dashboard.html")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving dashboard at http://{args.host}:{args.port}/experiment_results_dashboard.html")
    print("GT metadata proxy available at /api/gt-metadata")
    print("Live attempt proxy available at /api/live-attempt")
    server.serve_forever()


if __name__ == "__main__":
    main()
