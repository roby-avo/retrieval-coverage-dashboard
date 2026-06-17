from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .retrieval import MAX_RETURNED_CANDIDATES, alpaca_query_fingerprint


@dataclass
class ImportStats:
    run_id: int
    name: str
    table_count: int
    mention_count: int
    candidate_count: int
    covered_count: int


@dataclass
class RowImportStats:
    mention_count: int
    candidate_count: int
    covered_count: int


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_qids(values: Any) -> list[str]:
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    if not isinstance(values, list | tuple | set):
        return []

    qids: list[str] = []
    seen: set[str] = set()
    for value in values:
        qid = _as_text(value)
        if not qid:
            continue
        if "/" in qid:
            qid = qid.rsplit("/", 1)[-1]
        if qid in seen:
            continue
        seen.add(qid)
        qids.append(qid)
    return qids


def _candidate_stages(candidate: dict[str, Any]) -> list[str]:
    raw = candidate.get("retrieval_stages")
    if isinstance(raw, str):
        raw = [raw]
    stages: list[str] = []
    if isinstance(raw, list | tuple):
        stages.extend(str(item) for item in raw if item)
    if candidate.get("retrieval_stage"):
        stages.insert(0, str(candidate["retrieval_stage"]))

    result: list[str] = []
    seen: set[str] = set()
    for stage in stages:
        if stage in seen:
            continue
        seen.add(stage)
        result.append(stage)
    return result


def _entity_by_qid(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities = row.get("gt_entities")
    if not isinstance(entities, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        qid = _as_text(entity.get("qid"))
        if qid:
            result[qid] = entity
    return result


def _mention_metadata(row: dict[str, Any]) -> dict[str, Any]:
    duplicated_ranked_payloads = {"candidates", "reranked"}
    return {key: value for key, value in row.items() if key not in duplicated_ranked_payloads}


def _limited_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return []

    limited: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            limited.append(candidate)
        if len(limited) >= MAX_RETURNED_CANDIDATES:
            break
    return limited


def _alpaca_request_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    backend_requests = row.get("backend_requests")
    if not isinstance(backend_requests, list) or not backend_requests:
        return None
    first_request = backend_requests[0]
    if not isinstance(first_request, dict):
        return None
    request_payload = first_request.get("request")
    return request_payload if isinstance(request_payload, dict) else None


def _retrieval_fingerprint(row: dict[str, Any]) -> str | None:
    request_payload = _alpaca_request_payload(row)
    if not request_payload:
        return None
    return alpaca_query_fingerprint(request_payload)


def create_import_run(
    conn: psycopg.Connection[Any],
    *,
    name: str,
    source_path: str | None = None,
    source_filename: str | None = None,
    raw_summary: dict[str, Any] | None = None,
    raw_sampling_config: dict[str, Any] | None = None,
    replace_existing: bool = False,
) -> int:
    if replace_existing and source_path:
        conn.execute("DELETE FROM runs WHERE source_path = %s", (source_path,))

    run = conn.execute(
        """
        INSERT INTO runs (name, source_path, source_filename, raw_summary, raw_sampling_config)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            name,
            source_path,
            source_filename,
            Jsonb(raw_summary or {}),
            Jsonb(raw_sampling_config or {}),
        ),
    ).fetchone()
    if not run:
        raise RuntimeError("Could not create run")
    return int(run["id"])


def upsert_experiment_table(
    conn: psycopg.Connection[Any],
    *,
    run_id: int,
    dataset_id: str | None,
    table_id: str | None,
    sample_limit: int | None = None,
    raw_summary: dict[str, Any] | None = None,
    raw_profile: dict[str, Any] | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> int:
    table_row = conn.execute(
        """
        INSERT INTO experiment_tables
            (run_id, dataset_id, table_id, sample_limit, raw_summary, raw_profile, raw_payload)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id, dataset_id, table_id)
        DO UPDATE SET
            sample_limit = EXCLUDED.sample_limit,
            raw_summary = EXCLUDED.raw_summary,
            raw_profile = EXCLUDED.raw_profile,
            raw_payload = EXCLUDED.raw_payload
        RETURNING id
        """,
        (
            run_id,
            dataset_id,
            table_id,
            sample_limit,
            Jsonb(raw_summary or {}),
            Jsonb(raw_profile or {}),
            Jsonb(raw_payload or {}),
        ),
    ).fetchone()
    if not table_row:
        raise RuntimeError("Could not create experiment table")
    return int(table_row["id"])


def import_experiment_row(
    conn: psycopg.Connection[Any],
    *,
    run_id: int,
    table_db_id: int | None,
    row: dict[str, Any],
) -> RowImportStats:
    qids = _clean_qids(row.get("gold_qids") or row.get("gt_qids") or [])
    candidates = _limited_candidates(row)
    retrieval_fingerprint = _retrieval_fingerprint(row)
    mention = conn.execute(
        """
        INSERT INTO mentions (
            run_id, table_db_id, cell_key, dataset_id, table_id, row_id, col_id,
            mention, mention_text, lookup_text, primary_gt_qid, selected_qid,
            selected_label, final_correct, coverage_correct, hit_at_1, hit_at_5,
            hit_at_10, hit_at_k, best_gt_rank, retrieved_count, candidate_count,
            candidate_backend, query_engine, raw_payload, retrieval_fingerprint,
            retrieval_cache_candidate_count
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (run_id, cell_key) DO NOTHING
        RETURNING id
        """,
        (
            run_id,
            table_db_id,
            _as_text(row.get("cell_key")),
            _as_text(row.get("dataset_id")),
            _as_text(row.get("table_id")),
            _as_int(row.get("row_id")),
            _as_int(row.get("col_id")),
            _as_text(row.get("mention")),
            _as_text(row.get("mention_text")),
            _as_text(row.get("lookup_text")),
            _as_text(row.get("primary_gt_qid")) or (qids[0] if qids else None),
            _as_text(row.get("selected_qid")),
            _as_text(row.get("selected_label")),
            _as_bool(row.get("final_correct")),
            _as_bool(row.get("coverage_correct")),
            _as_bool(row.get("hit_at_1")),
            _as_bool(row.get("hit_at_5")),
            _as_bool(row.get("hit_at_10")),
            _as_bool(row.get("hit_at_k")),
            _as_int(row.get("best_gt_rank") or row.get("gold_rank")),
            _as_int(row.get("retrieved_count")),
            _as_int(row.get("candidate_count")),
            _as_text(row.get("candidate_backend")),
            _as_text(row.get("query_engine")),
            Jsonb(_mention_metadata(row)),
            retrieval_fingerprint,
            len(candidates),
        ),
    ).fetchone()
    if not mention:
        return RowImportStats(mention_count=0, candidate_count=0, covered_count=0)

    mention_id = int(mention["id"])
    raw_entities = _entity_by_qid(row)
    primary_qid = _as_text(row.get("primary_gt_qid")) or (qids[0] if qids else None)
    gold_rows = [
        (
            mention_id,
            qid,
            index,
            qid == primary_qid,
            Jsonb(raw_entities.get(qid) or {}),
        )
        for index, qid in enumerate(qids, start=1)
    ]
    if gold_rows:
        conn.cursor().executemany(
            """
            INSERT INTO gold_qids (mention_id, qid, ordinal, is_primary, raw_entity)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (mention_id, qid) DO NOTHING
            """,
            gold_rows,
        )

    candidate_count = 0
    if candidates:
        candidate_rows = []
        qid_set = set(qids)
        for rank, candidate in enumerate(candidates, start=1):
            candidate_qid = _as_text(candidate.get("qid"))
            stages = _candidate_stages(candidate)
            candidate_rows.append(
                (
                    mention_id,
                    rank,
                    _as_int(candidate.get("rank") or candidate.get("merged_rank") or candidate.get("es_rank")),
                    candidate_qid,
                    _as_text(candidate.get("label")),
                    _as_text(candidate.get("item_category")),
                    _as_text(candidate.get("coarse_type")),
                    _as_text(candidate.get("fine_type")),
                    _as_text(candidate.get("retrieval_system"))
                    or _as_text(row.get("candidate_backend"))
                    or _as_text(row.get("query_engine")),
                    stages[0] if stages else _as_text(candidate.get("retrieval_stage")),
                    stages,
                    _as_float(candidate.get("score") or candidate.get("final_score")),
                    _as_float(candidate.get("es_score")),
                    _as_float(candidate.get("heuristic_score")),
                    bool(candidate.get("isSelected")),
                    bool(candidate.get("isGold")) or bool(candidate_qid and candidate_qid in qid_set),
                    Jsonb(candidate),
                )
            )

        if candidate_rows:
            conn.cursor().executemany(
                """
                INSERT INTO candidates (
                    mention_id, rank, source_rank, qid, label, item_category, coarse_type,
                    fine_type, retrieval_system, retrieval_stage, retrieval_stages, score,
                    es_score, heuristic_score, selected, gold_match, raw_payload
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                candidate_rows,
            )
            candidate_count = len(candidate_rows)

    covered = bool(row.get("hit_at_k") or row.get("coverage_correct"))
    return RowImportStats(mention_count=1, candidate_count=candidate_count, covered_count=1 if covered else 0)


def update_run_counters(
    conn: psycopg.Connection[Any],
    *,
    run_id: int,
    table_count: int | None = None,
    mention_delta: int = 0,
    candidate_delta: int = 0,
    covered_delta: int = 0,
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET table_count = coalesce(%s, table_count),
            mention_count = mention_count + %s,
            candidate_count = candidate_count + %s,
            covered_count = covered_count + %s
        WHERE id = %s
        """,
        (table_count, mention_delta, candidate_delta, covered_delta, run_id),
    )


def finalize_import_run(
    conn: psycopg.Connection[Any],
    *,
    run_id: int,
    raw_summary: dict[str, Any] | None = None,
) -> ImportStats:
    counts = conn.execute(
        """
        SELECT
            (SELECT count(*) FROM experiment_tables WHERE run_id = %s) AS table_count,
            (SELECT count(*) FROM mentions WHERE run_id = %s) AS mention_count,
            (SELECT count(*) FROM candidates c JOIN mentions m ON m.id = c.mention_id WHERE m.run_id = %s) AS candidate_count,
            (
                SELECT count(*)
                FROM mentions m
                WHERE m.run_id = %s
                  AND (
                    coalesce(m.coverage_correct, false)
                    OR m.best_gt_rank IS NOT NULL
                    OR EXISTS (
                      SELECT 1
                      FROM candidates c
                      JOIN gold_qids g ON g.mention_id = m.id AND g.qid = c.qid
                      WHERE c.mention_id = m.id
                    )
                  )
            ) AS covered_count
        """,
        (run_id, run_id, run_id, run_id),
    ).fetchone()
    if not counts:
        raise RuntimeError("Could not calculate run counters")

    run = conn.execute(
        """
        UPDATE runs
        SET table_count = %s,
            mention_count = %s,
            candidate_count = %s,
            covered_count = %s,
            raw_summary = coalesce(%s::jsonb, raw_summary)
        WHERE id = %s
        RETURNING name
        """,
        (
            int(counts["table_count"] or 0),
            int(counts["mention_count"] or 0),
            int(counts["candidate_count"] or 0),
            int(counts["covered_count"] or 0),
            Jsonb(raw_summary) if raw_summary is not None else None,
            run_id,
        ),
    ).fetchone()
    if not run:
        raise RuntimeError("Could not finalize run")
    return ImportStats(
        run_id=run_id,
        name=str(run["name"]),
        table_count=int(counts["table_count"] or 0),
        mention_count=int(counts["mention_count"] or 0),
        candidate_count=int(counts["candidate_count"] or 0),
        covered_count=int(counts["covered_count"] or 0),
    )
