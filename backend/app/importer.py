from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .retrieval import MAX_RETURNED_CANDIDATES, candidate_retrieval_cache_key


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


def _limited_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates[:MAX_RETURNED_CANDIDATES] if isinstance(candidate, dict)]


def _mention_metadata(row: dict[str, Any]) -> dict[str, Any]:
    duplicated_payloads = {"candidates", "reranked", "backend_response"}
    return {key: value for key, value in row.items() if key not in duplicated_payloads}


def _query_plan_batch_id(row: dict[str, Any]) -> int | None:
    query_plan = row.get("query_plan")
    if not isinstance(query_plan, dict):
        return None
    llm_inspection = query_plan.get("llm_inspection")
    if not isinstance(llm_inspection, dict):
        return None
    return _as_int(llm_inspection.get("batch_id"))


def _candidate_cache_key(row: dict[str, Any]) -> str | None:
    if not _limited_candidates(row):
        return None
    query_text = row.get("query_text")
    if not query_text and isinstance(row.get("query_plan"), dict):
        query_text = row["query_plan"].get("optimized_query")
    query_plan_source = row.get("query_engine")
    if not query_plan_source and isinstance(row.get("query_plan"), dict):
        query_plan_source = row["query_plan"].get("query_plan_source")
    return candidate_retrieval_cache_key(row.get("mention_text") or row.get("lookup_text"), query_text, query_plan_source or "heuristic")


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
    candidate_count = len(_limited_candidates(row))
    best_gt_rank = _as_int(row.get("best_gt_rank") or row.get("gold_rank"))
    candidate_cache_key = _candidate_cache_key(row)
    query_plan_batch_id = _query_plan_batch_id(row)
    mention = conn.execute(
        """
        INSERT INTO mentions (
            run_id, table_db_id, query_plan_batch_id, candidate_cache_key,
            cell_key, dataset_id, table_id, row_id, col_id, mention_text,
            lookup_text, primary_gt_qid, best_gt_rank, retrieved_count,
            candidate_count, candidate_backend, query_engine, raw_payload
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (run_id, cell_key) DO NOTHING
        RETURNING id
        """,
        (
            run_id,
            table_db_id,
            query_plan_batch_id,
            candidate_cache_key,
            _as_text(row.get("cell_key")),
            _as_text(row.get("dataset_id")),
            _as_text(row.get("table_id")),
            _as_int(row.get("row_id")),
            _as_int(row.get("col_id")),
            _as_text(row.get("mention_text") or row.get("mention")),
            _as_text(row.get("lookup_text")),
            _as_text(row.get("primary_gt_qid")) or (qids[0] if qids else None),
            best_gt_rank,
            _as_int(row.get("retrieved_count")) or candidate_count,
            candidate_count,
            _as_text(row.get("candidate_backend")),
            _as_text(row.get("query_engine")),
            Jsonb(_mention_metadata(row)),
        ),
    ).fetchone()
    if not mention:
        return RowImportStats(mention_count=0, candidate_count=0, covered_count=0)

    mention_id = int(mention["id"])
    if query_plan_batch_id:
        conn.execute(
            """
            UPDATE llm_prompt_tasks
            SET mention_id = %s
            WHERE batch_id = %s
              AND task_id = %s
            """,
            (mention_id, query_plan_batch_id, _as_text(row.get("cell_key"))),
        )

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

    return RowImportStats(
        mention_count=1,
        candidate_count=candidate_count,
        covered_count=1 if best_gt_rank is not None else 0,
    )


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
            (SELECT coalesce(sum(candidate_count), 0) FROM mentions WHERE run_id = %s) AS candidate_count,
            (SELECT count(*) FROM mentions WHERE run_id = %s AND best_gt_rank IS NOT NULL) AS covered_count
        """,
        (run_id, run_id, run_id, run_id),
    ).fetchone()
    if not counts:
        raise RuntimeError("Could not finalize run")
    run = conn.execute(
        """
        UPDATE runs
        SET table_count = %s,
            mention_count = %s,
            candidate_count = %s,
            covered_count = %s,
            raw_summary = %s
        WHERE id = %s
        RETURNING id, name
        """,
        (
            int(counts["table_count"] or 0),
            int(counts["mention_count"] or 0),
            int(counts["candidate_count"] or 0),
            int(counts["covered_count"] or 0),
            Jsonb(raw_summary or {}),
            run_id,
        ),
    ).fetchone()
    if not run:
        raise RuntimeError("Could not update run summary")
    return ImportStats(
        run_id=int(run["id"]),
        name=str(run["name"]),
        table_count=int(counts["table_count"] or 0),
        mention_count=int(counts["mention_count"] or 0),
        candidate_count=int(counts["candidate_count"] or 0),
        covered_count=int(counts["covered_count"] or 0),
    )
