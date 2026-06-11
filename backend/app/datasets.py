from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import psycopg

from .db import connect


TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]+")
QID_RE = re.compile(r"Q\d+")
GENERIC_HEADER_RE = re.compile(r"col\d+", re.IGNORECASE)
BRACKETED_NOTE_RE = re.compile(r"\[[^\]]+\]")


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text or "")]


def canonical_lookup_text(text: str) -> str:
    cleaned = BRACKETED_NOTE_RE.sub(" ", text or "")
    cleaned = cleaned.replace("*", " ")
    return " ".join(_tokenize(cleaned))


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def sample_lookup_id(sample: dict[str, Any]) -> str:
    return (
        f"{sample.get('dataset', '')}::"
        f"{sample.get('table_id', '')}::"
        f"{sample.get('row_id', '')}::"
        f"{sample.get('col_id', '')}"
    )


def source_dataset_inventory(conn: psycopg.Connection[Any]) -> list[dict[str, Any]]:
    return list(
        conn.execute(
            """
            SELECT id AS dataset_id, directory_name, table_count, mention_count, imported_at, metadata
            FROM source_datasets
            ORDER BY id
            """
        ).fetchall()
    )


def _sample_count(limit: int | None, available: int) -> int:
    if limit is None or int(limit) <= 0:
        return available
    return min(int(limit), available)


def _is_generic_header(value: str) -> bool:
    header = (value or "").strip()
    if not header:
        return True
    if header == "#":
        return True
    return bool(GENERIC_HEADER_RE.fullmatch(header))


def _build_lookup_context(sample: dict[str, Any]) -> list[str]:
    snippets: list[str] = []
    for value in sample.get("header", []):
        if not _is_generic_header(value):
            snippets.append(value)
    for index, value in enumerate(sample.get("target_row", [])):
        if index != sample["col_id"] and value:
            snippets.append(value)
    for row in sample.get("rows_before", []) + sample.get("rows_after", []):
        for value in row:
            if value:
                snippets.append(value)
    return snippets


def _source_root() -> Path:
    return Path(os.environ.get("SOURCE_DATA_ROOT", "/source-data")).resolve()


def _runtime_table_path(table_row: dict[str, Any]) -> Path | None:
    metadata = table_row.get("metadata") if isinstance(table_row.get("metadata"), dict) else {}
    raw_path = table_row.get("source_path") or metadata.get("source_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path
        if path.is_absolute():
            rebased = _source_root() / str(path).removeprefix("/source-data/").lstrip("/")
            if rebased.exists():
                return rebased
    relative_path = metadata.get("relative_path")
    if relative_path:
        path = _source_root() / str(relative_path)
        if path.exists():
            return path
    return None


def _runtime_dataset_path(dataset: dict[str, Any], metadata_key: str, relative_key: str) -> Path | None:
    metadata = dataset.get("metadata") if isinstance(dataset.get("metadata"), dict) else {}
    raw_path = metadata.get(metadata_key)
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path
        if path.is_absolute():
            rebased = _source_root() / str(path).removeprefix("/source-data/").lstrip("/")
            if rebased.exists():
                return rebased
    relative_path = metadata.get(relative_key)
    if relative_path:
        path = _source_root() / str(relative_path)
        if path.exists():
            return path
    return None


def _extract_qids(raw_value: str) -> list[str]:
    return dedupe_preserve_order(QID_RE.findall(raw_value or ""))


def _read_gt_rows_for_tables(dataset: dict[str, Any], table_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    gt_path = _runtime_dataset_path(dataset, "gt_source_path", "gt_relative_path")
    if gt_path is None:
        raise FileNotFoundError(f"GT source file is not available for dataset {dataset['dataset_id']}")

    by_table: dict[str, list[dict[str, Any]]] = {table_id: [] for table_id in table_ids}
    with gt_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4 or row[0] not in table_ids:
                continue
            qids = _extract_qids(row[3])
            if not qids:
                continue
            try:
                row_id = int(row[1])
                col_id = int(row[2])
            except ValueError:
                continue
            by_table[row[0]].append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "table_id": row[0],
                    "row_id": row_id,
                    "col_id": col_id,
                    "gt_raw_value": row[3],
                    "qids": qids,
                    "primary_qid": qids[0],
                    "gt_source_file": gt_path.name,
                }
            )
    return by_table


def _csv_table_context(
    table_path: Path,
    table_row: dict[str, Any],
    *,
    row_id: int,
    col_id: int,
    context_rows: int,
) -> dict[str, Any]:
    data_index = row_id - 1
    if data_index < 0:
        data_index = row_id
    start_index = max(0, data_index - context_rows)
    end_index = data_index + context_rows
    header: list[str] = []
    selected_rows: dict[int, list[str]] = {}

    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration:
            header = []
        for index, row in enumerate(reader):
            if index > end_index:
                break
            if index >= start_index:
                selected_rows[index] = [cell.strip() for cell in row]

    if data_index not in selected_rows:
        raise IndexError(f"Row {row_id} is out of bounds for {table_row['dataset_id']} / {table_row['table_id']}")

    target_row = selected_rows[data_index]
    rows_before = [selected_rows[index] for index in range(start_index, data_index) if index in selected_rows]
    rows_after = [selected_rows[index] for index in range(data_index + 1, end_index + 1) if index in selected_rows]
    mention_text = target_row[col_id] if col_id < len(target_row) else ""
    header_cell = header[col_id] if col_id < len(header) else ""
    return {
        "original_table_name": table_row.get("original_table_name"),
        "header": header,
        "header_cell": header_cell,
        "rows_before": rows_before,
        "target_row": target_row,
        "rows_after": rows_after,
        "mention_text": mention_text,
        "table_num_rows": int(table_row.get("num_rows") or 0),
        "table_num_cols": int(table_row.get("num_cols") or max((len(row) for row in selected_rows.values()), default=0)),
        "table_source_path": str(table_path),
    }


def _table_context(
    table_row: dict[str, Any],
    *,
    row_id: int,
    col_id: int,
    context_rows: int,
) -> dict[str, Any]:
    table_path = _runtime_table_path(table_row)
    if table_path is None:
        raise FileNotFoundError(f"Table source file is not available for {table_row['dataset_id']} / {table_row['table_id']}")
    return _csv_table_context(table_path, table_row, row_id=row_id, col_id=col_id, context_rows=context_rows)


def build_random_sample_bundle_from_db(config: dict[str, Any]) -> dict[str, Any]:
    requested = [str(item) for item in config.get("requested_datasets") or [] if str(item).strip()]
    dataset_allowlist = set(config.get("dataset_allowlist") or [])
    table_allowlist_by_dataset = config.get("table_allowlist_by_dataset") or {}
    context_rows = int(config.get("context_rows") or 0)

    with connect() as conn:
        inventory = source_dataset_inventory(conn)
        if requested:
            inventory = [item for item in inventory if item["dataset_id"] in set(requested)]
        if dataset_allowlist:
            inventory = [item for item in inventory if item["dataset_id"] in dataset_allowlist]
        inventory.sort(key=lambda item: item["dataset_id"])

        rng = random.Random(int(config.get("random_seed") or 0))
        dataset_count = _sample_count(config.get("dataset_sample_size"), len(inventory))
        selected_datasets = rng.sample(inventory, dataset_count) if dataset_count else []
        selected_datasets.sort(key=lambda item: item["dataset_id"])

        samples: list[dict[str, Any]] = []
        sampling_manifest: list[dict[str, Any]] = []
        warnings: list[str] = []

        for dataset in selected_datasets:
            dataset_id = dataset["dataset_id"]
            table_rows = list(
                conn.execute(
                    """
                    SELECT table_id, metadata, coalesce((metadata->>'gt_mention_count')::int, 0) AS available_records
                    FROM source_tables
                    WHERE dataset_id = %s
                      AND coalesce((metadata->>'gt_mention_count')::int, 0) > 0
                    ORDER BY table_id
                    """,
                    (dataset_id,),
                ).fetchall()
            )
            available_by_table = {str(row["table_id"]): int(row["available_records"] or 0) for row in table_rows}
            table_ids = sorted(available_by_table)
            allowlisted_tables = table_allowlist_by_dataset.get(dataset_id) or []
            if allowlisted_tables:
                allowed = {str(table_id) for table_id in allowlisted_tables}
                table_ids = [table_id for table_id in table_ids if table_id in allowed]

            dataset_rng = random.Random(f"{int(config.get('random_seed') or 0)}:{dataset_id}")
            table_count = _sample_count(config.get("tables_per_dataset"), len(table_ids))
            selected_table_ids = dataset_rng.sample(table_ids, table_count) if table_count else []
            selected_table_ids.sort()
            gt_rows_by_table = _read_gt_rows_for_tables(dataset, set(selected_table_ids)) if selected_table_ids else {}

            for table_id in selected_table_ids:
                available_records = available_by_table.get(table_id, 0)
                record_count = _sample_count(config.get("records_per_table"), available_records)
                table_payload = conn.execute(
                    """
                    SELECT dataset_id, table_id, source_path, original_table_name, num_rows, num_cols, metadata
                    FROM source_tables
                    WHERE dataset_id = %s AND table_id = %s
                    """,
                    (dataset_id, table_id),
                ).fetchone()
                if not table_payload:
                    warnings.append(f"{dataset_id}/{table_id}: source table missing")
                    continue
                records = gt_rows_by_table.get(table_id, [])
                selected_records = dataset_rng.sample(records, record_count) if record_count else []
                selected_records.sort(key=lambda item: (item["row_id"], item["col_id"], item["gt_raw_value"]))
                sampling_manifest.append(
                    {
                        "dataset": dataset_id,
                        "table_id": table_id,
                        "available_records": available_records,
                        "sampled_records": len(selected_records),
                    }
                )

                for record in selected_records:
                    try:
                        context = _table_context(table_payload, row_id=int(record["row_id"]), col_id=int(record["col_id"]), context_rows=context_rows)
                    except (FileNotFoundError, IndexError) as exc:
                        warnings.append(str(exc))
                        continue
                    sample = {
                        "dataset": dataset_id,
                        "dataset_dir": dataset.get("directory_name"),
                        "table_id": table_id,
                        "row_id": int(record["row_id"]),
                        "col_id": int(record["col_id"]),
                        "gt_source_file": record.get("gt_source_file"),
                        "gt_raw_value": record.get("gt_raw_value"),
                        "gt_qids": list(record.get("qids") or []),
                        "primary_gt_qid": record.get("primary_qid"),
                        **context,
                    }
                    sample["lookup_text"] = canonical_lookup_text(sample["mention_text"])
                    sample["lookup_context"] = _build_lookup_context(sample)
                    sample["sample_id"] = sample_lookup_id(sample)
                    sample["lookup_wikipedia_url"] = None
                    sample["lookup_dbpedia_url"] = None
                    sample["lookup_url_hint_source"] = None
                    sample["url_hint_confidence"] = None
                    sample["url_hint_reason"] = None
                    samples.append(sample)

        return {
            "warnings": warnings,
            "dataset_inventory": selected_datasets,
            "sampling_manifest": sampling_manifest,
            "samples": samples,
        }
