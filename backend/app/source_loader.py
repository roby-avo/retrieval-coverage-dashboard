from __future__ import annotations

import csv
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from psycopg.types.json import Jsonb

from .db import connect, init_database


QID_RE = re.compile(r"Q\d+")
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
DATASET_DIRECTORY_ALIASES = {
    "HardTablesR1_2022": "HardTableR1_2022",
    "HardTablesR2_2022": "HardTableR2_2022",
}


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _extract_qids(raw_value: str) -> list[str]:
    return _dedupe(QID_RE.findall(raw_value or ""))


def _discover_gt_path(dataset_dir: Path) -> Path:
    gt_dir = dataset_dir / "gt"
    candidates = [
        gt_dir / "cea.csv",
        gt_dir / "cea_gt.csv",
        *sorted(gt_dir.glob("CEA*_gt_WD.csv")),
        *sorted(gt_dir.glob("CEA*_gt.csv")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in sorted(gt_dir.glob("*.csv")):
        name = candidate.name.casefold()
        if name.startswith("cea") and "_gt" in name:
            return candidate
    raise FileNotFoundError(f"No supported CEA ground-truth file found under {gt_dir}")


def _resolve_dataset(root: Path, requested_name: str) -> dict[str, Any] | None:
    candidate_names = _dedupe([requested_name, DATASET_DIRECTORY_ALIASES.get(requested_name, "")])
    for candidate_name in candidate_names:
        dataset_dir = root / candidate_name
        if not dataset_dir.is_dir():
            continue
        gt_path = _discover_gt_path(dataset_dir)
        filename_map_path = dataset_dir / "gt" / "filename_map.json"
        table_dir = dataset_dir / "tables"
        return {
            "requested_name": requested_name,
            "directory_name": dataset_dir.name,
            "dataset_dir": dataset_dir,
            "gt_path": gt_path,
            "table_dir": table_dir,
            "table_file_index": {path.stem: path for path in table_dir.glob("*.csv")},
            "filename_map_path": filename_map_path if filename_map_path.exists() else None,
        }
    return None


@lru_cache(maxsize=16)
def _load_filename_map(path_text: str) -> dict[str, str]:
    path = Path(path_text)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_table_path(dataset_spec: dict[str, Any], table_id: str) -> Path:
    indexed = dataset_spec.get("table_file_index") or {}
    direct = indexed.get(table_id)
    if direct:
        return direct
    prefix_matches = [path for stem, path in indexed.items() if stem.startswith(table_id)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    raise FileNotFoundError(f"Table file for {dataset_spec['requested_name']} / {table_id} not found")


def _gt_table_stats(dataset_spec: dict[str, Any]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    gt_path = Path(dataset_spec["gt_path"])
    with gt_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            qids = _extract_qids(row[3])
            if not qids:
                continue
            try:
                row_id = int(row[1])
                col_id = int(row[2])
            except ValueError:
                continue
            table_stats = stats.setdefault(row[0], {"mention_count": 0, "max_row_id": 0, "max_col_id": 0})
            table_stats["mention_count"] += 1
            table_stats["max_row_id"] = max(table_stats["max_row_id"], row_id)
            table_stats["max_col_id"] = max(table_stats["max_col_id"], col_id)
    return stats


def _table_metadata(
    dataset_spec: dict[str, Any],
    table_id: str,
    *,
    source_root: Path,
    max_row_id: int,
    max_col_id: int,
    mention_count: int,
) -> dict[str, Any]:
    table_path = _resolve_table_path(dataset_spec, table_id)
    stat = table_path.stat()
    relative_path = table_path.relative_to(source_root)
    original_table_name = None
    if dataset_spec.get("filename_map_path"):
        original_table_name = _load_filename_map(str(dataset_spec["filename_map_path"])).get(table_id)
    return {
        "source_path": f"/source-data/{relative_path.as_posix()}",
        "num_rows": max_row_id + 1,
        "num_cols": max_col_id + 1,
        "original_table_name": original_table_name,
        "metadata": {
            "source_filename": table_path.name,
            "source_path": f"/source-data/{relative_path.as_posix()}",
            "relative_path": relative_path.as_posix(),
            "file_size_bytes": stat.st_size,
            "file_mtime": stat.st_mtime,
            "gt_mention_count": mention_count,
            "row_count_source": "max_ground_truth_row_id",
            "storage_mode": "filesystem_lazy",
            "stores_table_headers": False,
            "stores_table_records": False,
            "stores_ground_truth_records": False,
        },
    }


def _requested_datasets() -> list[str]:
    raw = os.environ.get("SOURCE_DATASETS")
    if not raw:
        return DEFAULT_DATASETS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _source_path(source_root: Path, path: Path) -> tuple[str, str]:
    relative_path = path.relative_to(source_root).as_posix()
    return f"/source-data/{relative_path}", relative_path


def seed_source_data(*, source_root: Path, requested_datasets: Sequence[str], force: bool = False) -> dict[str, Any]:
    init_database()
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"SOURCE_DATA_ROOT does not exist or is not a directory: {source_root}")

    imported: list[dict[str, Any]] = []
    warnings: list[str] = []
    with connect() as conn:
        existing = conn.execute(
            """
            SELECT
                (SELECT count(*) FROM source_datasets) AS dataset_count,
                (
                    SELECT count(*)
                    FROM source_tables
                    WHERE source_path IS NOT NULL
                      AND metadata->>'storage_mode' = 'filesystem_lazy'
                ) AS filesystem_table_count,
                (SELECT count(*) FROM source_tables) AS table_count,
                (SELECT array_agg(id) FROM source_datasets) AS dataset_ids
            """
        ).fetchone()
        dataset_count = int(existing["dataset_count"] if existing else 0)
        table_count = int(existing["table_count"] if existing else 0)
        filesystem_table_count = int(existing["filesystem_table_count"] if existing else 0)
        existing_dataset_ids = set(existing["dataset_ids"] or []) if existing else set()
        requested_dataset_ids = set(requested_datasets)
        metadata_complete = dataset_count > 0 and table_count > 0 and table_count == filesystem_table_count

        if force or (table_count and table_count != filesystem_table_count):
            with conn.transaction():
                conn.execute("DELETE FROM source_datasets")
        elif metadata_complete and requested_dataset_ids.issubset(existing_dataset_ids):
            return {"seeded": False, "reason": "source metadata already populated", "imported": [], "warnings": []}

        for dataset_id in requested_datasets:
            try:
                dataset_spec = _resolve_dataset(source_root, dataset_id)
                if dataset_spec is None:
                    warnings.append(f"{dataset_id}: dataset directory not found under {source_root}")
                    continue
                table_stats = _gt_table_stats(dataset_spec)
                gt_source_path, gt_relative_path = _source_path(source_root, Path(dataset_spec["gt_path"]))
            except Exception as exc:
                warnings.append(f"{dataset_id}: {exc}")
                continue

            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO source_datasets (id, directory_name, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        directory_name = EXCLUDED.directory_name,
                        imported_at = now(),
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        dataset_id,
                        dataset_spec["directory_name"],
                        Jsonb(
                            {
                                "gt_source_file": Path(dataset_spec["gt_path"]).name,
                                "gt_source_path": gt_source_path,
                                "gt_relative_path": gt_relative_path,
                                "storage_mode": "filesystem_lazy",
                                "stores_ground_truth_records": False,
                            }
                        ),
                    ),
                )

                conn.execute("DELETE FROM source_tables WHERE dataset_id = %s", (dataset_id,))

                imported_table_ids: set[str] = set()
                for table_id, stats in sorted(table_stats.items()):
                    try:
                        payload = _table_metadata(
                            dataset_spec,
                            table_id,
                            source_root=source_root,
                            max_row_id=stats["max_row_id"],
                            max_col_id=stats["max_col_id"],
                            mention_count=stats["mention_count"],
                        )
                    except Exception as exc:
                        warnings.append(f"{dataset_id}/{table_id}: {exc}")
                        continue
                    conn.execute(
                        """
                        INSERT INTO source_tables (
                            dataset_id, table_id, source_path, original_table_name, num_rows, num_cols, metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (dataset_id, table_id) DO UPDATE SET
                            source_path = EXCLUDED.source_path,
                            original_table_name = EXCLUDED.original_table_name,
                            num_rows = EXCLUDED.num_rows,
                            num_cols = EXCLUDED.num_cols,
                            metadata = EXCLUDED.metadata,
                            imported_at = now()
                        """,
                        (
                            dataset_id,
                            table_id,
                            payload["source_path"],
                            payload["original_table_name"],
                            payload["num_rows"],
                            payload["num_cols"],
                            Jsonb(payload["metadata"]),
                        ),
                    )
                    imported_table_ids.add(table_id)

                conn.execute(
                    """
                    UPDATE source_datasets
                    SET table_count = %s,
                        mention_count = %s,
                        imported_at = now()
                    WHERE id = %s
                    """,
                    (
                        len(imported_table_ids),
                        sum(stats["mention_count"] for table_id, stats in table_stats.items() if table_id in imported_table_ids),
                        dataset_id,
                    ),
                )
                imported.append(
                    {
                        "dataset_id": dataset_id,
                        "table_count": len(imported_table_ids),
                        "mention_count": sum(stats["mention_count"] for table_id, stats in table_stats.items() if table_id in imported_table_ids),
                    }
                )

    return {"seeded": True, "imported": imported, "warnings": warnings}


def main() -> None:
    source_root = Path(os.environ.get("SOURCE_DATA_ROOT", "/source-data"))
    force = os.environ.get("SOURCE_DATA_FORCE", "0").strip().lower() in {"1", "true", "yes"}
    result = seed_source_data(source_root=source_root, requested_datasets=_requested_datasets(), force=force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
