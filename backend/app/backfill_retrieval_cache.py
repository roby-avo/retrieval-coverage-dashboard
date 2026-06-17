from __future__ import annotations

import argparse
from typing import Any

from .db import connect, init_database
from .retrieval import alpaca_query_fingerprint


def _alpaca_request_payload(raw_payload: Any) -> dict[str, Any] | None:
    if not isinstance(raw_payload, dict):
        return None
    backend_requests = raw_payload.get("backend_requests")
    if not isinstance(backend_requests, list) or not backend_requests:
        return None
    first_request = backend_requests[0]
    if not isinstance(first_request, dict):
        return None
    request_payload = first_request.get("request")
    return request_payload if isinstance(request_payload, dict) else None


def backfill_retrieval_fingerprints(batch_size: int = 500, limit: int | None = None) -> int:
    total_updated = 0
    remaining = limit
    while True:
        current_batch_size = batch_size if remaining is None else min(batch_size, remaining)
        if current_batch_size <= 0:
            break
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    m.id,
                    m.raw_payload,
                    (
                        SELECT count(*)
                        FROM candidates c
                        WHERE c.mention_id = m.id
                    ) AS stored_candidate_count
                FROM mentions m
                WHERE m.raw_payload ? 'backend_requests'
                  AND (
                    m.retrieval_fingerprint IS NULL
                    OR m.retrieval_cache_candidate_count = 0
                  )
                ORDER BY m.id
                LIMIT %s
                """,
                (current_batch_size,),
            ).fetchall()
            if not rows:
                break

            updates: list[tuple[str, int, int]] = []
            for row in rows:
                request_payload = _alpaca_request_payload(row.get("raw_payload"))
                stored_candidate_count = int(row.get("stored_candidate_count") or 0)
                if not request_payload or stored_candidate_count <= 0:
                    continue
                updates.append(
                    (
                        alpaca_query_fingerprint(request_payload),
                        stored_candidate_count,
                        int(row["id"]),
                    )
                )

            if updates:
                conn.cursor().executemany(
                    """
                    UPDATE mentions
                    SET retrieval_fingerprint = %s,
                        retrieval_cache_candidate_count = %s
                    WHERE id = %s
                    """,
                    updates,
                )
                conn.commit()

            updated = len(updates)
            total_updated += updated
            if remaining is not None:
                remaining -= len(rows)
            if updated == 0 and len(rows) < current_batch_size:
                break
            print(f"updated={total_updated}")

    return total_updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill retrieval fingerprints for candidate-table cache reuse.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    init_database()
    updated = backfill_retrieval_fingerprints(batch_size=max(1, args.batch_size), limit=args.limit)
    print(f"done updated={updated}")


if __name__ == "__main__":
    main()
