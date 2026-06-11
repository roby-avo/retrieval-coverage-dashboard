#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="${DATASETS_DIR:-$ROOT_DIR/Datasets}"

if [[ ! -d "$DATASETS_DIR" ]]; then
  echo "Datasets directory not found: $DATASETS_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"

docker compose up -d db
docker compose run --rm \
  -v "$DATASETS_DIR:/source-data:ro" \
  -e SOURCE_DATA_ROOT=/source-data \
  -e SOURCE_DATASETS="${SOURCE_DATASETS:-}" \
  -e SOURCE_DATA_FORCE="${SOURCE_DATA_FORCE:-0}" \
  api python -m app.source_loader
