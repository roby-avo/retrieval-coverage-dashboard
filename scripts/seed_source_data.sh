#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="${DATASETS_DIR:-$ROOT_DIR/Datasets}"
COMPOSE_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/seed_source_data.sh [--dev|--prod|-f COMPOSE_FILE]

Seeds source dataset metadata into the Postgres database used by the selected
Docker Compose stack. Defaults to docker-compose.yml unless COMPOSE_FILE is set.

Options:
  --dev                 Use docker-compose.yml
  --prod, --production  Use docker-compose.prod.yml
  -f, --file FILE       Use a specific Compose file; may be repeated
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      COMPOSE_ARGS=(-f docker-compose.yml)
      shift
      ;;
    --prod|--production)
      COMPOSE_ARGS=(-f docker-compose.prod.yml)
      shift
      ;;
    -f|--file)
      if [[ $# -lt 2 ]]; then
        echo "Missing Compose file after $1" >&2
        exit 2
      fi
      COMPOSE_ARGS+=(-f "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$DATASETS_DIR" ]]; then
  echo "Datasets directory not found: $DATASETS_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ ${#COMPOSE_ARGS[@]} -gt 0 ]]; then
  echo "Using Docker Compose file(s): ${COMPOSE_ARGS[*]}" >&2
elif [[ -n "${COMPOSE_FILE:-}" ]]; then
  echo "Using Docker Compose file(s) from COMPOSE_FILE=$COMPOSE_FILE" >&2
else
  echo "Using Docker Compose default files" >&2
fi

docker compose "${COMPOSE_ARGS[@]}" up -d db
docker compose "${COMPOSE_ARGS[@]}" run --rm \
  -v "$DATASETS_DIR:/source-data:ro" \
  -e SOURCE_DATA_ROOT=/source-data \
  -e SOURCE_DATASETS="${SOURCE_DATASETS:-}" \
  -e SOURCE_DATA_FORCE="${SOURCE_DATA_FORCE:-0}" \
  api python -m app.source_loader
