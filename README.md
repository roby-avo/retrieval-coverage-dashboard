# retrieval-coverage-dashboard
Dashboard to analyze metrics related to coverage in candidate retrieval task 

## Source Data

The dashboard discovers experiment source samples from Postgres metadata:

- `source_datasets`
- `source_tables`

Create a local `.env` from `.env.example` and fill in the Alpaca token before starting services:

```bash
cp .env.example .env
```

Populate that metadata explicitly before starting experiments:

```bash
./scripts/seed_source_data.sh
```

The script starts Postgres if needed, mounts `./Datasets` into the loader container, and stores discovery metadata in the database. It does not store table headers, table records, or ground-truth records. `source_tables` keeps file location, estimated row/column counts, file size, and per-table mention counts. Re-running the script imports newly added supported dataset directories while leaving already-populated metadata alone unless forced.

During an experiment, the runner samples from Postgres metadata and reads only the selected ground-truth rows plus the small selected table row window from the stored source paths.

Useful options:

```bash
SOURCE_DATA_FORCE=1 ./scripts/seed_source_data.sh
SOURCE_DATASETS=Round1_T2D,HardTablesR2 ./scripts/seed_source_data.sh
SOURCE_DATASETS=all ./scripts/seed_source_data.sh
DATASETS_DIR=/path/to/Datasets ./scripts/seed_source_data.sh
```

Then run the app:

```bash
docker compose up --build
```

LLM query planning is configured from the app's Settings tab. On-prem OpenAI-compatible servers can use a `/v1/chat/completions` URL, for example `http://localhost:8000/v1/chat/completions`.

Postgres is configured for local trusted access in Docker Compose. The Adminer database client is available at `http://localhost:8082` by default and auto-connects to the local Postgres server as `postgres` without an Adminer or database password. Override `ADMINER_PORT` in `.env` if you want a different host port.
