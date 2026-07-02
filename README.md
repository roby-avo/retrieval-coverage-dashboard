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

The script starts Postgres if needed, mounts `./Datasets` into the loader container, and stores discovery metadata in the shared database volume used by both development and production Compose stacks. It defaults to the development stack. If the production stack is already running, target that Compose file explicitly:

```bash
./scripts/seed_source_data.sh --prod
```

It does not store table headers, table records, or ground-truth records. `source_tables` keeps file location, estimated row/column counts, file size, and per-table mention counts. Re-running the script imports newly added supported dataset directories while leaving already-populated metadata alone unless forced.

During an experiment, the runner samples from Postgres metadata and reads only the selected ground-truth rows plus the small selected table row window from the stored source paths.

Useful options:

```bash
SOURCE_DATA_FORCE=1 ./scripts/seed_source_data.sh
SOURCE_DATASETS=Round1_T2D,HardTablesR2 ./scripts/seed_source_data.sh
SOURCE_DATASETS=all ./scripts/seed_source_data.sh
DATASETS_DIR=/path/to/Datasets ./scripts/seed_source_data.sh
SOURCE_DATA_FORCE=1 ./scripts/seed_source_data.sh --prod
./scripts/seed_source_data.sh -f docker-compose.prod.yml
```

Then run the app:

```bash
docker compose up --build
```

For a production-style run that exposes only the frontend, use:

```bash
docker compose -f docker-compose.prod.yml up --build
```

The UI is served at `http://localhost:5173` by default. Set `WEB_PORT` to publish a different host port; the API and Postgres services stay internal to the Compose network.

LLM query planning is configured from the app's Settings tab. For OpenRouter, set the provider type to `OpenRouter`, paste an OpenRouter API key, and choose any OpenRouter model id such as `openai/gpt-oss-120b`. Leave the route provider blank to let OpenRouter route across compatible providers, or set it to a provider slug such as `cerebras` to prioritize that route, including configured OpenRouter BYOK keys. Provider fallbacks stay enabled by default; disable them only when you need strict routing.

For direct Cerebras, use the `Cerebras` provider or set `LLM_PROVIDER=cerebras` with `CEREBRAS_API_KEY`. The default direct Cerebras endpoint is `https://api.cerebras.ai/v1/chat/completions` and the default model is `gpt-oss-120b`. API keys are provider-specific: OpenRouter with route provider `cerebras` still uses an OpenRouter API key, while the direct `Cerebras` provider uses a Cerebras API key. The Settings tab includes a small "Test selected LLM" check that sends a minimal JSON chat-completion request using the currently selected provider/model before you run an experiment.

The Run Experiment panel can estimate LLM tokens and OpenRouter cost for the currently selected dataset target before starting the run. On-prem OpenAI-compatible servers can use a `/v1/chat/completions` URL, for example `http://localhost:8000/v1/chat/completions`.

Postgres is configured for local trusted access in Docker Compose. The Adminer database client is available at `http://localhost:8082` by default and auto-connects to the local Postgres server as `postgres` without an Adminer or database password. Override `ADMINER_PORT` in `.env` if you want a different host port.
