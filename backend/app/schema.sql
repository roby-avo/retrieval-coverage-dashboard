CREATE TABLE IF NOT EXISTS runs (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    source_path TEXT,
    source_filename TEXT,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_sampling_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    table_count INTEGER NOT NULL DEFAULT 0,
    mention_count INTEGER NOT NULL DEFAULT 0,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    covered_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_runs_imported_at ON runs (imported_at DESC);

CREATE TABLE IF NOT EXISTS source_datasets (
    id TEXT PRIMARY KEY,
    directory_name TEXT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    table_count INTEGER NOT NULL DEFAULT 0,
    mention_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS source_tables (
    dataset_id TEXT NOT NULL REFERENCES source_datasets(id) ON DELETE CASCADE,
    table_id TEXT NOT NULL,
    source_path TEXT,
    original_table_name TEXT,
    num_rows INTEGER NOT NULL DEFAULT 0,
    num_cols INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset_id, table_id)
);

CREATE INDEX IF NOT EXISTS idx_source_tables_dataset ON source_tables (dataset_id);

CREATE TABLE IF NOT EXISTS candidate_retrieval_cache (
    cache_key TEXT PRIMARY KEY,
    mention_text TEXT NOT NULL,
    normalized_mention TEXT NOT NULL,
    query_text TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    query_plan_source TEXT NOT NULL DEFAULT 'legacy',
    query_payload JSONB NOT NULL,
    response_payload JSONB NOT NULL,
    cached_candidate_count INTEGER NOT NULL,
    hit_count BIGINT NOT NULL DEFAULT 0,
    first_cached_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_cached_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_hit_at TIMESTAMPTZ
);

ALTER TABLE candidate_retrieval_cache
    ADD COLUMN IF NOT EXISTS query_plan_source TEXT NOT NULL DEFAULT 'legacy';

ALTER TABLE candidate_retrieval_cache
    DROP CONSTRAINT IF EXISTS candidate_retrieval_cache_normalized_mention_normalized_query_key;
ALTER TABLE candidate_retrieval_cache
    DROP CONSTRAINT IF EXISTS candidate_retrieval_cache_normalized_mention_normalized_que_key;

CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_retrieval_cache_unique_source
    ON candidate_retrieval_cache (normalized_mention, normalized_query, query_plan_source);
DROP INDEX IF EXISTS idx_candidate_retrieval_cache_lookup;
CREATE INDEX IF NOT EXISTS idx_candidate_retrieval_cache_lookup_source ON candidate_retrieval_cache (normalized_mention, normalized_query, query_plan_source, cached_candidate_count DESC);
CREATE INDEX IF NOT EXISTS idx_candidate_retrieval_cache_last_cached ON candidate_retrieval_cache (last_cached_at DESC);
CREATE INDEX IF NOT EXISTS idx_candidate_retrieval_cache_candidate_count ON candidate_retrieval_cache (cached_candidate_count DESC);

CREATE TABLE IF NOT EXISTS experiment_tables (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    dataset_id TEXT,
    table_id TEXT,
    sample_limit INTEGER,
    raw_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_profile JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (run_id, dataset_id, table_id)
);

CREATE INDEX IF NOT EXISTS idx_experiment_tables_run ON experiment_tables (run_id);
CREATE INDEX IF NOT EXISTS idx_experiment_tables_dataset ON experiment_tables (dataset_id);

CREATE TABLE IF NOT EXISTS llm_prompt_batches (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT REFERENCES runs(id) ON DELETE CASCADE,
    job_id BIGINT,
    provider TEXT,
    endpoint TEXT,
    model TEXT,
    prompt_template TEXT NOT NULL DEFAULT 'entity_retrieval_query_plan_v1',
    task_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'completed',
    error TEXT,
    request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_prompt_batches_run ON llm_prompt_batches (run_id, id);
CREATE INDEX IF NOT EXISTS idx_llm_prompt_batches_job ON llm_prompt_batches (job_id, id);
CREATE INDEX IF NOT EXISTS idx_llm_prompt_batches_job_status_created ON llm_prompt_batches (job_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_prompt_batches_created ON llm_prompt_batches (created_at DESC);

CREATE TABLE IF NOT EXISTS mentions (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    table_db_id BIGINT REFERENCES experiment_tables(id) ON DELETE CASCADE,
    query_plan_batch_id BIGINT REFERENCES llm_prompt_batches(id) ON DELETE SET NULL,
    candidate_cache_key TEXT REFERENCES candidate_retrieval_cache(cache_key) ON DELETE SET NULL,
    cell_key TEXT,
    dataset_id TEXT,
    table_id TEXT,
    row_id INTEGER,
    col_id INTEGER,
    mention_text TEXT,
    lookup_text TEXT,
    primary_gt_qid TEXT,
    best_gt_rank INTEGER,
    retrieved_count INTEGER NOT NULL DEFAULT 0,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    candidate_backend TEXT,
    query_engine TEXT,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, cell_key)
);

CREATE INDEX IF NOT EXISTS idx_mentions_run ON mentions (run_id, id);
CREATE INDEX IF NOT EXISTS idx_mentions_dataset ON mentions (dataset_id);
CREATE INDEX IF NOT EXISTS idx_mentions_table ON mentions (table_id);
CREATE INDEX IF NOT EXISTS idx_mentions_run_dataset ON mentions (run_id, dataset_id, id);
CREATE INDEX IF NOT EXISTS idx_mentions_run_best_rank ON mentions (run_id, best_gt_rank);
CREATE INDEX IF NOT EXISTS idx_mentions_candidate_cache_key ON mentions (candidate_cache_key, candidate_count DESC) WHERE candidate_cache_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mentions_lookup_text ON mentions USING gin (to_tsvector('simple', coalesce(mention_text, '') || ' ' || coalesce(lookup_text, '')));

CREATE TABLE IF NOT EXISTS gold_qids (
    id BIGSERIAL PRIMARY KEY,
    mention_id BIGINT NOT NULL REFERENCES mentions(id) ON DELETE CASCADE,
    qid TEXT NOT NULL,
    ordinal INTEGER NOT NULL DEFAULT 1,
    is_primary BOOLEAN NOT NULL DEFAULT false,
    raw_entity JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (mention_id, qid)
);

CREATE INDEX IF NOT EXISTS idx_gold_qids_mention ON gold_qids (mention_id);
CREATE INDEX IF NOT EXISTS idx_gold_qids_qid ON gold_qids (qid);

CREATE TABLE IF NOT EXISTS llm_prompt_tasks (
    id BIGSERIAL PRIMARY KEY,
    batch_id BIGINT NOT NULL REFERENCES llm_prompt_batches(id) ON DELETE CASCADE,
    mention_id BIGINT REFERENCES mentions(id) ON DELETE SET NULL,
    task_id TEXT NOT NULL,
    mention_text TEXT,
    lookup_text TEXT,
    plan_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (batch_id, task_id)
);

CREATE INDEX IF NOT EXISTS idx_llm_prompt_tasks_batch ON llm_prompt_tasks (batch_id, id);
CREATE INDEX IF NOT EXISTS idx_llm_prompt_tasks_mention ON llm_prompt_tasks (mention_id) WHERE mention_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_llm_prompt_tasks_task_id ON llm_prompt_tasks (task_id);

CREATE TABLE IF NOT EXISTS feedback_notes (
    id BIGSERIAL PRIMARY KEY,
    mention_id BIGINT NOT NULL REFERENCES mentions(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    category TEXT NOT NULL DEFAULT 'note',
    note TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_feedback_notes_mention ON feedback_notes (mention_id, created_at DESC);

CREATE TABLE IF NOT EXISTS live_attempts (
    id BIGSERIAL PRIMARY KEY,
    mention_id BIGINT NOT NULL REFERENCES mentions(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    candidate_count INTEGER NOT NULL,
    query_text TEXT,
    human_guidance TEXT,
    covered BOOLEAN NOT NULL DEFAULT false,
    covered_qids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_attempts_mention ON live_attempts (mention_id, created_at DESC);

CREATE TABLE IF NOT EXISTS experiment_jobs (
    id BIGSERIAL PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL DEFAULT 'queued',
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    message TEXT,
    error TEXT,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_path TEXT,
    query_plan_output_path TEXT,
    stage_progress JSONB NOT NULL DEFAULT '{}'::jsonb,
    imported_run_id BIGINT REFERENCES runs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_experiment_jobs_status ON experiment_jobs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiment_jobs_created ON experiment_jobs (created_at DESC);
