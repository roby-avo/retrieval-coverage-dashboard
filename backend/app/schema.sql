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

ALTER TABLE source_tables
    ADD COLUMN IF NOT EXISTS source_path TEXT;

DROP TABLE IF EXISTS source_ground_truth;

ALTER TABLE source_tables
    DROP COLUMN IF EXISTS header,
    DROP COLUMN IF EXISTS rows;

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

CREATE TABLE IF NOT EXISTS mentions (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    table_db_id BIGINT REFERENCES experiment_tables(id) ON DELETE CASCADE,
    cell_key TEXT,
    dataset_id TEXT,
    table_id TEXT,
    row_id INTEGER,
    col_id INTEGER,
    mention TEXT,
    mention_text TEXT,
    lookup_text TEXT,
    primary_gt_qid TEXT,
    selected_qid TEXT,
    selected_label TEXT,
    final_correct BOOLEAN,
    coverage_correct BOOLEAN,
    hit_at_1 BOOLEAN,
    hit_at_5 BOOLEAN,
    hit_at_10 BOOLEAN,
    hit_at_k BOOLEAN,
    best_gt_rank INTEGER,
    retrieved_count INTEGER,
    candidate_count INTEGER,
    candidate_backend TEXT,
    query_engine TEXT,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, cell_key)
);

CREATE INDEX IF NOT EXISTS idx_mentions_run ON mentions (run_id, id);
CREATE INDEX IF NOT EXISTS idx_mentions_dataset ON mentions (dataset_id);
CREATE INDEX IF NOT EXISTS idx_mentions_table ON mentions (table_id);
CREATE INDEX IF NOT EXISTS idx_mentions_lookup_text ON mentions USING gin (to_tsvector('simple', coalesce(mention, '') || ' ' || coalesce(lookup_text, '')));

ALTER TABLE mentions
    ADD COLUMN IF NOT EXISTS retrieval_fingerprint TEXT,
    ADD COLUMN IF NOT EXISTS retrieval_cache_candidate_count INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_mentions_retrieval_fingerprint ON mentions (retrieval_fingerprint, retrieval_cache_candidate_count DESC) WHERE retrieval_fingerprint IS NOT NULL;

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

CREATE TABLE IF NOT EXISTS candidates (
    id BIGSERIAL PRIMARY KEY,
    mention_id BIGINT NOT NULL REFERENCES mentions(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    source_rank INTEGER,
    qid TEXT,
    label TEXT,
    item_category TEXT,
    coarse_type TEXT,
    fine_type TEXT,
    retrieval_system TEXT,
    retrieval_stage TEXT,
    retrieval_stages TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    score DOUBLE PRECISION,
    es_score DOUBLE PRECISION,
    heuristic_score DOUBLE PRECISION,
    selected BOOLEAN NOT NULL DEFAULT false,
    gold_match BOOLEAN NOT NULL DEFAULT false,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (mention_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_candidates_mention_rank ON candidates (mention_id, rank);
CREATE INDEX IF NOT EXISTS idx_candidates_mention_qid_rank ON candidates (mention_id, qid, rank) WHERE qid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_candidates_qid ON candidates (qid);
CREATE INDEX IF NOT EXISTS idx_candidates_stage ON candidates (retrieval_stage);
CREATE INDEX IF NOT EXISTS idx_candidates_gold_match ON candidates (gold_match);
CREATE INDEX IF NOT EXISTS idx_mentions_run_dataset ON mentions (run_id, dataset_id, id);
CREATE INDEX IF NOT EXISTS idx_mentions_run_best_rank ON mentions (run_id, best_gt_rank);

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
    imported_run_id BIGINT REFERENCES runs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

ALTER TABLE experiment_jobs
    ADD COLUMN IF NOT EXISTS stage_progress JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_experiment_jobs_status ON experiment_jobs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiment_jobs_created ON experiment_jobs (created_at DESC);

CREATE TABLE IF NOT EXISTS alpaca_search_cache (
    fingerprint TEXT PRIMARY KEY,
    query_payload JSONB NOT NULL,
    response_payload JSONB NOT NULL,
    cached_candidate_count INTEGER NOT NULL,
    hit_count BIGINT NOT NULL DEFAULT 0,
    first_cached_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_cached_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_hit_at TIMESTAMPTZ
);

ALTER TABLE alpaca_search_cache
    ADD COLUMN IF NOT EXISTS cached_candidate_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS hit_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS first_cached_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS last_cached_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS last_hit_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_alpaca_search_cache_last_cached ON alpaca_search_cache (last_cached_at DESC);
CREATE INDEX IF NOT EXISTS idx_alpaca_search_cache_candidate_count ON alpaca_search_cache (cached_candidate_count DESC);
