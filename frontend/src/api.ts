export type Run = {
  id: number;
  name: string;
  source_path?: string;
  source_filename?: string;
  imported_at: string;
  table_count: number;
  mention_count: number;
  candidate_count: number;
  covered_count: number;
  imported_coverage?: number;
  raw_summary?: Record<string, unknown>;
};

export type ExperimentConfig = {
  name?: string;
  requested_datasets: string[];
  dataset_sample_size: number;
  tables_per_dataset: number;
  records_per_table: number;
  random_seed: number;
  context_rows: number;
  table_context_preview_rows: number;
  max_candidates: number;
  dashboard_candidate_limit: number;
  save_full_debug_output: boolean;
  enable_recall_query_expansion: boolean;
  recall_query_variant_limit: number;
  recall_context_term_limit: number;
  recall_token_combo_limit: number;
  enable_llm_url_hints: boolean;
  url_hint_boost: number;
  url_hint_confidence_threshold: number;
  dataset_allowlist: string[];
  table_allowlist_by_dataset: Record<string, string[]>;
  max_tasks_per_llm_request: number;
  llm_parallel_requests: number;
  openrouter_parallel_requests: number;
  max_workers: number;
  llm_enabled: boolean;
  llm_provider: string;
  llm_provider_name: string;
  llm_api_url: string;
  llm_api_key: string;
  llm_model: string;
  llm_reasoning_effort: string;
  llm_temperature: number;
  llm_timeout_seconds: number;
  llm_max_retries: number;
  llm_site_url: string;
  llm_app_name: string;
  llm_max_tokens?: number | null;
  use_openrouter: boolean;
  use_heuristic_fallback_on_llm_failure: boolean;
  openrouter_model: string;
  openrouter_provider: string;
};

export type StageProgress = {
  label: string;
  current: number;
  total: number;
  status: "running" | "completed" | "failed";
  started_at?: string;
  elapsed_seconds?: number;
  eta_seconds?: number | null;
  finished_at?: string | null;
};

export type ExperimentJob = {
  id: number;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  progress_current: number;
  progress_total: number;
  message?: string;
  error?: string;
  config: ExperimentConfig;
  output_path?: string;
  query_plan_output_path?: string;
  imported_run_id?: number;
  imported_run_name?: string;
  stage_progress?: Record<string, StageProgress>;
  created_at: string;
  started_at?: string;
  finished_at?: string;
};

export type ConfigStatus = {
  alpaca_configured: boolean;
  llm_configured: boolean;
  llm_provider?: string;
  llm_provider_name?: string;
  llm_api_url?: string;
  llm_model?: string;
  openrouter_configured: boolean;
};

export type SourceDataset = {
  dataset_id: string;
  directory_name?: string;
  table_count: number;
  mention_count: number;
  imported_at?: string;
  metadata?: Record<string, unknown>;
};

export type DatabaseSize = {
  database_name: string;
  total_bytes: number;
  total_pretty: string;
  tables: Array<{
    schema_name: string;
    table_name: string;
    total_bytes: number;
    table_bytes: number;
    index_bytes: number;
    estimated_rows: number;
  }>;
};

export type CoveragePoint = {
  k: number;
  total: number;
  covered: number;
  coverage: number;
};

export type MentionRow = {
  id: number;
  cell_key?: string;
  dataset_id?: string;
  table_id?: string;
  row_id?: number;
  col_id?: number;
  mention?: string;
  lookup_text?: string;
  primary_gt_qid?: string;
  selected_qid?: string;
  selected_label?: string;
  final_correct?: boolean;
  coverage_correct?: boolean;
  hit_at_1?: boolean;
  hit_at_5?: boolean;
  hit_at_10?: boolean;
  hit_at_k?: boolean;
  best_gt_rank?: number;
  retrieved_count?: number;
  candidate_count?: number;
  covered_by_imported_candidates: boolean;
  imported_best_rank?: number;
};

export type GoldQid = {
  qid: string;
  ordinal: number;
  is_primary: boolean;
  raw_entity?: Record<string, unknown>;
};

export type TableContextRow = {
  row_id?: number;
  relative_position?: number;
  is_target?: boolean;
  cells?: unknown[];
  mention_cell?: string;
};

export type TableContext = {
  header?: unknown[];
  target_row_id?: number;
  target_col_id?: number;
  header_cell?: string;
  rows?: TableContextRow[];
};

export type Candidate = {
  id?: number;
  rank: number;
  source_rank?: number;
  qid?: string;
  label?: string;
  item_category?: string;
  coarse_type?: string;
  fine_type?: string;
  retrieval_system?: string;
  retrieval_stage?: string;
  retrieval_stages?: string[];
  score?: number;
  es_score?: number;
  heuristic_score?: number;
  selected?: boolean;
  gold_match?: boolean;
  raw_payload?: Record<string, unknown>;
};

export type FeedbackNote = {
  id: number;
  created_at: string;
  category: string;
  note: string;
  metadata?: Record<string, unknown>;
};

export type LiveAttempt = {
  id: number;
  created_at: string;
  candidate_count: number;
  query_text?: string;
  human_guidance?: string;
  covered: boolean;
  covered_qids: string[];
  request_payload: Record<string, unknown>;
  response_payload: Record<string, unknown>;
  error?: string;
  candidates?: Candidate[];
};

export type ImprovementDiagnostics = {
  covered_in_retrieval_window?: boolean;
  covered_qids?: string[];
  retrieved_count?: number;
  gold_entities?: Array<{
    qid?: string;
    label?: string;
    description?: string;
    coarse_type?: string;
    fine_type?: string;
  }>;
  candidate_type_distribution?: Array<{
    coarse_type?: string;
    fine_type?: string;
    count?: number;
  }>;
  ner_type_hint?: {
    gold_types?: Array<{ qid?: string; coarse_type?: string; fine_type?: string }>;
    type_mismatch_with_top_candidates?: boolean;
    suggested_rule_tokens?: string[];
  };
  context_token_hint?: {
    mention_tokens?: string[];
    candidate_context_tokens_to_consider?: string[];
  };
  recommendations?: string[];
};

export type MentionDetail = {
  mention: MentionRow & {
    run_name: string;
    raw_payload?: Record<string, unknown>;
  };
  gold_qids: GoldQid[];
  candidates: Candidate[];
  feedback: FeedbackNote[];
  live_attempts: LiveAttempt[];
  table_context?: TableContext;
};

export type GoldMetadataResult = {
  requested_qids: string[];
  resolved_qid?: string | null;
  entity?: Candidate | null;
  all_found_qids: string[];
};

export type Filters = {
  datasets: Array<{ dataset_id: string; mention_count: number }>;
  retrieval_stages: Array<{ retrieval_stage: string; candidate_count: number }>;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  const text = await response.text();
  let data: any = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text };
    }
  }
  if (!response.ok) {
    const detail = data?.detail ?? data?.error ?? response.statusText;
    if (typeof detail === "string") throw new Error(detail);
    throw new Error(detail?.error ?? detail?.message ?? JSON.stringify(detail));
  }
  return data as T;
}

export const api = {
  experimentDefaults: () => request<ExperimentConfig>("/api/experiment-defaults"),
  configStatus: () => request<ConfigStatus>("/api/config-status"),
  sourceDatasets: () => request<SourceDataset[]>("/api/source-datasets"),
  databaseSize: () => request<DatabaseSize>("/api/database-size"),
  experimentJobs: () => request<ExperimentJob[]>("/api/experiment-jobs"),
  experimentJob: (id: number) => request<ExperimentJob>(`/api/experiment-jobs/${id}`),
  clearFailedExperimentJobs: () =>
    request<{ deleted: number }>("/api/experiment-jobs/failed", {
      method: "DELETE"
    }),
  startExperimentJob: (config: Partial<ExperimentConfig>) =>
    request<ExperimentJob>("/api/experiment-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config })
    }),
  runs: () => request<Run[]>("/api/runs"),
  run: (id: number) => request<Run>(`/api/runs/${id}`),
  filters: (id: number) => request<Filters>(`/api/runs/${id}/filters`),
  coverage: (id: number) => request<CoveragePoint[]>(`/api/runs/${id}/coverage`),
  mentions: (
    id: number,
    params: { offset?: number; limit?: number; covered?: string; dataset_id?: string; search?: string }
  ) => {
    const search = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
    });
    return request<{ total: number; rows: MentionRow[] }>(`/api/runs/${id}/mentions?${search}`);
  },
  mention: (id: number) => request<MentionDetail>(`/api/mentions/${id}`),
  goldMetadata: (id: number) => request<GoldMetadataResult>(`/api/mentions/${id}/gold-metadata`),
  addFeedback: (mentionId: number, body: { category: string; note: string }) =>
    request<FeedbackNote>(`/api/mentions/${mentionId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  liveAttempt: (mentionId: number, body: { candidate_count: number; query_text?: string; human_guidance?: string; llm_config?: Partial<ExperimentConfig> }) =>
    request<LiveAttempt>(`/api/mentions/${mentionId}/live-attempt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    })
};
