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
  llm_query_plan_batch_count?: number;
  llm_query_plan_completed_count?: number;
  llm_query_plan_failed_count?: number;
  llm_query_plan_incomplete_batch_count?: number;
  llm_query_plan_requested_task_count?: number;
  llm_query_plan_returned_task_count?: number;
  llm_query_plan_usable_task_count?: number;
  llm_query_plan_missing_task_count?: number;
  llm_query_plan_prompt_tokens?: number;
  llm_query_plan_completion_tokens?: number;
  llm_query_plan_total_tokens?: number;
  llm_query_plan_total_cost_usd?: number;
  llm_query_plan_priced_batch_count?: number;
  llm_query_plan_response_reported_cost_count?: number;
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
  openrouter_api_key?: string;
  cerebras_api_key?: string;
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
  openrouter_allow_fallbacks: boolean;
  openrouter_model: string;
  openrouter_provider: string;
};

export type StageProgress = {
  label: string;
  current: number;
  total: number;
  status: "running" | "completed" | "failed" | "cancelled";
  started_at?: string;
  elapsed_seconds?: number;
  eta_seconds?: number | null;
  finished_at?: string | null;
};

export type ExperimentJob = {
  id: number;
  status: "queued" | "running" | "completed" | "failed" | "cancel_requested" | "cancelled";
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
  llm_query_plan_batch_count?: number;
  llm_query_plan_completed_count?: number;
  llm_query_plan_failed_count?: number;
  llm_query_plan_incomplete_batch_count?: number;
  llm_query_plan_requested_task_count?: number;
  llm_query_plan_returned_task_count?: number;
  llm_query_plan_usable_task_count?: number;
  llm_query_plan_missing_task_count?: number;
  llm_query_plan_prompt_tokens?: number;
  llm_query_plan_completion_tokens?: number;
  llm_query_plan_total_tokens?: number;
  llm_query_plan_total_cost_usd?: number;
  llm_query_plan_priced_batch_count?: number;
  llm_query_plan_response_reported_cost_count?: number;
  stage_progress?: Record<string, StageProgress>;
  created_at: string;
  started_at?: string;
  finished_at?: string;
};

export type LlmQueryPlanBatch = {
  id: number;
  run_id?: number | null;
  job_id?: number | null;
  provider?: string | null;
  endpoint?: string | null;
  model?: string | null;
  prompt_template: string;
  task_count: number;
  status: string;
  error?: string | null;
  created_at: string;
  returned_task_count?: number;
  matched_returned_task_count?: number;
  usable_task_count?: number;
  missing_task_count?: number;
  missing_task_ids?: string[];
  requested_task_ids?: string[];
  unknown_returned_task_count?: number;
  unknown_returned_task_ids?: string[];
  invalid_task_count?: number;
  response_parse_error?: string | null;
  explanation?: string;
  parse_warning?: string | null;
  attempts?: unknown;
  usage?: unknown;
  usage_cost?: {
    cost_kind?: string;
    pricing_source?: string | null;
    total_cost_usd?: number | null;
    input_cost_usd?: number | null;
    output_cost_usd?: number | null;
    prompt_cost_usd?: number | null;
    completion_cost_usd?: number | null;
    request_cost_usd?: number | null;
    input_tokens?: number;
    output_tokens?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    token_source?: string;
    request_count?: number;
    notes?: string[];
  } | null;
  heuristic_analysis?: {
    heuristic_plan_count?: number;
    zero_candidate_count?: number;
    retrieval_error_count?: number;
    heuristic_retrieval_problem_count?: number;
  };
  response_content?: string | null;
  parsed_response?: Record<string, unknown> | null;
  task_details?: Array<{
    task_id: string;
    mention_id?: number | null;
    mention_text?: string | null;
    lookup_text?: string | null;
    state: "usable" | "missing" | "returned_not_usable";
    returned: boolean;
    usable: boolean;
    plan_source?: string;
    optimized_query?: string | null;
    normalized_mention?: string | null;
    coarse_type?: string | null;
    fine_type?: string | null;
    wikipedia_url?: string | null;
    dbpedia_url?: string | null;
    candidate_count?: number | null;
    retrieved_count?: number | null;
    best_gt_rank?: number | null;
    retrieval_error?: string | null;
    troubleshooting_flags?: string[];
  }>;
  returned_tasks?: Array<{
    id?: string | null;
    optimized_query?: string | null;
    normalized_mention?: string | null;
    coarse_type?: string | null;
    fine_type?: string | null;
    wikipedia_url?: string | null;
    dbpedia_url?: string | null;
    aliases?: unknown[];
    context_expansion_terms?: unknown[];
  }>;
  unknown_returned_tasks?: Array<{
    id?: string | null;
    optimized_query?: string | null;
    normalized_mention?: string | null;
    coarse_type?: string | null;
    fine_type?: string | null;
    wikipedia_url?: string | null;
    dbpedia_url?: string | null;
  }>;
};

export type HeuristicPlanAnalysis = {
  summary: {
    heuristic_plan_count?: number;
    llm_fallback_error_count?: number;
    zero_candidate_count?: number;
    retrieval_error_count?: number;
    retrieval_problem_count?: number;
    missed_count?: number;
    covered_count?: number;
    coverage?: number;
  };
  rows: Array<{
    id: number;
    cell_key?: string | null;
    dataset_id?: string | null;
    table_id?: string | null;
    row_id?: number | null;
    col_id?: number | null;
    mention_text?: string | null;
    lookup_text?: string | null;
    primary_gt_qid?: string | null;
    candidate_count?: number;
    retrieved_count?: number | null;
    best_gt_rank?: number | null;
    query_engine?: string | null;
    query_plan_source?: string | null;
    query_plan_error?: string | null;
    retrieval_error?: string | null;
    optimized_query?: string | null;
    normalized_mention?: string | null;
    coarse_type?: string | null;
    fine_type?: string | null;
    troubleshooting_flags?: string[];
  }>;
};

export type ConfigStatus = {
  alpaca_configured: boolean;
  llm_configured: boolean;
  llm_provider?: string;
  llm_provider_name?: string;
  llm_api_url?: string;
  llm_model?: string;
  openrouter_configured: boolean;
  cerebras_configured?: boolean;
  openrouter_allow_fallbacks?: boolean;
};

export type LlmTestResult = {
  ok: boolean;
  provider?: string | null;
  provider_name?: string | null;
  endpoint?: string | null;
  model?: string | null;
  response_model?: string | null;
  response_provider?: string | null;
  response_usage?: unknown;
  usage_cost?: unknown;
  response_id?: string | null;
  content?: string;
};

export type LlmUsageEstimate = {
  model: string;
  provider?: string | null;
  route_provider?: string | null;
  estimation_method: string;
  token_estimate: {
    prompt_tokens: number;
    input_tokens?: number;
    max_completion_tokens?: number;
    estimated_completion_tokens?: number;
    output_tokens?: number;
    completion_tokens_per_request?: number;
    completion_token_source?: string;
    total_tokens: number;
  };
  pricing_source?: string | null;
  pricing?: {
    input_per_token?: number | null;
    output_per_token?: number | null;
    prompt_per_token?: number | null;
    completion_per_token?: number | null;
    request?: number | null;
    input_per_million?: number | null;
    output_per_million?: number | null;
    prompt_per_million?: number | null;
    completion_per_million?: number | null;
    estimated_input_cost_usd?: number | null;
    estimated_output_cost_usd?: number | null;
    estimated_prompt_cost_usd?: number | null;
    estimated_completion_cost_usd?: number | null;
    estimated_subtotal_cost_usd?: number | null;
    estimated_total_cost_usd?: number | null;
    cost_safety_multiplier?: number;
  } | null;
  model_info?: {
    id?: string;
    name?: string;
    context_length?: number;
    top_provider?: Record<string, unknown>;
  } | null;
  notes: string[];
  target?: {
    sampled_mentions: number;
    llm_request_count: number;
    max_tasks_per_llm_request: number;
    dataset_inventory?: Array<Record<string, unknown>>;
    sampling_manifest?: Array<Record<string, unknown>>;
    warnings?: string[];
  };
  request_estimates?: Array<Record<string, unknown>>;
};

export type SourceDataset = {
  dataset_id: string;
  directory_name?: string;
  table_count: number;
  mention_count: number;
  imported_at?: string;
  metadata?: Record<string, unknown>;
};

export type SourceDiscoveryResult = {
  seeded: boolean;
  reason?: string;
  source_root: string;
  requested_datasets: string[];
  imported: Array<{
    dataset_id: string;
    table_count: number;
    mention_count: number;
  }>;
  warnings: string[];
  inventory: SourceDataset[];
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
  query_plan_batch?: Record<string, unknown> | null;
};

export type GoldMetadataResult = {
  requested_qids: string[];
  resolved_qid?: string | null;
  entity?: Candidate | null;
  entities?: Candidate[];
  ner_types?: Array<{ qid?: string; coarse_type?: string | null; fine_type?: string | null }>;
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
  discoverSourceDatasets: (body: { source_root?: string; requested_datasets?: string[]; force?: boolean } = {}) =>
    request<SourceDiscoveryResult>("/api/source-datasets/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  databaseSize: () => request<DatabaseSize>("/api/database-size"),
  experimentJobs: () => request<ExperimentJob[]>("/api/experiment-jobs"),
  experimentJob: (id: number) => request<ExperimentJob>(`/api/experiment-jobs/${id}`),
  cancelExperimentJob: (id: number) =>
    request<ExperimentJob>(`/api/experiment-jobs/${id}/cancel`, {
      method: "POST"
    }),
  clearFailedExperimentJobs: () =>
    request<{ deleted: number }>("/api/experiment-jobs/failed", {
      method: "DELETE"
    }),
  deleteExperimentJob: (id: number) =>
    request<{ deleted: boolean; id: number; status: string; imported_run_id?: number | null }>(`/api/experiment-jobs/${id}`, {
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
  llmQueryPlanBatches: (params: { run_id?: number; job_id?: number; problem_only?: boolean; limit?: number; include_details?: boolean }) => {
    const search = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) search.set(key, String(value));
    });
    return request<LlmQueryPlanBatch[]>(`/api/llm-query-plan-batches?${search}`);
  },
  heuristicPlanAnalysis: (params: { run_id: number; problem_only?: boolean; limit?: number }) => {
    const search = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) search.set(key, String(value));
    });
    return request<HeuristicPlanAnalysis>(`/api/heuristic-plan-analysis?${search}`);
  },
  deleteRun: (id: number) =>
    request<{ deleted: boolean; id: number; name: string }>(`/api/runs/${id}`, {
      method: "DELETE"
    }),
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
  estimateLlmUsage: (body: { input: string; model?: string; max_completion_tokens?: number | null; config?: Partial<ExperimentConfig> }) =>
    request<LlmUsageEstimate>("/api/llm/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  testLlm: (config: Partial<ExperimentConfig>) =>
    request<LlmTestResult>("/api/llm/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config })
    }),
  estimateExperiment: (config: Partial<ExperimentConfig>) =>
    request<LlmUsageEstimate>("/api/experiment-estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config })
    }),
  liveAttempt: (mentionId: number, body: { candidate_count: number; query_text?: string; human_guidance?: string; llm_config?: Partial<ExperimentConfig> }) =>
    request<LiveAttempt>(`/api/mentions/${mentionId}/live-attempt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    })
};
