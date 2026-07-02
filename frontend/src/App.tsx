import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Calculator,
  Check,
  Clipboard,
  Database,
  ListChecks,
  Play,
  RefreshCw,
  Search,
  Send,
  Settings2,
  Square,
  Trash2
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  api,
  Candidate,
  ConfigStatus,
  CoveragePoint,
  DatabaseSize,
  ExperimentConfig,
  ExperimentJob,
  Filters,
  GoldMetadataResult,
  HeuristicPlanAnalysis,
  ImprovementDiagnostics,
  LiveAttempt,
  LlmTestResult,
  LlmQueryPlanBatch,
  LlmUsageEstimate,
  MentionDetail,
  MentionRow,
  Run,
  SourceDiscoveryResult,
  SourceDataset
} from "./api";

const PAGE_SIZE = 100;
const MAX_RETRIEVAL_CANDIDATES = 1000;
const MAX_RETURNED_CANDIDATES = 1000;
const CANDIDATE_PAGE_SIZE = 100;
const DEFAULT_TABLE_SAMPLE_SIZE = 5;
const DEFAULT_RECORD_SAMPLE_SIZE = 10;
const LLM_SETTINGS_STORAGE_KEY = "coverage-dashboard-llm-settings";
const LLM_PROVIDER_KEYS_STORAGE_KEY = "coverage-dashboard-llm-provider-api-keys";
const LLM_SETTING_KEYS = [
  "llm_enabled",
  "llm_provider",
  "llm_provider_name",
  "llm_api_url",
  "llm_model",
  "llm_reasoning_effort",
  "llm_temperature",
  "llm_timeout_seconds",
  "llm_max_retries",
  "llm_site_url",
  "llm_app_name",
  "llm_max_tokens",
  "llm_parallel_requests",
  "use_heuristic_fallback_on_llm_failure",
  "openrouter_allow_fallbacks"
] as const;
type WorkspaceView = "overview" | "mentions" | "llm-plans" | "heuristic-plans" | "monitor" | "settings";
type InspectorTab = "overview" | "live" | "candidates" | "inspection" | "feedback";
type LlmSettingKey = (typeof LLM_SETTING_KEYS)[number];
type LlmSettings = Pick<ExperimentConfig, LlmSettingKey>;
type ProviderApiKeys = Record<string, string>;

function apiKeyScope(config: Pick<ExperimentConfig, "llm_provider">): string {
  const provider = String(config.llm_provider || "").trim().toLowerCase();
  if (provider === "openrouter" || provider === "cerebras") return provider;
  if (provider === "openai_compatible") return "openai_compatible";
  return provider || "llm";
}

function loadSavedProviderApiKeys(): ProviderApiKeys {
  try {
    const raw = window.localStorage.getItem(LLM_PROVIDER_KEYS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveProviderApiKey(scope: string, key: string) {
  const keys = loadSavedProviderApiKeys();
  if (key) keys[scope] = key;
  else delete keys[scope];
  window.localStorage.setItem(LLM_PROVIDER_KEYS_STORAGE_KEY, JSON.stringify(keys));
}

function applyProviderApiKey(config: ExperimentConfig, key: string): ExperimentConfig {
  const next = { ...config, llm_api_key: key, openrouter_api_key: "", cerebras_api_key: "" };
  if (next.llm_provider === "openrouter") next.openrouter_api_key = key;
  if (next.llm_provider === "cerebras") next.cerebras_api_key = key;
  return next;
}

function loadSavedLlmSettings(): Partial<LlmSettings> {
  try {
    const raw = window.localStorage.getItem(LLM_SETTINGS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function llmSettingsFromConfig(config: ExperimentConfig): LlmSettings {
  return {
    llm_enabled: config.llm_enabled,
    llm_provider: config.llm_provider,
    llm_provider_name: config.llm_provider_name,
    llm_api_url: config.llm_api_url,
    llm_model: config.llm_model,
    llm_reasoning_effort: config.llm_reasoning_effort,
    llm_temperature: config.llm_temperature,
    llm_timeout_seconds: config.llm_timeout_seconds,
    llm_max_retries: config.llm_max_retries,
    llm_site_url: config.llm_site_url,
    llm_app_name: config.llm_app_name,
    llm_max_tokens: config.llm_max_tokens,
    llm_parallel_requests: config.llm_parallel_requests,
    use_heuristic_fallback_on_llm_failure: config.use_heuristic_fallback_on_llm_failure,
    openrouter_allow_fallbacks: config.openrouter_allow_fallbacks
  };
}

function applyLlmSettings(config: ExperimentConfig, saved: Partial<LlmSettings>): ExperimentConfig {
  const next = { ...config, ...saved };
  next.use_openrouter = next.llm_enabled;
  next.use_heuristic_fallback_on_llm_failure = true;
  next.openrouter_parallel_requests = next.llm_parallel_requests;
  next.openrouter_model = next.llm_model;
  next.openrouter_provider = next.llm_provider_name;
  next.openrouter_allow_fallbacks = next.openrouter_allow_fallbacks ?? true;
  const savedKeys = loadSavedProviderApiKeys();
  const selectedKey = savedKeys[apiKeyScope(next)] || "";
  return applyProviderApiKey(next, selectedKey);
}

function providerDisplayName(config: ExperimentConfig | null): string {
  if (!config?.llm_enabled) return "Heuristic";
  if (config.llm_provider === "cerebras") return "Cerebras";
  if (config.llm_provider === "openrouter") {
    return config.llm_provider_name ? `OpenRouter via ${config.llm_provider_name}` : "OpenRouter";
  }
  if (config.llm_provider_name) return config.llm_provider_name;
  return config.llm_provider === "openai_compatible" ? "OpenAI-compatible" : config.llm_provider || "LLM";
}

function apiKeyFieldLabel(config: ExperimentConfig): string {
  if (config.llm_provider === "openrouter") return "OpenRouter API key";
  if (config.llm_provider === "cerebras") return "Cerebras API key";
  if (config.llm_provider === "openai_compatible") return "OpenAI-compatible API key";
  return "API key";
}

function pct(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function compactNumber(value: number | undefined | null): string {
  if (value === undefined || value === null) return "-";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value);
}

function compactUsd(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "-";
  if (value === 0) return "$0";
  if (value < 0.0001) return `$${value.toExponential(2)}`;
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 6 }).format(value);
}

function formatBytes(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const digits = unitIndex >= 3 ? 2 : unitIndex === 0 ? 0 : 1;
  return `${size.toFixed(digits)} ${units[unitIndex]}`;
}

function readableError(exc: unknown): string {
  const raw = String((exc as Error)?.message ?? exc);
  if (raw.includes("Cannot search on field [qid]")) {
    return "Alpaca GT lookup is using a stale backend that still searches qid. Restart the API so it uses _id.";
  }
  if (raw.includes("ALPACA_TOKEN")) {
    return "Alpaca token is not configured.";
  }
  try {
    const parsed = JSON.parse(raw);
    const detail = typeof parsed?.detail === "string" ? parsed.detail : parsed?.detail;
    if (detail) return readableError(new Error(typeof detail === "string" ? detail : JSON.stringify(detail)));
  } catch {
    // Keep the original message below.
  }
  return raw.length > 260 ? `${raw.slice(0, 260)}...` : raw;
}

function coverageBadge(covered?: boolean): string {
  return covered ? "covered" : "missed";
}

function stageLabel(stage: string): string {
  const labels: Record<string, string> = {
    queued: "Queued",
    starting: "Starting",
    loading: "Loading runner",
    sampling: "Sampling mentions",
    sampled: "Sample ready",
    query_plans: "LLM query plans",
    metadata: "GT metadata",
    alpaca_search: "Alpaca retrieval",
    writing: "Writing JSON",
    importing: "Importing",
    finalizing: "Finalizing",
    cancelling: "Cancelling",
    cancelled: "Cancelled",
    completed: "Completed",
    failed: "Failed"
  };
  return labels[stage] ?? stage.replace(/_/g, " ");
}

function jobProgress(job: ExperimentJob): number {
  if (job.status === "completed") return 100;
  if (job.status === "cancelled") return 100;
  if (!job.progress_total) return job.status === "running" ? 8 : 0;
  return Math.max(2, Math.min(100, Math.round((job.progress_current / job.progress_total) * 100)));
}

function isActiveJob(job: ExperimentJob): boolean {
  return job.status === "queued" || job.status === "running" || job.status === "cancel_requested";
}

function llmPlanStatsSummary(item: Run | ExperimentJob | null | undefined): string {
  const batches = item?.llm_query_plan_batch_count ?? 0;
  if (!batches) return "LLM plans: -";
  const failed = item?.llm_query_plan_failed_count ?? 0;
  const incomplete = item?.llm_query_plan_incomplete_batch_count ?? 0;
  const missing = item?.llm_query_plan_missing_task_count ?? 0;
  const requested = item?.llm_query_plan_requested_task_count ?? 0;
  const cost = item?.llm_query_plan_priced_batch_count ? ` · ${compactUsd(item.llm_query_plan_total_cost_usd)} LLM` : "";
  return `LLM plans ${compactNumber(failed)}/${compactNumber(batches)} failed · ${compactNumber(incomplete)} incomplete · ${compactNumber(missing)}/${compactNumber(requested)} missing tasks${cost}`;
}

function usageCostLabel(cost: LlmQueryPlanBatch["usage_cost"] | undefined | null): string {
  if (!cost) return "-";
  return compactUsd(cost.total_cost_usd);
}

function costSourceLabel(cost: LlmQueryPlanBatch["usage_cost"] | undefined | null): string {
  if (!cost?.cost_kind) return "No price";
  if (cost.cost_kind === "response_reported") return "Response reported";
  if (cost.cost_kind === "actual_tokens_catalog_pricing") return "Actual tokens, catalog price";
  if (cost.cost_kind === "actual_tokens_no_price") return "Actual tokens, price unavailable";
  if (cost.cost_kind === "actual_tokens_price_lookup_failed") return "Actual tokens, price lookup failed";
  return cost.cost_kind.replace(/_/g, " ");
}

function matchingSourceDatasets(config: ExperimentConfig, sourceDatasets: SourceDataset[]): SourceDataset[] {
  const targets = config.dataset_allowlist ?? [];
  const requested = config.requested_datasets ?? [];
  const ids = targets.length ? targets : requested;
  if (!sourceDatasets.length || !ids.length) return sourceDatasets;
  const allowed = new Set(ids);
  return sourceDatasets.filter((dataset) => allowed.has(dataset.dataset_id));
}

function estimateMentions(config: ExperimentConfig | null, sourceDatasets: SourceDataset[] = []): number | null {
  if (!config) return null;
  const targetedDatasets = config.dataset_allowlist?.length ?? 0;
  const requestedDatasets = config.requested_datasets?.length ?? 0;
  const matchingDatasets = matchingSourceDatasets(config, sourceDatasets);
  const availableDatasets = matchingDatasets.length || targetedDatasets || requestedDatasets || config.dataset_sample_size;
  const datasetCount = config.dataset_sample_size > 0 ? Math.min(config.dataset_sample_size, availableDatasets) : availableDatasets;
  if (matchingDatasets.length && config.tables_per_dataset <= 0 && config.records_per_table <= 0) {
    const selectedDatasets = matchingDatasets.slice(0, datasetCount || matchingDatasets.length);
    return selectedDatasets.reduce((total, dataset) => total + Number(dataset.mention_count || 0), 0);
  }
  return datasetCount * config.tables_per_dataset * config.records_per_table;
}

function isEntireTargetDatasetRun(config: ExperimentConfig | null): boolean {
  if (!config) return false;
  return (config.dataset_allowlist?.length ?? 0) === 1 && config.tables_per_dataset <= 0 && config.records_per_table <= 0;
}

function experimentDatasetLabel(config: ExperimentConfig): string {
  const targets = config.dataset_allowlist ?? [];
  if (targets.length === 1) return targets[0];
  if (targets.length > 1) return `${targets.length} targeted datasets`;
  return `${config.dataset_sample_size || "all"} random datasets`;
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "-";
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${rest.toString().padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${(minutes % 60).toString().padStart(2, "0")}m`;
}

function progressPercent(current: number, total: number): number {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((current / total) * 100)));
}

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [experimentConfig, setExperimentConfig] = useState<ExperimentConfig | null>(null);
  const [configStatus, setConfigStatus] = useState<ConfigStatus | null>(null);
  const [sourceDatasets, setSourceDatasets] = useState<SourceDataset[]>([]);
  const [databaseSize, setDatabaseSize] = useState<DatabaseSize | null>(null);
  const [jobs, setJobs] = useState<ExperimentJob[]>([]);
  const [activeJobId, setActiveJobId] = useState<number | null>(null);
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("overview");
  const [coverage, setCoverage] = useState<CoveragePoint[]>([]);
  const [filters, setFilters] = useState<Filters>({ datasets: [], retrieval_stages: [] });
  const [coveredFilter, setCoveredFilter] = useState("missed");
  const [datasetFilter, setDatasetFilter] = useState("");
  const [search, setSearch] = useState("");
  const [mentions, setMentions] = useState<MentionRow[]>([]);
  const [mentionTotal, setMentionTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [detail, setDetail] = useState<MentionDetail | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [jobBusy, setJobBusy] = useState(false);
  const [discoveryBusy, setDiscoveryBusy] = useState(false);
  const [discoveryForce, setDiscoveryForce] = useState(false);
  const [discoveryResult, setDiscoveryResult] = useState<SourceDiscoveryResult | null>(null);
  const [estimateBusy, setEstimateBusy] = useState(false);
  const [runEstimate, setRunEstimate] = useState<LlmUsageEstimate | null>(null);
  const [llmPlanBatches, setLlmPlanBatches] = useState<LlmQueryPlanBatch[]>([]);
  const [llmBatchProblemOnly, setLlmBatchProblemOnly] = useState(true);
  const [selectedLlmBatchId, setSelectedLlmBatchId] = useState<number | null>(null);
  const [heuristicPlanAnalysis, setHeuristicPlanAnalysis] = useState<HeuristicPlanAnalysis | null>(null);
  const [heuristicProblemOnly, setHeuristicProblemOnly] = useState(true);
  const [estimateError, setEstimateError] = useState("");
  const [settingsSaved, setSettingsSaved] = useState("");
  const [llmTestBusy, setLlmTestBusy] = useState(false);
  const [llmTestResult, setLlmTestResult] = useState<LlmTestResult | null>(null);
  const [llmTestError, setLlmTestError] = useState("");

  async function refreshRuns(nextRunId?: number) {
    const [runRows, jobRows, dbSize] = await Promise.all([api.runs(), api.experimentJobs(), api.databaseSize()]);
    setRuns(runRows);
    setJobs(jobRows);
    setDatabaseSize(dbSize);

    const nextId = nextRunId ?? selectedRunId ?? runRows[0]?.id ?? null;
    setSelectedRunId(nextId);
    if (nextId) {
      const run = runRows.find((item) => item.id === nextId) ?? (await api.run(nextId));
      setSelectedRun(run);
    } else {
      setSelectedRun(null);
      setLlmPlanBatches([]);
      setHeuristicPlanAnalysis(null);
    }
  }

  async function refreshRunData(runId: number) {
    const [run, coverageRows, filterRows, mentionRows, problemBatches, heuristicAnalysis] = await Promise.all([
      api.run(runId),
      api.coverage(runId),
      api.filters(runId),
      api.mentions(runId, {
        limit: PAGE_SIZE,
        offset,
        covered: coveredFilter,
        dataset_id: datasetFilter,
        search
      }),
      api.llmQueryPlanBatches({ run_id: runId, problem_only: llmBatchProblemOnly, limit: 100, include_details: true }),
      api.heuristicPlanAnalysis({ run_id: runId, problem_only: heuristicProblemOnly, limit: 200 })
    ]);
    setSelectedRun(run);
    setCoverage(coverageRows);
    setFilters(filterRows);
    setMentions(mentionRows.rows);
    setMentionTotal(mentionRows.total);
    setLlmPlanBatches(problemBatches);
    setHeuristicPlanAnalysis(heuristicAnalysis);
    if (problemBatches.length && !problemBatches.some((batch) => batch.id === selectedLlmBatchId)) {
      setSelectedLlmBatchId(problemBatches[0].id);
    }
    if (mentionRows.rows.length && !detail) {
      setDetail(await api.mention(mentionRows.rows[0].id));
    }
  }

  async function refreshRunSelection(runId: number) {
    await refreshRuns(runId);
    await refreshRunData(runId);
  }

  useEffect(() => {
    Promise.all([
      refreshRuns(),
      api.experimentDefaults().then((config) => setExperimentConfig(applyLlmSettings(config, loadSavedLlmSettings()))),
      api.configStatus().then(setConfigStatus),
      api.sourceDatasets().then(setSourceDatasets)
    ]).catch((exc) =>
      setError(String(exc.message ?? exc))
    );
  }, []);

  useEffect(() => {
    const hasActiveJob = jobs.some(isActiveJob);
    if (!hasActiveJob) return;
    const timer = window.setInterval(async () => {
      try {
        const jobRows = await api.experimentJobs();
        setJobs(jobRows);
        const activeImported = jobRows.find((job) => isActiveJob(job) && job.imported_run_id);
        if (activeImported?.imported_run_id) {
          await refreshRunSelection(activeImported.imported_run_id);
        }
        const completed = jobRows.find((job) => job.id === activeJobId && job.status === "completed" && job.imported_run_id);
        if (completed?.imported_run_id) {
          await refreshRunSelection(completed.imported_run_id);
        }
      } catch (exc) {
        setError(String((exc as Error).message ?? exc));
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [jobs, activeJobId]);

  useEffect(() => {
    if (!selectedRunId) return;
    refreshRunData(selectedRunId).catch((exc) => setError(String(exc.message ?? exc)));
  }, [selectedRunId, coveredFilter, datasetFilter, offset, llmBatchProblemOnly, heuristicProblemOnly]);

  const chartData = useMemo(
    () =>
      coverage.map((item) => ({
        ...item,
        coveragePercent: Number((item.coverage * 100).toFixed(2))
      })),
    [coverage]
  );

  const estimatedMentions = estimateMentions(experimentConfig, sourceDatasets);
  const entireTargetDatasetRun = isEntireTargetDatasetRun(experimentConfig);
  const selectedRunLlmBatchCount = selectedRun?.llm_query_plan_batch_count ?? 0;
  const selectedRunLlmFailureCount = selectedRun?.llm_query_plan_failed_count ?? 0;
  const selectedRunLlmIncompleteCount = selectedRun?.llm_query_plan_incomplete_batch_count ?? 0;
  const selectedRunLlmRequestedTasks = selectedRun?.llm_query_plan_requested_task_count ?? 0;
  const selectedRunLlmReturnedTasks = selectedRun?.llm_query_plan_returned_task_count ?? 0;
  const selectedRunLlmUsableTasks = selectedRun?.llm_query_plan_usable_task_count ?? 0;
  const selectedRunLlmMissingTasks = selectedRun?.llm_query_plan_missing_task_count ?? 0;
  const selectedRunLlmInputTokens = selectedRun?.llm_query_plan_prompt_tokens ?? 0;
  const selectedRunLlmOutputTokens = selectedRun?.llm_query_plan_completion_tokens ?? 0;
  const selectedRunLlmTokens = selectedRun?.llm_query_plan_total_tokens ?? 0;
  const selectedRunLlmCost = selectedRun?.llm_query_plan_total_cost_usd ?? null;
  const selectedRunLlmPricedBatches = selectedRun?.llm_query_plan_priced_batch_count ?? 0;
  const selectedRunLlmResponseCostBatches = selectedRun?.llm_query_plan_response_reported_cost_count ?? 0;
  const selectedRunLlmFailureLabel = selectedRunLlmBatchCount
    ? `${compactNumber(selectedRunLlmFailureCount)} / ${compactNumber(selectedRunLlmBatchCount)}`
    : "0";
  const selectedLlmBatch = llmPlanBatches.find((batch) => batch.id === selectedLlmBatchId) ?? llmPlanBatches[0] ?? null;
  const selectedProviderEnvKeyConfigured = experimentConfig?.llm_provider === "cerebras"
    ? Boolean(configStatus?.cerebras_configured)
    : experimentConfig?.llm_provider === "openrouter"
      ? Boolean(configStatus?.openrouter_configured)
      : Boolean(configStatus?.llm_configured);
  const selectedProviderKeyReady = Boolean(experimentConfig?.llm_api_key || selectedProviderEnvKeyConfigured);

  function updateExperimentConfig<K extends keyof ExperimentConfig>(key: K, value: ExperimentConfig[K]) {
    setRunEstimate(null);
    setEstimateError("");
    setLlmTestResult(null);
    setLlmTestError("");
    setExperimentConfig((current) => {
      if (!current) return current;
      const next = { ...current, [key]: value };
      if (key === "llm_api_key") {
        return applyProviderApiKey(next, String(value));
      }
      if (key === "llm_enabled") {
        next.use_openrouter = Boolean(value);
      }
      if (key === "llm_parallel_requests") {
        next.openrouter_parallel_requests = Number(value);
      }
      if (key === "llm_model") {
        next.openrouter_model = String(value);
      }
      if (key === "llm_provider_name") {
        next.openrouter_provider = String(value);
      }
      if (key === "llm_provider") {
        const provider = String(value);
        if (provider === "cerebras") {
          next.llm_enabled = true;
          next.use_openrouter = true;
          next.llm_provider_name = "Cerebras";
          next.llm_api_url = "https://api.cerebras.ai/v1/chat/completions";
          next.llm_model = "gpt-oss-120b";
          next.openrouter_model = next.llm_model;
          next.openrouter_provider = next.llm_provider_name;
        }
        if (provider === "openrouter") {
          next.llm_api_url = "https://openrouter.ai/api/v1/chat/completions";
          next.openrouter_model = next.llm_model;
          next.openrouter_provider = next.llm_provider_name;
        }
      }
      if (key === "llm_provider") {
        const savedKeys = loadSavedProviderApiKeys();
        return applyProviderApiKey(next, savedKeys[apiKeyScope(next)] || "");
      }
      return next;
    });
    if (LLM_SETTING_KEYS.includes(key as LlmSettingKey) || key === "llm_api_key") setSettingsSaved("");
  }

  function saveLlmSettings() {
    if (!experimentConfig) return;
    saveProviderApiKey(apiKeyScope(experimentConfig), experimentConfig.llm_api_key || "");
    window.localStorage.setItem(LLM_SETTINGS_STORAGE_KEY, JSON.stringify(llmSettingsFromConfig(experimentConfig)));
    setSettingsSaved("LLM settings saved for this browser");
  }

  async function estimateExperimentCost() {
    if (!experimentConfig) return;
    setEstimateBusy(true);
    setEstimateError("");
    setRunEstimate(null);
    try {
      setRunEstimate(await api.estimateExperiment(experimentConfig));
    } catch (exc) {
      setEstimateError(readableError(exc));
    } finally {
      setEstimateBusy(false);
    }
  }

  function useOpenRouterPreset() {
    updateExperimentConfig("llm_enabled", true);
    updateExperimentConfig("llm_provider", "openrouter");
    updateExperimentConfig("llm_provider_name", "");
    updateExperimentConfig("llm_api_url", "https://openrouter.ai/api/v1/chat/completions");
    updateExperimentConfig("llm_model", "openai/gpt-oss-120b");
    updateExperimentConfig("openrouter_allow_fallbacks", true);
  }

  function useCerebrasPreset() {
    updateExperimentConfig("llm_enabled", true);
    updateExperimentConfig("llm_provider", "cerebras");
    updateExperimentConfig("llm_provider_name", "Cerebras");
    updateExperimentConfig("llm_api_url", "https://api.cerebras.ai/v1/chat/completions");
    updateExperimentConfig("llm_model", "gpt-oss-120b");
  }

  function useOpenAiCompatiblePreset() {
    updateExperimentConfig("llm_enabled", true);
    updateExperimentConfig("llm_provider", "openai_compatible");
    updateExperimentConfig("llm_provider_name", "On premise");
  }

  async function testSelectedLlm() {
    if (!experimentConfig) return;
    setLlmTestBusy(true);
    setLlmTestError("");
    setLlmTestResult(null);
    try {
      setLlmTestResult(await api.testLlm(experimentConfig));
    } catch (exc) {
      setLlmTestError(readableError(exc));
    } finally {
      setLlmTestBusy(false);
    }
  }

  function updateTargetDataset(datasetId: string) {
    setRunEstimate(null);
    setEstimateError("");
    setExperimentConfig((current) => {
      if (!current) return current;
      if (!datasetId) {
        return {
          ...current,
          dataset_allowlist: [],
          tables_per_dataset: current.tables_per_dataset > 0 ? current.tables_per_dataset : DEFAULT_TABLE_SAMPLE_SIZE,
          records_per_table: current.records_per_table > 0 ? current.records_per_table : DEFAULT_RECORD_SAMPLE_SIZE
        };
      }
      return { ...current, dataset_allowlist: [datasetId], dataset_sample_size: 1 };
    });
  }

  function updateEntireDataset(enabled: boolean) {
    setRunEstimate(null);
    setEstimateError("");
    setExperimentConfig((current) => {
      if (!current) return current;
      if (enabled) {
        return { ...current, tables_per_dataset: 0, records_per_table: 0 };
      }
      return {
        ...current,
        tables_per_dataset: current.tables_per_dataset > 0 ? current.tables_per_dataset : DEFAULT_TABLE_SAMPLE_SIZE,
        records_per_table: current.records_per_table > 0 ? current.records_per_table : DEFAULT_RECORD_SAMPLE_SIZE
      };
    });
  }

  async function startExperiment() {
    if (!experimentConfig) return;
    if (isEntireTargetDatasetRun(experimentConfig)) {
      const datasetId = experimentConfig.dataset_allowlist[0];
      const mentionCount = estimateMentions(experimentConfig, sourceDatasets) ?? 0;
      const candidateSlots = mentionCount * experimentConfig.max_candidates;
      const confirmed = window.confirm(
        `Run the entire ${datasetId} dataset?\n\nThis will process ${compactNumber(mentionCount)} mentions with a retrieval window of ${compactNumber(
          experimentConfig.max_candidates
        )}, for up to ${compactNumber(candidateSlots)} candidate checks before filtering/import limits.\n\nContinue?`
      );
      if (!confirmed) return;
    }
    setJobBusy(true);
    setStatus("Starting background experiment");
    setError("");
    try {
      const job = await api.startExperimentJob(experimentConfig);
      setActiveJobId(job.id);
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      setWorkspaceView("monitor");
      setStatus(`Experiment job ${job.id} queued`);
    } catch (exc) {
      setError(String((exc as Error).message ?? exc));
    } finally {
      setJobBusy(false);
    }
  }

  async function clearFailedJobs() {
    setError("");
    try {
      const result = await api.clearFailedExperimentJobs();
      setJobs(await api.experimentJobs());
      setStatus(`Cleared ${result.deleted} stopped job${result.deleted === 1 ? "" : "s"}`);
    } catch (exc) {
      setError(String((exc as Error).message ?? exc));
    }
  }

  async function discoverSourceData() {
    setDiscoveryBusy(true);
    setDiscoveryResult(null);
    setError("");
    try {
      const result = await api.discoverSourceDatasets({ force: discoveryForce });
      setDiscoveryResult(result);
      setSourceDatasets(result.inventory);
      setDatabaseSize(await api.databaseSize());
      const importedMentions = result.imported.reduce((total, dataset) => total + Number(dataset.mention_count || 0), 0);
      setStatus(
        result.seeded
          ? `Discovered ${result.imported.length} dataset${result.imported.length === 1 ? "" : "s"} with ${compactNumber(importedMentions)} mentions`
          : result.reason || "Source metadata is already current"
      );
    } catch (exc) {
      setError(readableError(exc));
    } finally {
      setDiscoveryBusy(false);
    }
  }

  async function cancelJob(jobId: number) {
    setError("");
    try {
      const job = await api.cancelExperimentJob(jobId);
      setJobs((current) => current.map((item) => (item.id === job.id ? job : item)));
      setStatus(`Cancellation requested for job ${jobId}`);
    } catch (exc) {
      setError(readableError(exc));
    }
  }

  async function deleteJob(job: ExperimentJob) {
    const confirmed = window.confirm(
      `Delete job ${job.id}?\n\nThis removes the job card and any prompt traces that are not attached to an imported run. Imported runs are kept.`
    );
    if (!confirmed) return;
    setError("");
    try {
      await api.deleteExperimentJob(job.id);
      setJobs(await api.experimentJobs());
      setDatabaseSize(await api.databaseSize());
      setStatus(`Deleted job ${job.id}`);
    } catch (exc) {
      setError(readableError(exc));
    }
  }

  async function deleteSelectedRun() {
    if (!selectedRunId || !selectedRun) return;
    const confirmed = window.confirm(
      `Delete run "${selectedRun.name}"?\n\nThis removes the run, mentions, retrieval notes, and live attempts stored under it. Candidate cache entries are shared and will remain available for reuse.`
    );
    if (!confirmed) return;
    setError("");
    try {
      const deleted = await api.deleteRun(selectedRunId);
      setDetail(null);
      setMentions([]);
      setMentionTotal(0);
      setOffset(0);
      setStatus(`Deleted run ${deleted.name}`);
      const nextRuns = await api.runs();
      setRuns(nextRuns);
      const nextRunId = nextRuns[0]?.id ?? null;
      setSelectedRunId(nextRunId);
      setSelectedRun(nextRuns[0] ?? null);
      setJobs(await api.experimentJobs());
      setDatabaseSize(await api.databaseSize());
    } catch (exc) {
      setError(readableError(exc));
    }
  }

  async function selectMention(row: MentionRow) {
    setError("");
    setDetail(await api.mention(row.id));
  }

  async function applySearch() {
    if (!selectedRunId) return;
    setOffset(0);
    const mentionRows = await api.mentions(selectedRunId, {
      limit: PAGE_SIZE,
      offset: 0,
      covered: coveredFilter,
      dataset_id: datasetFilter,
      search
    });
    setMentions(mentionRows.rows);
    setMentionTotal(mentionRows.total);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <h1>Coverage Dashboard</h1>
          <div className="muted">Postgres-backed retrieval analysis</div>
        </div>

        <section className="panel run-panel">
          <div className="panel-title">
            <Play size={17} />
            Run Experiment
          </div>
          {experimentConfig && (
            <>
              <div className="run-summary">
                <span>{experimentDatasetLabel(experimentConfig)}</span>
                <strong>
                  {compactNumber(estimatedMentions)} mentions x {compactNumber(experimentConfig.max_candidates)} retrieval window
                </strong>
              </div>
              <div className="config-status-grid">
                <span className={`config-chip ${configStatus?.alpaca_configured ? "ok" : "warn"}`}>
                  Alpaca {configStatus?.alpaca_configured ? "ready" : "token missing"}
                </span>
                <span className={`config-chip ${selectedProviderKeyReady ? "ok" : "warn"}`}>
                  {providerDisplayName(experimentConfig)} {selectedProviderKeyReady ? "ready" : "key missing"}
                </span>
              </div>
              <label>
                Run name
                <input
                  value={experimentConfig.name ?? ""}
                  placeholder="webapp_random_sampler"
                  onChange={(event) => updateExperimentConfig("name", event.target.value)}
                />
              </label>
              <label>
                Dataset target
                <select value={experimentConfig.dataset_allowlist?.[0] ?? ""} onChange={(event) => updateTargetDataset(event.target.value)}>
                  <option value="">Random seeded datasets</option>
                  {sourceDatasets.map((dataset) => (
                    <option key={dataset.dataset_id} value={dataset.dataset_id}>
                      {dataset.dataset_id} ({compactNumber(dataset.mention_count)} mentions)
                    </option>
                  ))}
                </select>
              </label>
              <div className="source-discovery">
                <button type="button" onClick={discoverSourceData} disabled={discoveryBusy}>
                  <Database size={15} />
                  {discoveryBusy ? "Discovering..." : "Discover data"}
                </button>
                <label className="checkbox-row compact-checkbox">
                  <input
                    type="checkbox"
                    checked={discoveryForce}
                    onChange={(event) => setDiscoveryForce(event.target.checked)}
                  />
                  Force refresh
                </label>
              </div>
              {discoveryResult?.warnings.length ? (
                <details className="discovery-warnings">
                  <summary>{discoveryResult.warnings.length} discovery warning{discoveryResult.warnings.length === 1 ? "" : "s"}</summary>
                  <pre>{discoveryResult.warnings.slice(0, 12).join("\n")}</pre>
                </details>
              ) : null}
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  disabled={!experimentConfig.dataset_allowlist?.length}
                  checked={entireTargetDatasetRun}
                  onChange={(event) => updateEntireDataset(event.target.checked)}
                />
                Entire dataset
              </label>
              <div className="config-grid">
                <label>
                  Seed
                  <input
                    type="number"
                    value={experimentConfig.random_seed}
                    onChange={(event) => updateExperimentConfig("random_seed", Number(event.target.value))}
                  />
                </label>
                <label>
                  Retrieval window
                  <input
                    type="number"
                    min={1}
                    max={MAX_RETRIEVAL_CANDIDATES}
                    value={experimentConfig.max_candidates}
                    onChange={(event) => {
                      const value = Math.min(MAX_RETRIEVAL_CANDIDATES, Math.max(1, Number(event.target.value)));
                      updateExperimentConfig("max_candidates", value);
                      updateExperimentConfig("dashboard_candidate_limit", Math.min(value, MAX_RETURNED_CANDIDATES));
                    }}
                  />
                </label>
                <label>
                  Datasets
                  <input
                    type="number"
                    min={0}
                    max={Math.max(sourceDatasets.length, experimentConfig.requested_datasets.length, 1)}
                    value={experimentConfig.dataset_sample_size}
                    onChange={(event) => updateExperimentConfig("dataset_sample_size", Number(event.target.value))}
                  />
                </label>
                <label>
                  Tables
                  <input
                    type="number"
                    min={0}
                    disabled={entireTargetDatasetRun}
                    value={experimentConfig.tables_per_dataset}
                    onChange={(event) => updateExperimentConfig("tables_per_dataset", Number(event.target.value))}
                  />
                </label>
                <label>
                  Records
                  <input
                    type="number"
                    min={0}
                    disabled={entireTargetDatasetRun}
                    value={experimentConfig.records_per_table}
                    onChange={(event) => updateExperimentConfig("records_per_table", Number(event.target.value))}
                  />
                </label>
                <label>
                  Workers
                  <input
                    type="number"
                    min={1}
                    max={32}
                    value={experimentConfig.max_workers}
                    onChange={(event) => updateExperimentConfig("max_workers", Number(event.target.value))}
                  />
                </label>
              </div>
              <details className="advanced-config">
                <summary>
                  <Settings2 size={15} />
                  Advanced run params
                </summary>
                <div className="config-grid">
                  <label>
                    Context rows
                    <input
                      type="number"
                      min={0}
                      value={experimentConfig.context_rows}
                      onChange={(event) => updateExperimentConfig("context_rows", Number(event.target.value))}
                    />
                  </label>
                  <label>
                    LLM batch
                    <input
                      type="number"
                      min={1}
                      value={experimentConfig.max_tasks_per_llm_request}
                      onChange={(event) => updateExperimentConfig("max_tasks_per_llm_request", Number(event.target.value))}
                    />
                  </label>
                  <label>
                    LLM parallel
                    <input
                      type="number"
                      min={1}
                      max={16}
                      value={experimentConfig.llm_parallel_requests}
                      onChange={(event) => updateExperimentConfig("llm_parallel_requests", Number(event.target.value))}
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={experimentConfig.llm_enabled}
                      onChange={(event) => updateExperimentConfig("llm_enabled", event.target.checked)}
                    />
                    LLM query planner
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={experimentConfig.enable_recall_query_expansion}
                      onChange={(event) => updateExperimentConfig("enable_recall_query_expansion", event.target.checked)}
                    />
                    Recall expansion
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={experimentConfig.enable_llm_url_hints}
                      onChange={(event) => updateExperimentConfig("enable_llm_url_hints", event.target.checked)}
                    />
                    URL hints
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={experimentConfig.save_full_debug_output}
                      onChange={(event) => updateExperimentConfig("save_full_debug_output", event.target.checked)}
                    />
                    Full debug JSON
                  </label>
                </div>
              </details>
              <div className="run-estimate">
                <button type="button" onClick={estimateExperimentCost} disabled={estimateBusy || !experimentConfig.llm_enabled}>
                  <Calculator size={15} />
                  {estimateBusy ? "Estimating..." : "Estimate LLM cost"}
                </button>
                {estimateError && <div className="error-box compact">{estimateError}</div>}
                {runEstimate && (
                  <div className="metadata-grid estimate-grid">
                    <div className="metadata-chip">
                      <span>Mentions</span>
                      <strong>{compactNumber(runEstimate.target?.sampled_mentions)}</strong>
                    </div>
                    <div className="metadata-chip">
                      <span>LLM requests</span>
                      <strong>{compactNumber(runEstimate.target?.llm_request_count)}</strong>
                    </div>
                    <div className="metadata-chip">
                      <span>Input tokens</span>
                      <strong>{compactNumber(runEstimate.token_estimate.prompt_tokens)}</strong>
                    </div>
                    <div className="metadata-chip">
                      <span>Output est.</span>
                      <strong>{compactNumber(runEstimate.token_estimate.estimated_completion_tokens ?? runEstimate.token_estimate.max_completion_tokens)}</strong>
                    </div>
                    <div className="metadata-chip">
                      <span>Total tokens</span>
                      <strong>{compactNumber(runEstimate.token_estimate.total_tokens)}</strong>
                    </div>
                    <div className="metadata-chip">
                      <span>Est. cost</span>
                      <strong>{compactUsd(runEstimate.pricing?.estimated_total_cost_usd)}</strong>
                    </div>
                  </div>
                )}
              </div>
              <button className="primary" disabled={jobBusy} onClick={startExperiment}>
                <Play size={16} />
                {jobBusy ? "Starting..." : "Start background run"}
              </button>
            </>
          )}
          <JobList
            jobs={jobs}
            sourceDatasets={sourceDatasets}
            onOpenRun={(runId) => runId && refreshRunSelection(runId)}
            onClearFailed={clearFailedJobs}
            onCancelJob={cancelJob}
            onDeleteJob={deleteJob}
          />
        </section>

        <section className="panel">
          <div className="panel-title">
            <Database size={17} />
            Data
          </div>
          <label>
            Run
            <select
              value={selectedRunId ?? ""}
              onChange={(event) => {
                setDetail(null);
                setOffset(0);
                setSelectedRunId(Number(event.target.value));
              }}
            >
              {runs.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.name}
                </option>
              ))}
            </select>
          </label>
          <button onClick={() => (selectedRunId ? refreshRunSelection(selectedRunId) : refreshRuns())}>
            <RefreshCw size={16} />
            Refresh
          </button>
          <button onClick={deleteSelectedRun} disabled={!selectedRunId}>
            <Trash2 size={16} />
            Delete run
          </button>
          <a className="link-button" href="http://localhost:8082" target="_blank" rel="noreferrer">
            <Database size={16} />
            SQL data
          </a>
          {databaseSize && (
            <div className="db-size">
              <div className="db-size-total">
                <span>{databaseSize.database_name}</span>
                <strong>{formatBytes(databaseSize.total_bytes)}</strong>
              </div>
              <div className="db-size-table-list">
                {databaseSize.tables.slice(0, 5).map((table) => (
                  <div className="db-size-row" key={`${table.schema_name}.${table.table_name}`}>
                    <span>{table.table_name}</span>
                    <strong>{formatBytes(table.total_bytes)}</strong>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {status && <div className="status">{status}</div>}
        {error && (
          <div className="error-box">
            <AlertCircle size={16} />
            {error}
          </div>
        )}
      </aside>

      <main className="workspace">
        <header className="topline">
          <div>
            <div className="eyebrow">{selectedRun?.source_filename ?? "No run selected"}</div>
            <h2>{selectedRun?.name ?? "Start an experiment"}</h2>
          </div>
          <div className="stat-strip">
            <Metric label="Tables" value={compactNumber(selectedRun?.table_count)} />
            <Metric label="Mentions" value={compactNumber(selectedRun?.mention_count)} />
            <Metric label="Candidates" value={compactNumber(selectedRun?.candidate_count)} />
            <Metric label="Any-Hit Coverage" value={pct(selectedRun?.imported_coverage)} />
            <Metric label="LLM Plan Failures" value={selectedRunLlmFailureLabel} />
          </div>
        </header>

        <nav className="workspace-tabs">
          <button className={workspaceView === "overview" ? "active" : ""} onClick={() => setWorkspaceView("overview")}>
            <BarChart3 size={16} />
            Overview
          </button>
          <button className={workspaceView === "mentions" ? "active" : ""} onClick={() => setWorkspaceView("mentions")}>
            <ListChecks size={16} />
            Mentions
          </button>
          <button className={workspaceView === "llm-plans" ? "active" : ""} onClick={() => setWorkspaceView("llm-plans")}>
            <Clipboard size={16} />
            LLM Plans
          </button>
          <button className={workspaceView === "heuristic-plans" ? "active" : ""} onClick={() => setWorkspaceView("heuristic-plans")}>
            <AlertCircle size={16} />
            Heuristic Plans
          </button>
          <button className={workspaceView === "monitor" ? "active" : ""} onClick={() => setWorkspaceView("monitor")}>
            <Activity size={16} />
            Run Monitor
          </button>
          <button className={workspaceView === "settings" ? "active" : ""} onClick={() => setWorkspaceView("settings")}>
            <Settings2 size={16} />
            Settings
          </button>
        </nav>

        {workspaceView === "overview" && (
          <section className="analysis-grid">
            <div className="panel wide">
              <div className="panel-title">
                <BarChart3 size={17} />
                Coverage@K
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ left: 0, right: 16, top: 10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d7dee8" />
                    <XAxis dataKey="k" tickLine={false} axisLine={false} />
                    <YAxis
                      tickLine={false}
                      axisLine={false}
                      domain={[0, 100]}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip formatter={(value) => [`${Number(value).toFixed(1)}%`, "Coverage@K"]} />
                    <Line
                      type="monotone"
                      dataKey="coveragePercent"
                      stroke="#0f766e"
                      strokeWidth={2.5}
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel">
              <div className="panel-title">Retrieval Stages</div>
              <div className="stage-list">
                {filters.retrieval_stages.slice(0, 10).map((stage) => (
                  <div className="stage-row" key={stage.retrieval_stage}>
                    <span>{stage.retrieval_stage}</span>
                    <strong>{compactNumber(stage.candidate_count)}</strong>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {workspaceView === "llm-plans" && (
          <section className="llm-plan-grid">
            <div className="panel llm-plan-summary">
              <div className="panel-title">
                <Clipboard size={17} />
                LLM Query Plans
              </div>
              <div className="llm-plan-metrics">
                <Metric label="Batches" value={compactNumber(selectedRunLlmBatchCount)} />
                <Metric label="Failed" value={`${compactNumber(selectedRunLlmFailureCount)} / ${compactNumber(selectedRunLlmBatchCount)}`} />
                <Metric label="Incomplete" value={compactNumber(selectedRunLlmIncompleteCount)} />
                <Metric label="Missing Tasks" value={`${compactNumber(selectedRunLlmMissingTasks)} / ${compactNumber(selectedRunLlmRequestedTasks)}`} />
                <Metric label="Returned Tasks" value={compactNumber(selectedRunLlmReturnedTasks)} />
                <Metric label="Usable Tasks" value={compactNumber(selectedRunLlmUsableTasks)} />
                <Metric label="Input Tokens" value={compactNumber(selectedRunLlmInputTokens)} />
                <Metric label="Output Tokens" value={compactNumber(selectedRunLlmOutputTokens)} />
                <Metric label="Total Tokens" value={compactNumber(selectedRunLlmTokens)} />
                <Metric
                  label="LLM Cost"
                  value={selectedRunLlmPricedBatches ? compactUsd(selectedRunLlmCost) : "-"}
                />
              </div>
              <div className="llm-cost-note">
                {selectedRunLlmPricedBatches
                  ? `${compactNumber(selectedRunLlmPricedBatches)} priced batches · ${compactNumber(selectedRunLlmResponseCostBatches)} response-reported costs`
                  : "Cost appears only when the LLM service reports cost or a supported pricing lookup can price actual response tokens."}
              </div>
            </div>

            <div className="panel llm-plan-toolbar">
              <div className="table-tools">
                <div className="panel-title">Batch Trace</div>
                <label className="inline-toggle">
                  <input
                    type="checkbox"
                    checked={llmBatchProblemOnly}
                    onChange={(event) => setLlmBatchProblemOnly(event.target.checked)}
                  />
                  Problem batches only
                </label>
                <button onClick={() => selectedRunId && refreshRunData(selectedRunId)}>
                  <RefreshCw size={15} />
                  Refresh
                </button>
              </div>
            </div>

            <div className="llm-plan-workbench">
              <div className="panel llm-batch-list-panel">
                <div className="llm-batch-list">
                  {!llmPlanBatches.length && (
                    <div className="muted metadata-empty">
                      {selectedRun ? "No LLM query-plan batches match this filter." : "Select a run to inspect LLM query-plan traces."}
                    </div>
                  )}
                  {llmPlanBatches.map((batch) => {
                    const hasProblem = Boolean(batch.error || batch.parse_warning || batch.response_parse_error || (batch.missing_task_count ?? 0) > 0);
                    const isSelected = selectedLlmBatch?.id === batch.id;
                    return (
                      <button
                        className={`llm-batch-card ${hasProblem ? "problem" : "clean"} ${isSelected ? "selected" : ""}`}
                        key={batch.id}
                        onClick={() => setSelectedLlmBatchId(batch.id)}
                      >
                        <span>
                          <strong>Batch {batch.id}</strong>
                          <em>{batch.provider || "LLM"} · {batch.model || "model unknown"}</em>
                        </span>
                        <span className="llm-batch-card-counts">
                          <b>{compactNumber(batch.usable_task_count)} usable</b>
                          <b>{compactNumber(batch.missing_task_count)} missing</b>
                          <b>{compactNumber(batch.unknown_returned_task_count)} unknown IDs</b>
                          <b>{usageCostLabel(batch.usage_cost)}</b>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="panel llm-batch-inspector">
                {selectedLlmBatch ? (
                  <LlmBatchInspector batch={selectedLlmBatch} />
                ) : (
                  <div className="muted metadata-empty">Select a batch to inspect the LLM answer and task matching.</div>
                )}
              </div>
            </div>
          </section>
        )}

        {workspaceView === "heuristic-plans" && (
          <HeuristicPlanPanel
            analysis={heuristicPlanAnalysis}
            problemOnly={heuristicProblemOnly}
            selectedRun={selectedRun}
            onProblemOnlyChange={setHeuristicProblemOnly}
            onRefresh={() => selectedRunId && refreshRunData(selectedRunId)}
            onOpenMention={async (mentionId) => {
              setWorkspaceView("mentions");
              const nextDetail = await api.mention(mentionId);
              setDetail(nextDetail);
            }}
          />
        )}

        {workspaceView === "monitor" && (
          <RunMonitor
            jobs={jobs}
            sourceDatasets={sourceDatasets}
            onOpenRun={(runId) => runId && refreshRunSelection(runId)}
            onCancelJob={cancelJob}
            onDeleteJob={deleteJob}
          />
        )}

        {workspaceView === "settings" && experimentConfig && (
          <SettingsPanel
            config={experimentConfig}
            savedMessage={settingsSaved}
            onChange={updateExperimentConfig}
            onSave={saveLlmSettings}
            onOpenRouterPreset={useOpenRouterPreset}
            onCerebrasPreset={useCerebrasPreset}
            onOpenAiCompatiblePreset={useOpenAiCompatiblePreset}
            onTestLlm={testSelectedLlm}
            testBusy={llmTestBusy}
            testResult={llmTestResult}
            testError={llmTestError}
            envKeyConfigured={selectedProviderEnvKeyConfigured}
          />
        )}

        {workspaceView === "mentions" && (
          <section className="split">
            <div className="panel list-panel">
              <div className="table-tools">
                <div className="panel-title">Mentions</div>
                <select value={coveredFilter} onChange={(event) => setCoveredFilter(event.target.value)}>
                  <option value="missed">Missed</option>
                  <option value="covered">Covered</option>
                  <option value="all">All</option>
                </select>
                <select value={datasetFilter} onChange={(event) => setDatasetFilter(event.target.value)}>
                  <option value="">All datasets</option>
                  {filters.datasets.map((dataset) => (
                    <option key={dataset.dataset_id} value={dataset.dataset_id}>
                      {dataset.dataset_id}
                    </option>
                  ))}
                </select>
                <div className="search-box">
                  <Search size={15} />
                  <input value={search} onChange={(event) => setSearch(event.target.value)} onKeyDown={(event) => {
                    if (event.key === "Enter") applySearch();
                  }} />
                </div>
                <button onClick={applySearch}>
                  <Search size={15} />
                  Search
                </button>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Mention</th>
                      <th>Dataset</th>
                      <th>Table</th>
                      <th>Best Rank</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mentions.map((row) => (
                      <tr
                        key={row.id}
                        className={detail?.mention.id === row.id ? "selected-row" : ""}
                        onClick={() => selectMention(row)}
                      >
                        <td>
                          <strong>{row.mention ?? row.lookup_text ?? "-"}</strong>
                          <span>{row.primary_gt_qid ?? ""}</span>
                        </td>
                        <td>{row.dataset_id}</td>
                        <td>{row.table_id}</td>
                        <td>{row.imported_best_rank ?? "-"}</td>
                        <td>
                          <span className={`pill ${row.covered_by_imported_candidates ? "ok" : "miss"}`}>
                            {coverageBadge(row.covered_by_imported_candidates)}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="pager">
                <span>
                  {compactNumber(Math.min(offset + 1, mentionTotal))}-{compactNumber(Math.min(offset + PAGE_SIZE, mentionTotal))} / {compactNumber(mentionTotal)}
                </span>
                <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
                  Previous
                </button>
                <button disabled={offset + PAGE_SIZE >= mentionTotal} onClick={() => setOffset(offset + PAGE_SIZE)}>
                  Next
                </button>
              </div>
            </div>

            <MentionInspector
              detail={detail}
              llmConfig={experimentConfig ? llmSettingsFromConfig(experimentConfig) : undefined}
              onRefresh={async () => {
                if (detail) {
                  setDetail(await api.mention(detail.mention.id));
                }
              }}
            />
          </section>
        )}
      </main>
    </div>
  );
}

function SettingsPanel({
  config,
  savedMessage,
  onChange,
  onSave,
  onOpenRouterPreset,
  onCerebrasPreset,
  onOpenAiCompatiblePreset,
  onTestLlm,
  testBusy,
  testResult,
  testError,
  envKeyConfigured
}: {
  config: ExperimentConfig;
  savedMessage: string;
  onChange: <K extends keyof ExperimentConfig>(key: K, value: ExperimentConfig[K]) => void;
  onSave: () => void;
  onOpenRouterPreset: () => void;
  onCerebrasPreset: () => void;
  onOpenAiCompatiblePreset: () => void;
  onTestLlm: () => void;
  testBusy: boolean;
  testResult: LlmTestResult | null;
  testError: string;
  envKeyConfigured: boolean;
}) {
  const keyReady = Boolean(config.llm_api_key || envKeyConfigured);
  return (
    <section className="settings-grid">
      <div className="panel settings-main">
        <div className="panel-title">
          <Settings2 size={17} />
          LLM Provider
        </div>
        <div className="settings-actions">
          <button type="button" onClick={onOpenRouterPreset}>OpenRouter preset</button>
          <button type="button" onClick={onCerebrasPreset}>Cerebras preset</button>
          <button type="button" onClick={onOpenAiCompatiblePreset}>On-prem preset</button>
        </div>
        <div className="settings-form">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={config.llm_enabled}
              onChange={(event) => onChange("llm_enabled", event.target.checked)}
            />
            Use LLM query planner
          </label>
          <label>
            Provider type
            <select value={config.llm_provider} onChange={(event) => onChange("llm_provider", event.target.value)}>
              <option value="openrouter">OpenRouter</option>
              <option value="cerebras">Cerebras</option>
              <option value="openai_compatible">OpenAI-compatible</option>
              <option value="none">Heuristic only</option>
            </select>
          </label>
          <label>
            {config.llm_provider === "openrouter" ? "OpenRouter route provider" : "Provider label"}
            <input
              value={config.llm_provider_name}
              placeholder={config.llm_provider === "openrouter" ? "Optional: cerebras, fireworks, ..." : config.llm_provider === "cerebras" ? "Cerebras" : "On premise"}
              onChange={(event) => onChange("llm_provider_name", event.target.value)}
            />
          </label>
          {config.llm_provider === "openrouter" && (
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={config.openrouter_allow_fallbacks}
                onChange={(event) => onChange("openrouter_allow_fallbacks", event.target.checked)}
              />
              Allow OpenRouter provider fallbacks
            </label>
          )}
          <label>
            Chat completions URL
            <input
              value={config.llm_api_url}
              placeholder="http://localhost:8000/v1/chat/completions"
              onChange={(event) => onChange("llm_api_url", event.target.value)}
            />
          </label>
          <label>
            {apiKeyFieldLabel(config)}
            <input
              type="password"
              value={config.llm_api_key}
              placeholder={`Paste ${apiKeyFieldLabel(config)}`}
              onChange={(event) => onChange("llm_api_key", event.target.value)}
            />
          </label>
          <label>
            Model
            <input value={config.llm_model} placeholder="local-model-name" onChange={(event) => onChange("llm_model", event.target.value)} />
          </label>
          <div className="config-grid">
            <label>
              Parallel requests
              <input
                type="number"
                min={1}
                max={16}
                value={config.llm_parallel_requests}
                onChange={(event) => onChange("llm_parallel_requests", Number(event.target.value))}
              />
            </label>
            <label>
              Temperature
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={config.llm_temperature}
                onChange={(event) => onChange("llm_temperature", Number(event.target.value))}
              />
            </label>
            <label>
              Timeout seconds
              <input
                type="number"
                min={1}
                value={config.llm_timeout_seconds}
                onChange={(event) => onChange("llm_timeout_seconds", Number(event.target.value))}
              />
            </label>
            <label>
              Max retries
              <input
                type="number"
                min={1}
                value={config.llm_max_retries}
                onChange={(event) => onChange("llm_max_retries", Number(event.target.value))}
              />
            </label>
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={config.use_heuristic_fallback_on_llm_failure}
              disabled
              readOnly
            />
            Trace failed LLM plans and use heuristic fallback for unresolved tasks
          </label>
          <div className="settings-actions">
            <button type="button" onClick={onTestLlm} disabled={testBusy || !config.llm_enabled}>
              <Send size={16} />
              {testBusy ? "Testing..." : "Test selected LLM"}
            </button>
            <button className="primary" type="button" onClick={onSave}>
              <Check size={16} />
              Save settings
            </button>
            {savedMessage && <span className="settings-saved">{savedMessage}</span>}
          </div>
          {testResult && (
            <div className={`settings-test ${testResult.ok ? "ok" : "warn"}`}>
              {testResult.ok ? "LLM call succeeded" : "LLM call returned an empty answer"} · {testResult.response_model || testResult.model || "model unknown"}
            </div>
          )}
          {testError && <div className="settings-test warn">{testError}</div>}
        </div>
      </div>

      <aside className="panel settings-side">
        <div className="panel-title">Active Provider</div>
        <div className="metadata-grid">
          <div className="metadata-chip">
            <span>Status</span>
            <strong>{config.llm_enabled ? (keyReady ? "Ready" : "Key missing") : "Heuristic only"}</strong>
          </div>
          <div className="metadata-chip">
            <span>Provider</span>
            <strong>{providerDisplayName(config)}</strong>
          </div>
          <div className="metadata-chip">
            <span>Model</span>
            <strong>{config.llm_model || "-"}</strong>
          </div>
          <div className="metadata-chip">
            <span>URL</span>
            <strong title={config.llm_api_url}>{config.llm_api_url || "-"}</strong>
          </div>
        </div>
        <div className={`note ${keyReady ? "success-note" : ""}`}>
          <strong>{keyReady ? "API key available" : "API key needed"}</strong>
          <span>
            {config.llm_api_key
              ? `Using the ${apiKeyFieldLabel(config)} saved in this browser.`
              : envKeyConfigured
                ? `Using the ${apiKeyFieldLabel(config)} from the environment.`
                : `Paste a ${apiKeyFieldLabel(config)} above and save it in this browser.`}
          </span>
        </div>
      </aside>
    </section>
  );
}

function JobList({
  jobs,
  sourceDatasets,
  onOpenRun,
  onClearFailed,
  onCancelJob,
  onDeleteJob
}: {
  jobs: ExperimentJob[];
  sourceDatasets: SourceDataset[];
  onOpenRun: (runId?: number) => void;
  onClearFailed: () => void;
  onCancelJob: (jobId: number) => void;
  onDeleteJob: (job: ExperimentJob) => void;
}) {
  if (!jobs.length) {
    return <div className="job-empty">No web runs yet</div>;
  }
  return (
    <div className="job-list">
      <div className="job-list-head">
        <span>Recent web runs</span>
        {jobs.some((job) => job.status === "failed" || job.status === "cancelled") ? (
          <button className="text-button" onClick={onClearFailed}>
            Clear stopped
          </button>
        ) : (
          <span>{jobs.length}</span>
        )}
      </div>
      {jobs.slice(0, 4).map((job) => {
        const total = job.progress_total || 0;
        const current = job.progress_current || 0;
        const progress = jobProgress(job);
        const sampleCount = estimateMentions(job.config, sourceDatasets);
        return (
          <div className={`job-card ${job.status}`} key={job.id}>
            <div className="job-row">
              <strong>Job {job.id}</strong>
              <span className={`job-status ${job.status}`}>{job.status}</span>
            </div>
            <div className="job-message">{job.message || stageLabel(job.stage)}</div>
            <div className="job-config-line">
              {stageLabel(job.stage)} · {experimentDatasetLabel(job.config)} · seed {job.config.random_seed} · {compactNumber(sampleCount)} mentions · top{" "}
              {compactNumber(job.config.max_candidates)}
            </div>
            <div className="job-config-line">{llmPlanStatsSummary(job)}</div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <JobStageBars job={job} compact />
            <div className="job-row muted-row">
              <span>{stageLabel(job.stage)}</span>
              <span>{total ? `${current}/${total}` : `${progress}%`}</span>
            </div>
            {job.status === "completed" && job.imported_run_id && (
              <button onClick={() => onOpenRun(job.imported_run_id)}>
                <Database size={15} />
                Open run
              </button>
            )}
            {isActiveJob(job) && (
              <button onClick={() => onCancelJob(job.id)} disabled={job.status === "cancel_requested"}>
                <Square size={15} />
                {job.status === "cancel_requested" ? "Stopping" : "Stop"}
              </button>
            )}
            {!isActiveJob(job) && (
              <button onClick={() => onDeleteJob(job)}>
                <Trash2 size={15} />
                Delete job
              </button>
            )}
            {job.status === "failed" && job.error && (
              <details className="job-error">
                <summary>{job.error.split("\n")[0]}</summary>
                <pre>{job.error.split("\n").slice(1, 8).join("\n")}</pre>
              </details>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RunMonitor({
  jobs,
  sourceDatasets,
  onOpenRun,
  onCancelJob,
  onDeleteJob
}: {
  jobs: ExperimentJob[];
  sourceDatasets: SourceDataset[];
  onOpenRun: (runId?: number) => void;
  onCancelJob: (jobId: number) => void;
  onDeleteJob: (job: ExperimentJob) => void;
}) {
  const visibleJobs = jobs.slice(0, 8);
  return (
    <section className="monitor-grid">
      <div className="panel monitor-primary">
        <div className="panel-title">
          <Activity size={17} />
          Background Jobs
        </div>
        {!visibleJobs.length && <div className="empty-state">No web runs yet</div>}
        {visibleJobs.map((job) => (
          <article className={`monitor-job ${job.status}`} key={job.id}>
            <div className="monitor-job-head">
              <div>
                <div className="eyebrow">{stageLabel(job.stage)}</div>
                <h3>Job {job.id}</h3>
              </div>
              <span className={`job-status ${job.status}`}>{job.status}</span>
            </div>
            <div className="monitor-message">{job.message || stageLabel(job.stage)}</div>
            <JobStageBars job={job} />
            <div className="monitor-meta">
              <span>Seed {job.config.random_seed}</span>
              <span>{compactNumber(estimateMentions(job.config, sourceDatasets))} mentions</span>
              <span>top {compactNumber(job.config.max_candidates)} retrieval</span>
              <span>{llmPlanStatsSummary(job)}</span>
              {job.imported_run_id && <span>DB run {job.imported_run_id}</span>}
            </div>
            {job.imported_run_id && (
              <button onClick={() => onOpenRun(job.imported_run_id)}>
                <Database size={15} />
                Open database run
              </button>
            )}
            {isActiveJob(job) && (
              <button onClick={() => onCancelJob(job.id)} disabled={job.status === "cancel_requested"}>
                <Square size={15} />
                {job.status === "cancel_requested" ? "Stopping" : "Stop job"}
              </button>
            )}
            {!isActiveJob(job) && (
              <button onClick={() => onDeleteJob(job)}>
                <Trash2 size={15} />
                Delete job
              </button>
            )}
            {job.status === "failed" && job.error && (
              <details className="job-error">
                <summary>{job.error.split("\n")[0]}</summary>
                <pre>{job.error.split("\n").slice(1, 10).join("\n")}</pre>
              </details>
            )}
          </article>
        ))}
      </div>

      <div className="panel monitor-side">
        <div className="panel-title">Run Storage</div>
        <div className="storage-facts">
          <Metric label="Storage" value="Postgres" />
          <Metric label="Write Mode" value="Incremental" />
          <Metric label="Output JSON" value="Disabled" />
        </div>
        <div className="stage-list">
          {visibleJobs.slice(0, 4).map((job) => (
            <div className="stage-row" key={job.id}>
              <span>Job {job.id}</span>
              <strong>{stageLabel(job.stage)}</strong>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function JobStageBars({ job, compact = false }: { job: ExperimentJob; compact?: boolean }) {
  const progress = job.stage_progress ?? {};
  const queryPlans = progress.query_plans;
  const alpacaSearch = progress.alpaca_search;
  if (!queryPlans && !alpacaSearch) return null;

  return (
    <div className={compact ? "stage-bars compact" : "stage-bars"}>
      {queryPlans && <StageProgressBar progress={queryPlans} />}
      {alpacaSearch && <StageProgressBar progress={alpacaSearch} />}
    </div>
  );
}

function StageProgressBar({ progress }: { progress: NonNullable<ExperimentJob["stage_progress"]>[string] }) {
  const percent = progressPercent(progress.current, progress.total);
  return (
    <div className={`stage-progress ${progress.status}`}>
      <div className="stage-progress-head">
        <span>{progress.label}</span>
        <strong>{compactNumber(progress.current)} / {compactNumber(progress.total)}</strong>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>
      <div className="stage-progress-foot">
        <span>{percent}%</span>
        <span>Elapsed {formatDuration(progress.elapsed_seconds)}</span>
        <span>ETA {progress.status === "completed" ? "done" : formatDuration(progress.eta_seconds)}</span>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function HeuristicPlanPanel({
  analysis,
  problemOnly,
  selectedRun,
  onProblemOnlyChange,
  onRefresh,
  onOpenMention
}: {
  analysis: HeuristicPlanAnalysis | null;
  problemOnly: boolean;
  selectedRun: Run | null;
  onProblemOnlyChange: (value: boolean) => void;
  onRefresh: () => void;
  onOpenMention: (mentionId: number) => void;
}) {
  const summary = analysis?.summary ?? {};
  const rows = analysis?.rows ?? [];
  return (
    <section className="heuristic-plan-grid">
      <div className="panel heuristic-plan-summary">
        <div className="panel-title">
          <AlertCircle size={17} />
          Heuristic Query Plans
        </div>
        <div className="llm-plan-metrics">
          <Metric label="Heuristic Plans" value={compactNumber(summary.heuristic_plan_count)} />
          <Metric label="Fallback Errors" value={compactNumber(summary.llm_fallback_error_count)} />
          <Metric label="Zero Candidates" value={compactNumber(summary.zero_candidate_count)} />
          <Metric label="Alpaca Errors" value={compactNumber(summary.retrieval_error_count)} />
          <Metric label="Retrieval Issues" value={compactNumber(summary.retrieval_problem_count)} />
          <Metric label="Missed" value={compactNumber(summary.missed_count)} />
          <Metric label="Covered" value={compactNumber(summary.covered_count)} />
          <Metric label="Coverage" value={pct(summary.coverage)} />
        </div>
      </div>

      <div className="panel llm-plan-toolbar">
        <div className="table-tools">
          <div className="panel-title">Heuristic Trace</div>
          <label className="inline-toggle">
            <input
              type="checkbox"
              checked={problemOnly}
              onChange={(event) => onProblemOnlyChange(event.target.checked)}
            />
            Problem mentions only
          </label>
          <button onClick={onRefresh}>
            <RefreshCw size={15} />
            Refresh
          </button>
        </div>
      </div>

      <div className="panel heuristic-table-panel">
        {!selectedRun ? (
          <div className="muted metadata-empty">Select a run to inspect heuristic query plans.</div>
        ) : !rows.length ? (
          <div className="muted metadata-empty">No heuristic query plans match this filter.</div>
        ) : (
          <div className="llm-task-table-wrap">
            <table className="heuristic-task-table">
              <thead>
                <tr>
                  <th>Mention</th>
                  <th>Query</th>
                  <th>Retrieval</th>
                  <th>Issue Flags</th>
                  <th>Dataset</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr className={row.retrieval_error || row.candidate_count === 0 || !row.best_gt_rank ? "problem" : ""} key={row.id}>
                    <td>
                      <strong>{row.mention_text || row.lookup_text || "-"}</strong>
                      <span>{row.primary_gt_qid || ""}</span>
                    </td>
                    <td>
                      <strong>{row.optimized_query || row.lookup_text || "-"}</strong>
                      <span>{row.query_plan_source || row.query_engine || "heuristic"}</span>
                      {row.query_plan_error ? <span>{row.query_plan_error}</span> : null}
                    </td>
                    <td>
                      <strong>{compactNumber(row.candidate_count)} candidates</strong>
                      <span>{row.best_gt_rank ? `best rank ${row.best_gt_rank}` : row.retrieval_error ? "Alpaca error" : "not covered"}</span>
                      {row.retrieval_error ? <span>{row.retrieval_error}</span> : null}
                    </td>
                    <td>
                      {(row.troubleshooting_flags ?? []).map((flag) => (
                        <span className="llm-task-chip missing" key={flag}>{flag.replace(/_/g, " ")}</span>
                      ))}
                    </td>
                    <td>
                      <strong>{row.dataset_id || "-"}</strong>
                      <span>{row.table_id || ""}</span>
                    </td>
                    <td>
                      <button type="button" onClick={() => onOpenMention(row.id)}>Open</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

function LlmBatchInspector({ batch }: { batch: LlmQueryPlanBatch }) {
  const taskDetails = batch.task_details ?? [];
  const missingTasks = taskDetails.filter((task) => task.state === "missing" || !task.usable);
  const unknownReturnedTasks = batch.unknown_returned_tasks ?? [];
  const hasProblem = Boolean(
    batch.error ||
    batch.parse_warning ||
    batch.response_parse_error ||
    (batch.missing_task_count ?? 0) > 0
  );
  return (
    <div className="llm-inspector-content">
      <div className="llm-inspector-head">
        <div>
          <div className="panel-title">Batch {batch.id}</div>
          <p>{batch.explanation || "No explanation available for this batch."}</p>
        </div>
        <span className={`llm-status-pill ${hasProblem ? "problem" : "clean"}`}>{hasProblem ? "Needs inspection" : "Clean"}</span>
      </div>

      <div className="llm-diagnosis-grid">
        <Metric label="Requested" value={compactNumber(batch.task_count)} />
        <Metric label="Returned by LLM" value={compactNumber(batch.returned_task_count)} />
        <Metric label="Matched IDs" value={compactNumber(batch.matched_returned_task_count)} />
        <Metric label="Usable Plans" value={compactNumber(batch.usable_task_count)} />
        <Metric label="Missing Requested" value={compactNumber(batch.missing_task_count)} />
        <Metric label="Unknown Returned IDs" value={compactNumber(batch.unknown_returned_task_count)} />
        <Metric label="Input Tokens" value={compactNumber(batch.usage_cost?.input_tokens ?? batch.usage_cost?.prompt_tokens)} />
        <Metric label="Output Tokens" value={compactNumber(batch.usage_cost?.output_tokens ?? batch.usage_cost?.completion_tokens)} />
        <Metric label="Total Tokens" value={compactNumber(batch.usage_cost?.total_tokens)} />
        <Metric label="LLM Cost" value={usageCostLabel(batch.usage_cost)} />
        <Metric label="Cost Source" value={costSourceLabel(batch.usage_cost)} />
        <Metric label="Token Source" value={(batch.usage_cost?.token_source || "-").replace(/_/g, " ")} />
      </div>

      {(batch.error || batch.parse_warning || batch.response_parse_error) && (
        <div className="llm-alert">
          {batch.error || batch.response_parse_error || batch.parse_warning}
        </div>
      )}

      {unknownReturnedTasks.length > 0 && (
        <section className="llm-section">
          <div className="subhead">Returned By LLM But Not Requested</div>
          <div className="llm-task-chips">
            {unknownReturnedTasks.slice(0, 16).map((task) => (
              <span className="llm-task-chip unknown" key={cellText(task.id)}>
                {cellText(task.id)} · {cellText(task.optimized_query || task.normalized_mention || "no query")}
              </span>
            ))}
          </div>
        </section>
      )}

      <section className="llm-section">
        <div className="subhead">Requested Tasks</div>
        {taskDetails.length ? (
          <div className="llm-task-table-wrap">
            <table className="llm-task-table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Mention</th>
                  <th>LLM Query Plan</th>
                  <th>Type</th>
                  <th>Task ID</th>
                </tr>
              </thead>
              <tbody>
                {taskDetails.map((task) => (
                  <tr className={task.usable ? "" : "problem"} key={task.task_id}>
                    <td>
                      <span className={`llm-task-state ${task.usable ? "usable" : task.returned ? "returned-not-usable" : "missing"}`}>
                        {task.usable ? "usable" : task.returned ? "returned, not usable" : "missing"}
                      </span>
                    </td>
                    <td>
                      <strong>{task.mention_text || task.lookup_text || "-"}</strong>
                      <span>{task.lookup_text && task.lookup_text !== task.mention_text ? task.lookup_text : ""}</span>
                    </td>
                    <td>
                      <strong>{task.optimized_query || "-"}</strong>
                      <span>{task.normalized_mention || ""}</span>
                    </td>
                    <td>{[task.coarse_type, task.fine_type].filter(Boolean).join(" / ") || "-"}</td>
                    <td><code>{task.task_id}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="muted metadata-empty">No per-task rows were stored for this batch.</div>
        )}
      </section>

      {missingTasks.length > 0 && (
        <section className="llm-section">
          <div className="subhead">Missing Or Unusable Requested Tasks</div>
          <div className="llm-task-chips">
            {missingTasks.slice(0, 24).map((task) => (
              <span className="llm-task-chip missing" key={task.task_id}>
                {task.mention_text || task.lookup_text || task.task_id}
              </span>
            ))}
          </div>
        </section>
      )}

      {batch.usage_cost?.notes?.length ? (
        <section className="llm-section">
          <div className="subhead">Cost Notes</div>
          <div className="muted metadata-empty">{batch.usage_cost.notes.join(" ")}</div>
        </section>
      ) : null}

      <details className="llm-raw-answer">
        <summary>Raw LLM answer</summary>
        {batch.response_content ? (
          <pre>{batch.response_content}</pre>
        ) : (
          <div className="muted metadata-empty">No raw LLM answer was stored for this batch.</div>
        )}
      </details>
    </div>
  );
}

function liveAttemptDiagnostics(attempt: LiveAttempt | null | undefined): ImprovementDiagnostics | null {
  const payload = attempt?.response_payload as { improvement_diagnostics?: ImprovementDiagnostics } | undefined;
  return payload?.improvement_diagnostics ?? null;
}

function tokenChips(tokens: string[] | undefined) {
  if (!tokens?.length) return <span className="muted">none</span>;
  return tokens.slice(0, 10).map((token) => (
    <span className="qid" key={token}>
      {token}
    </span>
  ));
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  return String(value);
}

function rawValue(record: Record<string, unknown> | undefined, keys: string[]): unknown {
  if (!record) return undefined;
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null && record[key] !== "") return record[key];
  }
  return undefined;
}

function compactValue(value: unknown): string {
  if (Array.isArray(value)) return value.map((item) => compactValue(item)).filter(Boolean).slice(0, 4).join(", ");
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return cellText(value);
}

function metadataPairs(record: Record<string, unknown> | undefined, keys: Array<[string, string[]]>) {
  return keys
    .map(([label, aliases]) => [label, compactValue(rawValue(record, aliases))] as const)
    .filter(([, value]) => value);
}

function attemptSearchText(attempt: LiveAttempt | null | undefined): string {
  const response = attempt?.response_payload as {
    augmentation_terms?: unknown;
    context_terms?: unknown;
    query_plan_terms?: unknown;
    search_text?: unknown;
  } | undefined;
  const request = attempt?.request_payload as { query?: unknown } | undefined;
  const augmentationTerms = Array.isArray(response?.query_plan_terms)
    ? response.query_plan_terms.map((item) => cellText(item)).filter(Boolean)
    : Array.isArray(response?.augmentation_terms)
    ? response.augmentation_terms.map((item) => cellText(item)).filter(Boolean)
    : Array.isArray(response?.context_terms)
      ? response.context_terms.map((item) => cellText(item)).filter(Boolean)
      : [];
  const safeTerms = augmentationTerms.filter((term) => term.trim().split(/\s+/).length <= 2);
  if (safeTerms.length) {
    return [attempt?.query_text, safeTerms.join(" ")]
      .map((part) => cellText(part).trim())
      .filter(Boolean)
      .join(" ");
  }
  const compactAttemptText = [attempt?.query_text]
    .map((part) => cellText(part).trim())
    .filter(Boolean)
    .join(" ");
  const text = compactAttemptText || cellText(request?.query ?? response?.search_text ?? "");
  return text.length > 220 ? `${text.slice(0, 220)}...` : text;
}

function attemptAugmentationSummary(attempt: LiveAttempt | null | undefined): string {
  const response = attempt?.response_payload as {
    augmentation_error?: unknown;
    augmentation_source?: unknown;
    augmentation_terms?: unknown;
    query_plan_error?: unknown;
    query_plan_source?: unknown;
    query_plan_terms?: unknown;
  } | undefined;
  const terms = Array.isArray(response?.query_plan_terms)
    ? response.query_plan_terms.map((item) => cellText(item)).filter(Boolean)
    : Array.isArray(response?.augmentation_terms)
    ? response.augmentation_terms.map((item) => cellText(item)).filter(Boolean)
    : [];
  const safeTerms = terms.filter((term) => term.trim().split(/\s+/).length <= 2);
  const source = cellText(response?.query_plan_source || response?.augmentation_source || "none");
  const suffix = cellText(response?.query_plan_error || response?.augmentation_error) ? "fallback used" : source;
  return safeTerms.length ? `${safeTerms.join(", ")} (${suffix})` : `No extra terms (${suffix})`;
}

function requestJson(attempt: LiveAttempt | null | undefined): string {
  if (!attempt?.request_payload) return "";
  return JSON.stringify(attempt.request_payload, null, 2);
}

type InspectionItem = {
  title: string;
  subtitle: string;
  status?: string;
  payload?: unknown;
};

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function payloadString(value: unknown): string {
  if (value === undefined || value === null) return "";
  return JSON.stringify(value, null, 2);
}

function promptContentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        const record = asRecord(part);
        if (typeof record?.text === "string") return record.text;
        if (typeof record?.content === "string") return record.content;
        return payloadString(part);
      })
      .filter(Boolean)
      .join("\n");
  }
  if (content === undefined || content === null) return "";
  return payloadString(content);
}

function promptMessages(payload: unknown): unknown[] {
  const record = asRecord(payload);
  const requestBody = asRecord(record?.request_body) ?? asRecord(record?.request_payload) ?? record;
  return Array.isArray(requestBody?.messages) ? requestBody.messages : [];
}

function promptPlainText(messages: unknown[]): string {
  if (!messages.length) return "";
  return messages
    .map((message, index) => {
      const item = asRecord(message);
      const role = cellText(item?.role || `message ${index + 1}`).toUpperCase();
      const content = promptContentText(item?.content);
      return `${role}:\n${content}`;
    })
    .join("\n\n");
}

function promptMessagesText(messages: unknown[]): string {
  if (!messages.length) return "";
  return `messages=${JSON.stringify(messages)}`;
}

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.position = "fixed";
  textArea.style.left = "-9999px";
  textArea.style.top = "0";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  document.execCommand("copy");
  document.body.removeChild(textArea);
}

function inspectionItems(detail: MentionDetail, latestAttempt: LiveAttempt | null): InspectionItem[] {
  const rawMention = detail.mention.raw_payload ?? {};
  const queryPlan = asRecord(rawMention.query_plan) ?? asRecord(asRecord(rawMention.retrieval_debug)?.query_plan);
  const backendRequests = Array.isArray(rawMention.backend_requests) ? rawMention.backend_requests : [];
  const firstBackendRequest = asRecord(backendRequests[0]);
  const llmInspection = asRecord(queryPlan?.llm_inspection);
  const queryPlanBatch = asRecord(detail.query_plan_batch);
  const liveRequest = asRecord(latestAttempt?.request_payload);
  const liveResponse = asRecord(latestAttempt?.response_payload);

  const items: InspectionItem[] = [
    {
      title: "Imported LLM Prompt",
      subtitle: queryPlanBatch ? "Shared LLM query-plan request for this mention batch." : llmInspection?.sent === false ? "Prompt metadata available, but the LLM was not sent for this plan." : "LLM query-plan request for this mention batch.",
      status: cellText(queryPlanBatch?.status ?? queryPlan?.source ?? llmInspection?.sent ?? "not stored"),
      payload: queryPlanBatch ?? llmInspection ?? queryPlan,
    },
    {
      title: "Imported Candidate Fetch",
      subtitle: "Alpaca request used to fetch candidates for the imported mention.",
      status: cellText(firstBackendRequest?.status ?? "not stored"),
      payload: firstBackendRequest ?? asRecord(rawMention.retrieval_debug)?.sanitized_request_body,
    },
  ];

  if (latestAttempt) {
    items.push(
      {
        title: "Live LLM Prompt",
        subtitle: "LLM request used to plan the live query terms.",
        status: cellText(asRecord(liveRequest?.llm_query_plan)?.sent ?? liveResponse?.query_plan_source ?? "not stored"),
        payload: liveRequest?.llm_query_plan ?? liveResponse?.llm_query_plan,
      },
      {
        title: "Live Candidate Fetch",
        subtitle: "Alpaca request used for the latest live attempt.",
        status: latestAttempt.error ? "error" : latestAttempt.covered ? "covered" : "missed",
        payload: liveRequest?.alpaca_candidate_fetch ?? liveResponse?.alpaca_candidate_fetch ?? latestAttempt.request_payload,
      }
    );
  }

  return items;
}

function InspectionPanel({ items }: { items: InspectionItem[] }) {
  const [copiedKey, setCopiedKey] = useState("");

  async function copyPrompt(key: string, promptText: string) {
    try {
      await copyText(promptText);
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey((current) => (current === key ? "" : current)), 1600);
    } catch {
      setCopiedKey("");
    }
  }

  return (
    <section className="subsection grow">
      <div className="subhead">Inspection</div>
      <div className="inspection-list">
        {items.map((item) => {
          const body = payloadString(item.payload);
          const messages = promptMessages(item.payload);
          const promptText = promptPlainText(messages);
          const messagesText = promptMessagesText(messages);
          const plainCopyKey = `${item.title}:plain`;
          const messagesCopyKey = `${item.title}:messages`;
          return (
            <details className="inspection-card" key={item.title}>
              <summary>
                <span>
                  <strong>{item.title}</strong>
                  <em>{item.subtitle}</em>
                </span>
                <b>{item.status || "-"}</b>
              </summary>
              {promptText && (
                <div className="prompt-debug">
                  <div className="prompt-debug-head">
                    <strong>Prompt Plain Text</strong>
                    <div className="prompt-debug-actions">
                      <button type="button" onClick={() => copyPrompt(plainCopyKey, promptText)}>
                        {copiedKey === plainCopyKey ? <Check size={15} /> : <Clipboard size={15} />}
                        {copiedKey === plainCopyKey ? "Copied" : "Copy plain"}
                      </button>
                      {messagesText && (
                        <button type="button" onClick={() => copyPrompt(messagesCopyKey, messagesText)}>
                          {copiedKey === messagesCopyKey ? <Check size={15} /> : <Clipboard size={15} />}
                          {copiedKey === messagesCopyKey ? "Copied" : "Copy messages=[]"}
                        </button>
                      )}
                    </div>
                  </div>
                  <pre>{promptText}</pre>
                </div>
              )}
              {body ? <pre>{body}</pre> : <div className="muted metadata-empty">No inspection payload stored for this item.</div>}
            </details>
          );
        })}
      </div>
    </section>
  );
}

function MetadataChips({ pairs, empty }: { pairs: Array<readonly [string, string]>; empty: string }) {
  if (!pairs.length) return <div className="muted metadata-empty">{empty}</div>;
  return (
    <div className="metadata-grid">
      {pairs.map(([label, value]) => (
        <div className="metadata-chip" key={label}>
          <span>{label}</span>
          <strong title={value}>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function firstGoldCandidatePage(candidates: Candidate[]): number {
  const index = candidates.findIndex((candidate) => candidate.gold_match);
  return index >= 0 ? Math.floor(index / CANDIDATE_PAGE_SIZE) : 0;
}

function MentionInspector({
  detail,
  llmConfig,
  onRefresh
}: {
  detail: MentionDetail | null;
  llmConfig?: Partial<ExperimentConfig>;
  onRefresh: () => Promise<void>;
}) {
  const [category, setCategory] = useState("miss_reason");
  const [note, setNote] = useState("");
  const [liveQuery, setLiveQuery] = useState("");
  const [liveGuidance, setLiveGuidance] = useState("");
  const [candidateCount, setCandidateCount] = useState(300);
  const [liveCandidates, setLiveCandidates] = useState<Candidate[]>([]);
  const [candidatePage, setCandidatePage] = useState(0);
  const [activeTab, setActiveTab] = useState<InspectorTab>("overview");
  const [goldMetadata, setGoldMetadata] = useState<GoldMetadataResult | null>(null);
  const [localError, setLocalError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setLiveQuery(detail?.mention.lookup_text || detail?.mention.mention || "");
    setLiveCandidates([]);
    setCandidatePage(0);
    setActiveTab("overview");
    setGoldMetadata(null);
    setLocalError("");
  }, [detail?.mention.id]);

  if (!detail) {
    return (
      <div className="panel inspector empty-state">
        <Database size={22} />
        Select a mention
      </div>
    );
  }

  const activeDetail = detail;

  async function saveFeedback() {
    if (!note.trim()) return;
    setBusy(true);
    setLocalError("");
    try {
      await api.addFeedback(activeDetail.mention.id, { category, note });
      setNote("");
      await onRefresh();
    } catch (exc) {
      setLocalError(readableError(exc));
    } finally {
      setBusy(false);
    }
  }

  async function runLiveAttempt() {
    setBusy(true);
    setLocalError("");
    try {
      const result = await api.liveAttempt(activeDetail.mention.id, {
        candidate_count: candidateCount,
        query_text: liveQuery,
        human_guidance: liveGuidance,
        llm_config: llmConfig
      });
      const resultCandidates = result.candidates ?? [];
      setLiveCandidates(resultCandidates);
      setCandidatePage(firstGoldCandidatePage(resultCandidates));
      await onRefresh();
    } catch (exc) {
      setLocalError(readableError(exc));
      await onRefresh();
    } finally {
      setBusy(false);
    }
  }

  async function fetchGoldMetadata() {
    setBusy(true);
    setLocalError("");
    try {
      setGoldMetadata(await api.goldMetadata(activeDetail.mention.id));
    } catch (exc) {
      setLocalError(readableError(exc));
    } finally {
      setBusy(false);
    }
  }

  const latestAttempt = detail.live_attempts[0] ?? null;
  const latestAttemptCandidates = latestAttempt?.candidates ?? [];
  const candidates = liveCandidates.length ? liveCandidates : latestAttemptCandidates.length ? latestAttemptCandidates : detail.candidates;
  const goldCandidateIndex = candidates.findIndex((candidate) => candidate.gold_match);
  const goldCandidate = goldCandidateIndex >= 0 ? candidates[goldCandidateIndex] : null;
  const candidatePageCount = Math.max(1, Math.ceil(candidates.length / CANDIDATE_PAGE_SIZE));
  const boundedCandidatePage = Math.min(candidatePage, candidatePageCount - 1);
  const visibleCandidates = candidates.slice(
    boundedCandidatePage * CANDIDATE_PAGE_SIZE,
    boundedCandidatePage * CANDIDATE_PAGE_SIZE + CANDIDATE_PAGE_SIZE
  );
  const diagnostics = liveAttemptDiagnostics(latestAttempt);
  const inspectableItems = inspectionItems(detail, latestAttempt);
  const tableContextRows = detail.table_context?.rows ?? [];
  const tableHeader = detail.table_context?.header ?? [];
  const targetColId = detail.table_context?.target_col_id ?? detail.mention.col_id;
  const rawMention = detail.mention.raw_payload ?? {};
  const goldEntity = goldMetadata?.entity ?? detail.gold_qids.find((item) => item.raw_entity)?.raw_entity;
  const goldNerTypes = goldMetadata?.ner_types ?? [];
  const goldPairs = metadataPairs(goldEntity, [
    ["Label", ["label"]],
    ["Description", ["description", "context_string"]],
    ["Fine Type", ["fine_type"]],
    ["Coarse Type", ["coarse_type"]],
    ["Aliases", ["aliases", "labels"]],
    ["Category", ["item_category"]],
  ]);
  const mentionPairs = metadataPairs(rawMention, [
    ["Lookup", ["lookup_text", "mention_text"]],
    ["Header", ["header_cell", "column_header"]],
    ["Backend", ["candidate_backend", "query_engine"]],
    ["Query", ["query", "query_text", "alpaca_query"]],
    ["GT entities", ["gt_entities"]],
  ]);

  return (
    <div className="panel inspector">
      <div className="inspector-head">
        <div>
          <div className="eyebrow">{detail.mention.cell_key}</div>
          <h3>{detail.mention.mention ?? detail.mention.lookup_text}</h3>
        </div>
        <span className={`pill ${detail.mention.covered_by_imported_candidates ? "ok" : "miss"}`}>
          {coverageBadge(detail.mention.covered_by_imported_candidates)}
        </span>
      </div>

      {localError && (
        <div className="error-box compact">
          <AlertCircle size={15} />
          {localError}
        </div>
      )}

      <div className="kv-grid">
        <Metric label="Dataset" value={detail.mention.dataset_id ?? "-"} />
        <Metric label="Table" value={detail.mention.table_id ?? "-"} />
        <Metric label="Stored Candidates" value={`${compactNumber(detail.candidates.length)} / ${MAX_RETURNED_CANDIDATES}`} />
        <Metric label="Best Rank" value={compactNumber(detail.mention.imported_best_rank)} />
      </div>

      <nav className="inspector-tabs">
        {(["overview", "live", "candidates", "inspection", "feedback"] as InspectorTab[]).map((tab) => (
          <button
            className={activeTab === tab ? "active" : ""}
            key={tab}
            onClick={() => {
              setActiveTab(tab);
              if (tab === "candidates") setCandidatePage(firstGoldCandidatePage(candidates));
            }}
          >
            {tab}
          </button>
        ))}
      </nav>

      <div className="inspector-body">
        {activeTab === "overview" && (
          <>
            <section className="subsection compact-section">
              <div className="subhead">Gold QIDs and Metadata</div>
              <div className="qid-list">
                {detail.gold_qids.map((item) => (
                  <span className="qid" key={item.qid}>
                    {item.qid}
                  </span>
                ))}
              </div>
              <button onClick={fetchGoldMetadata} disabled={busy || !detail.gold_qids.length}>
                <Database size={15} />
                Fetch Alpaca GT
              </button>
              {goldMetadata && (
                <div className={`note ${goldMetadata.resolved_qid ? "success-note" : ""}`}>
                  <strong>{goldMetadata.resolved_qid ? `Resolved ${goldMetadata.resolved_qid}` : "No Alpaca QID found"}</strong>
                  <span>
                    {goldMetadata.entity
                      ? `${goldMetadata.entity.label ?? ""} fine=${goldMetadata.entity.fine_type ?? "-"} coarse=${goldMetadata.entity.coarse_type ?? "-"}`.trim()
                      : `Checked ${goldMetadata.requested_qids.join(", ")}`}
                  </span>
                  {goldNerTypes.length > 1 && (
                    <span>
                      {goldNerTypes
                        .map((item) => `${item.qid ?? "gold"}: fine=${item.fine_type ?? "-"} coarse=${item.coarse_type ?? "-"}`)
                        .join(" | ")}
                    </span>
                  )}
                </div>
              )}
              <MetadataChips pairs={goldPairs} empty="Fetch Alpaca GT to load lazy gold metadata." />
            </section>

            <section className="subsection compact-section">
              <div className="subhead">Mention Metadata</div>
              <MetadataChips pairs={mentionPairs} empty="No extra mention metadata stored." />
            </section>

            <section className="subsection grow">
              <div className="subhead">Table Context</div>
              {tableContextRows.length ? (
                <div className="context-table">
                  <table>
                    {tableHeader.length > 0 && (
                      <thead>
                        <tr>
                          <th>Row</th>
                          {tableHeader.map((cell, index) => (
                            <th key={`${cellText(cell)}-${index}`}>{cellText(cell) || `Col ${index}`}</th>
                          ))}
                        </tr>
                      </thead>
                    )}
                    <tbody>
                      {tableContextRows.map((row, rowIndex) => {
                        const cells = row.cells ?? [];
                        return (
                          <tr key={`${row.row_id ?? rowIndex}-${row.relative_position ?? rowIndex}`} className={row.is_target ? "target-row" : ""}>
                            <td>
                              <strong>{row.row_id ?? "-"}</strong>
                              <span>{row.is_target ? "source row" : row.relative_position ? `${row.relative_position}` : ""}</span>
                            </td>
                            {cells.map((cell, cellIndex) => (
                              <td
                                key={`${row.row_id ?? rowIndex}-${cellIndex}`}
                                className={row.is_target && cellIndex === targetColId ? "mention-cell" : ""}
                              >
                                {cellText(cell)}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-state compact">No table context stored for this mention</div>
              )}
            </section>
          </>
        )}

        {activeTab === "live" && (
          <>
            <section className="subsection compact-section">
              <div className="subhead">Live Attempt</div>
        <div className="live-grid">
          <label>
            Query
            <input value={liveQuery} onChange={(event) => setLiveQuery(event.target.value)} />
          </label>
          <label>
            Retrieval window
            <input
              type="number"
              min={1}
              max={MAX_RETRIEVAL_CANDIDATES}
              value={candidateCount}
              onChange={(event) =>
                setCandidateCount(Math.min(MAX_RETRIEVAL_CANDIDATES, Math.max(1, Number(event.target.value))))
              }
            />
          </label>
          <label className="live-guidance">
            ES query guidance
            <textarea value={liveGuidance} onChange={(event) => setLiveGuidance(event.target.value)} />
          </label>
          <button className="primary" onClick={runLiveAttempt} disabled={busy}>
            <Send size={15} />
            Run
          </button>
        </div>
        {detail.live_attempts.length > 0 && (
          <div className="notes">
            {detail.live_attempts.slice(0, 3).map((attempt) => (
              <div className={`note ${attempt.covered ? "success-note" : ""}`} key={attempt.id}>
                <strong>{attempt.covered ? "GT covered in live attempt" : "missed"} · top {compactNumber(attempt.candidate_count)}</strong>
                <span>
                  {attempt.covered_qids.length ? attempt.covered_qids.join(", ") : attempt.error || attempt.query_text || "-"}
                </span>
              </div>
            ))}
          </div>
        )}
            </section>
            {latestAttempt && (
              <section className="subsection compact-section">
                <div className="subhead">Alpaca Query Used</div>
                <div className="query-card">{attemptSearchText(latestAttempt)}</div>
                <div className="augmentation-line">
                  <strong>ES query plan terms</strong>
                  <span>{attemptAugmentationSummary(latestAttempt)}</span>
                </div>
                <details className="json-details">
                  <summary>Request JSON</summary>
                  <pre>{requestJson(latestAttempt)}</pre>
                </details>
              </section>
            )}
        {diagnostics && (
          <section className="notes compact-section">
            <div className="note">
              <strong>NER type hint</strong>
              <span>
                {(diagnostics.ner_type_hint?.gold_types ?? [])
                  .map((item) => `${item.qid ?? "gold"}: ${item.coarse_type ?? "-"} / ${item.fine_type ?? "-"}`)
                  .join(" · ") || "No gold type metadata available"}
              </span>
              {diagnostics.ner_type_hint?.type_mismatch_with_top_candidates && (
                <span>Top candidates are dominated by a different type.</span>
              )}
            </div>
            <div className="note">
              <strong>Context token hint</strong>
              <span>{tokenChips(diagnostics.context_token_hint?.candidate_context_tokens_to_consider)}</span>
            </div>
            {diagnostics.candidate_type_distribution?.length ? (
              <div className="note">
                <strong>Candidate type mix</strong>
                <span>
                  {diagnostics.candidate_type_distribution
                    .map((item) => `${item.coarse_type ?? "-"} / ${item.fine_type ?? "-"} (${item.count ?? 0})`)
                    .join(" · ")}
                </span>
              </div>
            ) : null}
            {diagnostics.recommendations?.length ? (
              <div className="note">
                <strong>Next system change</strong>
                <span>{diagnostics.recommendations.join(" ")}</span>
              </div>
            ) : null}
          </section>
        )}
          </>
        )}

        {activeTab === "feedback" && (
          <section className="subsection compact-section">
            <div className="subhead">Feedback</div>
        <div className="feedback-grid">
          <label>
            Category
            <select value={category} onChange={(event) => setCategory(event.target.value)}>
              <option value="miss_reason">Miss reason</option>
              <option value="retrieval_hint">Retrieval hint</option>
              <option value="ner_type_hint">NER type hint</option>
              <option value="context_token_hint">Context token hint</option>
              <option value="data_issue">Data issue</option>
              <option value="note">Note</option>
            </select>
          </label>
          <label>
            Note
            <textarea value={note} onChange={(event) => setNote(event.target.value)} />
          </label>
          <button onClick={saveFeedback} disabled={busy || !note.trim()}>
            Save
          </button>
        </div>
        {detail.feedback.length > 0 && (
          <div className="notes">
            {detail.feedback.slice(0, 4).map((item) => (
              <div className="note" key={item.id}>
                <strong>{item.category}</strong>
                <span>{item.note}</span>
              </div>
            ))}
          </div>
        )}
          </section>
        )}

        {activeTab === "inspection" && <InspectionPanel items={inspectableItems} />}

        {activeTab === "candidates" && (
          <section className="subsection grow">
            <div className="subhead">Candidates</div>
            {goldCandidate ? (
              <div className="gold-hit-summary">
                <strong>GT candidate covered</strong>
                <span>
                  Rank {compactNumber(goldCandidate.rank ?? goldCandidateIndex + 1)} · {goldCandidate.qid} · {goldCandidate.label ?? "-"}
                </span>
              </div>
            ) : latestAttempt?.covered ? (
              <div className="gold-hit-summary">
                <strong>GT covered in live attempt</strong>
                <span>{latestAttempt.covered_qids.join(", ")} was found in the retrieval window.</span>
              </div>
            ) : null}
        <div className="candidate-pager">
          <span>
            {compactNumber(candidates.length ? boundedCandidatePage * CANDIDATE_PAGE_SIZE + 1 : 0)}-
            {compactNumber(Math.min((boundedCandidatePage + 1) * CANDIDATE_PAGE_SIZE, candidates.length))} / {compactNumber(candidates.length)}
          </span>
          <button disabled={boundedCandidatePage === 0} onClick={() => setCandidatePage(Math.max(0, boundedCandidatePage - 1))}>
            Previous
          </button>
          <button
            disabled={boundedCandidatePage + 1 >= candidatePageCount}
            onClick={() => setCandidatePage(Math.min(candidatePageCount - 1, boundedCandidatePage + 1))}
          >
            Next
          </button>
        </div>
        <div className="candidate-table">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>QID</th>
                <th>Label</th>
                <th>Stage</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {visibleCandidates.map((candidate, index) => (
                <tr key={`${candidate.qid}-${candidate.rank}-${index}`} className={candidate.gold_match ? "gold-row" : ""}>
                  <td>{candidate.rank ?? boundedCandidatePage * CANDIDATE_PAGE_SIZE + index + 1}</td>
                  <td>{candidate.qid}</td>
                  <td>
                    <strong>{candidate.label}</strong>
                    <span>{candidate.raw_payload?.description as string ?? candidate.fine_type ?? candidate.coarse_type ?? ""}</span>
                  </td>
                  <td>{candidate.retrieval_stage ?? candidate.retrieval_system ?? "-"}</td>
                  <td>{candidate.score?.toFixed(3) ?? candidate.es_score?.toFixed(3) ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
          </section>
        )}
      </div>
    </div>
  );
}
