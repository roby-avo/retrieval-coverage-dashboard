import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Database,
  ListChecks,
  Play,
  RefreshCw,
  Search,
  Send,
  Settings2
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
  ImprovementDiagnostics,
  LiveAttempt,
  MentionDetail,
  MentionRow,
  Run,
  SourceDataset
} from "./api";

const PAGE_SIZE = 100;
const MAX_RETRIEVAL_CANDIDATES = 1000;
const MAX_RETURNED_CANDIDATES = 1000;
const CANDIDATE_PAGE_SIZE = 100;
type WorkspaceView = "overview" | "mentions" | "monitor";
type InspectorTab = "overview" | "live" | "candidates" | "inspection" | "feedback";

function pct(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function compactNumber(value: number | undefined | null): string {
  if (value === undefined || value === null) return "-";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value);
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
    completed: "Completed",
    failed: "Failed"
  };
  return labels[stage] ?? stage.replace(/_/g, " ");
}

function jobProgress(job: ExperimentJob): number {
  if (job.status === "completed") return 100;
  if (!job.progress_total) return job.status === "running" ? 8 : 0;
  return Math.max(2, Math.min(100, Math.round((job.progress_current / job.progress_total) * 100)));
}

function estimateMentions(config: ExperimentConfig | null): number | null {
  if (!config) return null;
  const targetedDatasets = config.dataset_allowlist?.length ?? 0;
  const requestedDatasets = config.requested_datasets?.length ?? 0;
  const availableDatasets = targetedDatasets || requestedDatasets || config.dataset_sample_size;
  const datasetCount = config.dataset_sample_size > 0 ? Math.min(config.dataset_sample_size, availableDatasets) : availableDatasets;
  return datasetCount * config.tables_per_dataset * config.records_per_table;
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
    }
  }

  async function refreshRunData(runId: number) {
    const [run, coverageRows, filterRows, mentionRows] = await Promise.all([
      api.run(runId),
      api.coverage(runId),
      api.filters(runId),
      api.mentions(runId, {
        limit: PAGE_SIZE,
        offset,
        covered: coveredFilter,
        dataset_id: datasetFilter,
        search
      })
    ]);
    setSelectedRun(run);
    setCoverage(coverageRows);
    setFilters(filterRows);
    setMentions(mentionRows.rows);
    setMentionTotal(mentionRows.total);
    if (mentionRows.rows.length && !detail) {
      setDetail(await api.mention(mentionRows.rows[0].id));
    }
  }

  useEffect(() => {
    Promise.all([
      refreshRuns(),
      api.experimentDefaults().then(setExperimentConfig),
      api.configStatus().then(setConfigStatus),
      api.sourceDatasets().then(setSourceDatasets),
      api.databaseSize().then(setDatabaseSize)
    ]).catch((exc) =>
      setError(String(exc.message ?? exc))
    );
  }, []);

  useEffect(() => {
    const hasActiveJob = jobs.some((job) => job.status === "queued" || job.status === "running");
    if (!hasActiveJob) return;
    const timer = window.setInterval(async () => {
      try {
        const jobRows = await api.experimentJobs();
        setJobs(jobRows);
        const activeImported = jobRows.find((job) => (job.status === "queued" || job.status === "running") && job.imported_run_id);
        if (activeImported?.imported_run_id) {
          await refreshRuns(activeImported.imported_run_id);
        }
        const completed = jobRows.find((job) => job.id === activeJobId && job.status === "completed" && job.imported_run_id);
        if (completed?.imported_run_id) {
          await refreshRuns(completed.imported_run_id);
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
  }, [selectedRunId, coveredFilter, datasetFilter, offset]);

  const chartData = useMemo(
    () =>
      coverage.map((item) => ({
        ...item,
        coveragePercent: Number((item.coverage * 100).toFixed(2))
      })),
    [coverage]
  );

  const estimatedMentions = estimateMentions(experimentConfig);

  function updateExperimentConfig<K extends keyof ExperimentConfig>(key: K, value: ExperimentConfig[K]) {
    setExperimentConfig((current) => (current ? { ...current, [key]: value } : current));
  }

  function updateTargetDataset(datasetId: string) {
    setExperimentConfig((current) => {
      if (!current) return current;
      if (!datasetId) {
        return { ...current, dataset_allowlist: [] };
      }
      return { ...current, dataset_allowlist: [datasetId], dataset_sample_size: 1 };
    });
  }

  async function startExperiment() {
    if (!experimentConfig) return;
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
      setStatus(`Cleared ${result.deleted} failed job${result.deleted === 1 ? "" : "s"}`);
    } catch (exc) {
      setError(String((exc as Error).message ?? exc));
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
                <span className={`config-chip ${configStatus?.openrouter_configured ? "ok" : "warn"}`}>
                  OpenRouter {configStatus?.openrouter_configured ? "ready" : "optional/missing"}
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
                    value={experimentConfig.tables_per_dataset}
                    onChange={(event) => updateExperimentConfig("tables_per_dataset", Number(event.target.value))}
                  />
                </label>
                <label>
                  Records
                  <input
                    type="number"
                    min={0}
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
                    OpenRouter parallel
                    <input
                      type="number"
                      min={1}
                      max={16}
                      value={experimentConfig.openrouter_parallel_requests}
                      onChange={(event) => updateExperimentConfig("openrouter_parallel_requests", Number(event.target.value))}
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={experimentConfig.use_openrouter}
                      onChange={(event) => updateExperimentConfig("use_openrouter", event.target.checked)}
                    />
                    OpenRouter
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
              <button className="primary" disabled={jobBusy} onClick={startExperiment}>
                <Play size={16} />
                {jobBusy ? "Starting..." : "Start background run"}
              </button>
            </>
          )}
          <JobList jobs={jobs} onOpenRun={(runId) => runId && refreshRuns(runId)} onClearFailed={clearFailedJobs} />
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
          <button onClick={() => refreshRuns()}>
            <RefreshCw size={16} />
            Refresh
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
          <button className={workspaceView === "monitor" ? "active" : ""} onClick={() => setWorkspaceView("monitor")}>
            <Activity size={16} />
            Run Monitor
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

        {workspaceView === "monitor" && (
          <RunMonitor jobs={jobs} onOpenRun={(runId) => runId && refreshRuns(runId)} />
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
                          <span>{row.selected_label ?? row.selected_qid ?? ""}</span>
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

function JobList({
  jobs,
  onOpenRun,
  onClearFailed
}: {
  jobs: ExperimentJob[];
  onOpenRun: (runId?: number) => void;
  onClearFailed: () => void;
}) {
  if (!jobs.length) {
    return <div className="job-empty">No web runs yet</div>;
  }
  return (
    <div className="job-list">
      <div className="job-list-head">
        <span>Recent web runs</span>
        {jobs.some((job) => job.status === "failed") ? (
          <button className="text-button" onClick={onClearFailed}>
            Clear failed
          </button>
        ) : (
          <span>{jobs.length}</span>
        )}
      </div>
      {jobs.slice(0, 4).map((job) => {
        const total = job.progress_total || 0;
        const current = job.progress_current || 0;
        const progress = jobProgress(job);
        const sampleCount = estimateMentions(job.config);
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

function RunMonitor({ jobs, onOpenRun }: { jobs: ExperimentJob[]; onOpenRun: (runId?: number) => void }) {
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
              <span>{compactNumber(estimateMentions(job.config))} mentions</span>
              <span>top {compactNumber(job.config.max_candidates)} retrieval</span>
              {job.imported_run_id && <span>DB run {job.imported_run_id}</span>}
            </div>
            {job.imported_run_id && (
              <button onClick={() => onOpenRun(job.imported_run_id)}>
                <Database size={15} />
                Open database run
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

function inspectionItems(detail: MentionDetail, latestAttempt: LiveAttempt | null): InspectionItem[] {
  const rawMention = detail.mention.raw_payload ?? {};
  const queryPlan = asRecord(rawMention.query_plan) ?? asRecord(asRecord(rawMention.retrieval_debug)?.query_plan);
  const backendRequests = Array.isArray(rawMention.backend_requests) ? rawMention.backend_requests : [];
  const firstBackendRequest = asRecord(backendRequests[0]);
  const llmInspection = asRecord(queryPlan?.llm_inspection);
  const liveRequest = asRecord(latestAttempt?.request_payload);
  const liveResponse = asRecord(latestAttempt?.response_payload);

  const items: InspectionItem[] = [
    {
      title: "Imported LLM Prompt",
      subtitle: llmInspection?.sent === false ? "Prompt metadata available, but OpenRouter was not sent for this plan." : "OpenRouter query-plan request for this mention batch.",
      status: cellText(queryPlan?.source ?? llmInspection?.sent ?? "not stored"),
      payload: llmInspection ?? queryPlan,
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
        subtitle: "OpenRouter request used to plan the live query terms.",
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
  return (
    <section className="subsection grow">
      <div className="subhead">Inspection</div>
      <div className="inspection-list">
        {items.map((item) => {
          const body = payloadString(item.payload);
          return (
            <details className="inspection-card" key={item.title}>
              <summary>
                <span>
                  <strong>{item.title}</strong>
                  <em>{item.subtitle}</em>
                </span>
                <b>{item.status || "-"}</b>
              </summary>
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

function MentionInspector({ detail, onRefresh }: { detail: MentionDetail | null; onRefresh: () => Promise<void> }) {
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
        human_guidance: liveGuidance
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
  const goldPairs = metadataPairs(goldEntity, [
    ["Label", ["label"]],
    ["Description", ["description", "context_string"]],
    ["Type", ["fine_type", "coarse_type"]],
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
                      ? `${goldMetadata.entity.label ?? ""} ${goldMetadata.entity.fine_type ?? ""} ${goldMetadata.entity.coarse_type ?? ""}`.trim()
                      : `Checked ${goldMetadata.requested_qids.join(", ")}`}
                  </span>
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
