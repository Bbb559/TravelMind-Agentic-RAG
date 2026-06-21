import { useMemo, useState } from 'react';
import { routeQuery, runWorkflow } from './api';
import {
  extractAgent,
  generationModeLabel,
  retrievalModeLabel,
  shouldShowGlobalSearchNotice,
} from './presentation';
import type {
  HybridBranchStatus,
  RAGAnswer,
  RouteDecision,
  RetrieverResult,
  SourceRef,
} from './types';

const samples = [
  '大理到双廊怎么去？',
  '从上海出发有哪些景点适合周末去？',
  '香港迪士尼怎么玩？',
  '台湾和西安哪个更适合亲子游？',
];

type ResultSnapshot = {
  kind: 'route' | 'workflow';
  submittedQuery: string;
  submittedAllowGlobalSearch: boolean;
  route: RouteDecision;
  answer: RAGAnswer | null;
};

function App() {
  const [query, setQuery] = useState(samples[0]);
  const [result, setResult] = useState<ResultSnapshot | null>(null);
  const [allowGlobalSearch, setAllowGlobalSearch] = useState(false);
  const [loading, setLoading] = useState<'route' | 'workflow' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const route = result?.route ?? null;
  const answer = result?.answer ?? null;
  const summary = useMemo(() => summarize(route, answer), [route, answer]);
  const topSources = (answer?.sources ?? []).slice(0, 3);
  const topRetrieved = (answer?.retrieved ?? []).slice(0, 3);
  const canSubmit = isMeaningfulQuery(query);

  async function handleRoute() {
    if (!isMeaningfulQuery(query)) {
      setError('请先输入具体旅游问题。');
      setResult(null);
      return;
    }
    setLoading('route');
    setError(null);
    try {
      const submittedQuery = query.trim();
      const payload = await routeQuery(submittedQuery);
      setResult({
        kind: 'route',
        submittedQuery,
        submittedAllowGlobalSearch: false,
        route: payload,
        answer: null,
      });
    } catch (err) {
      setError(toChineseError(err));
    } finally {
      setLoading(null);
    }
  }

  async function handleWorkflow() {
    if (!isMeaningfulQuery(query)) {
      setError('请先输入具体旅游问题。');
      setResult(null);
      return;
    }
    setLoading('workflow');
    setError(null);
    try {
      const submittedQuery = query.trim();
      const submittedAllowGlobalSearch = allowGlobalSearch;
      const payload = await runWorkflow(submittedQuery, submittedAllowGlobalSearch);
      const submittedRoute = {
        query: submittedQuery,
        route: payload.route,
        confidence: payload.confidence,
        reason: '来自完整工作流响应',
        query_type: 'workflow',
        entities: [],
        matched_terms: [],
        run_id: payload.run_id,
      };
      setResult({
        kind: 'workflow',
        submittedQuery,
        submittedAllowGlobalSearch,
        route: submittedRoute,
        answer: payload,
      });
    } catch (err) {
      setError(toChineseError(err));
    } finally {
      setLoading(null);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">TravelMind</p>
          <h1>TravelMind 智能旅游 Agent 工作台</h1>
          <p className="subtitle">SystemAgent 路由 + Naive / GraphRAG / Multimodal 子 Agent 的轻量演示</p>
        </div>
        <div className="status">Agentic RAG MVP</div>
      </header>

      <section className="query-panel">
        <label htmlFor="query">用户问题</label>
        <textarea
          id="query"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setResult(null);
            setError(null);
          }}
        />
        <div className="sample-row">
          {samples.map((sample) => (
            <button
              className="sample"
              key={sample}
              onClick={() => {
                setQuery(sample);
                setResult(null);
                setError(null);
              }}
            >
              {sample}
            </button>
          ))}
        </div>
        <label className="cost-toggle">
          <input
            type="checkbox"
            checked={allowGlobalSearch}
            disabled={loading !== null}
            onChange={(event) => setAllowGlobalSearch(event.target.checked)}
          />
          <span>
            <strong>启用高成本 GraphRAG Global Search</strong>
            <small>
              关闭时使用默认的官方 GraphRAG Local Search，仍会调用远程 chat/embedding；
              开启且服务许可后优先执行成本更高的 Global Search。
            </small>
          </span>
        </label>
        <div className="actions">
          <button onClick={handleRoute} disabled={loading !== null || !canSubmit}>
            {loading === 'route' ? '路由中...' : '仅测试路由'}
          </button>
          <button className="primary" onClick={handleWorkflow} disabled={loading !== null || !canSubmit}>
            {loading === 'workflow' ? '运行中...' : '运行完整问答'}
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </section>

      <section className="step-strip" aria-label="执行过程">
        {summary.steps.map((step) => (
          <span className="chip" key={step}>
            {step}
          </span>
        ))}
      </section>

      <section className="grid">
        <MetricCard label="路由结果" value={summary.route} />
        <MetricCard label="命中 Agent" value={summary.agent} />
        <MetricCard label="置信度" value={summary.confidence} />
        <MetricCard label="检索模式" value={summary.retrievalMode} />
        <div className="metric global-search-metric">
          <span>Global Search</span>
          <div className="global-search-status-list">
            {summary.globalSearchStatuses.map((item) => (
              <div key={item.label}>
                <b>{item.label}</b>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </div>
        <div className="metric llm-metric">
          <span>LLM 状态</span>
          <div className="llm-status-list">
            {summary.llmStatuses.map((item) => (
              <div key={item.label}>
                <b>{item.label}</b>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </div>
        <MetricCard label="兜底原因" value={summary.fallbackReason} />
      </section>

      {summary.globalSearchNotice && (
        <section className="global-search-notice" aria-live="polite">
          {summary.globalSearchNotice}
        </section>
      )}

      {summary.hybridBranches && (
        <HybridBranchPanel branches={summary.hybridBranches} />
      )}

      {result?.kind === 'workflow' && (
        <>
          <section className="panel answer-panel">
            <h2>最终答案</h2>
            <p>{answer?.answer}</p>
          </section>

          <section className="columns">
            <SourcesPanel sources={topSources} />
            <RetrievedPanel items={topRetrieved} />
          </section>
        </>
      )}

      <details className="panel debug-panel">
        <summary>调试信息</summary>
        <div className="debug-content">
          <p>
            <strong>run_id：</strong>
            <code>{summary.runId}</code>
          </p>
          <p className="muted">
            调试信息仅展示安全摘要；如需保存完整本地运行日志，可显式开启
            TRAVELMIND_RUN_LOG_ENABLED。
          </p>
          <div className="debug-chips">
            {summary.debugChips.map((chip) => (
              <span className="chip subtle" key={chip}>
                {chip}
              </span>
            ))}
          </div>
        </div>
      </details>
    </main>
  );
}

function HybridBranchPanel({ branches }: { branches: HybridBranchStatus }) {
  return (
    <section className="panel hybrid-branches">
      <h2>Hybrid 子分支状态</h2>
      <div className="hybrid-branch-grid">
        {Object.entries(branches).map(([name, branch]) => (
          <div key={name}>
            <strong>{name === 'graphrag' ? 'GraphRAG' : 'Multimodal'}</strong>
            <span>执行：{branchExecutionLabel(branch.execution)}</span>
            <span>证据：{branch.evidence_valid ? '有效' : '无效'}</span>
            <span>
              模式：
              {branch.retrieval_modes.length
                ? branch.retrieval_modes.map(retrievalModeLabel).join(' / ')
                : '无'}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SourcesPanel({ sources }: { sources: SourceRef[] }) {
  return (
    <section className="panel">
      <h2>来源 Top 3</h2>
      {sources.length ? (
        <div className="source-table">
          {sources.map((source, index) => (
            <div className="source-row" key={`${source.source_path}-${index}`}>
              <strong>{source.title ?? source.source_path ?? '未命名来源'}</strong>
              <span>{source.source_type}</span>
              <code>{source.source_path ?? '-'}</code>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">暂无来源</p>
      )}
    </section>
  );
}

function RetrievedPanel({ items }: { items: RetrieverResult[] }) {
  return (
    <section className="panel">
      <h2>检索片段 Top 3</h2>
      {items.length ? (
        <div className="retrieved-list">
          {items.map((item, index) => (
            <article className="retrieved-card" key={`${item.source_path}-${index}`}>
              <div className="retrieved-title">
                <strong>{item.title ?? '未命名片段'}</strong>
                <span>{retrievalModeLabel(String(item.metadata?.retrieval_mode ?? item.source_type))}</span>
              </div>
              <p>{preview(item.content)}</p>
              <div className="meta-line">
                <span>score：{item.score ?? '无'}</span>
                <span>{item.retriever_name}</span>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <p className="muted">暂无检索片段</p>
      )}
    </section>
  );
}

function summarize(route: RouteDecision | null, answer: RAGAnswer | null) {
  const trace = answer?.trace ?? [];
  const routeValue = route?.route ?? answer?.route ?? '-';
  const confidence = route?.confidence ?? answer?.confidence ?? '-';
  const firstRetrieved = answer?.retrieved?.[0];
  const retrievalMode = String(
    answer?.execution_status?.retrieval_mode
      ?? firstRetrieved?.metadata?.retrieval_mode
      ?? '-',
  );
  const graphRetrievalMode = findGraphRetrievalMode(
    answer?.retrieved ?? [],
    answer?.hybrid_branch_status?.graphrag.retrieval_modes ?? [],
  );
  const globalSearchStatus = answer?.global_search_status;
  const agent = answer?.execution_status?.agent ?? extractAgent(trace, routeValue);
  const llmStages = answer?.execution_status?.llm_stages;
  const generationMode = answer?.execution_status?.generation_mode ?? 'none';
  const llmStatuses = [
    { label: 'Router', value: llmStageLabel(llmStages?.router) },
    { label: 'Grade', value: llmStageLabel(llmStages?.grade) },
    { label: 'Rewrite', value: llmStageLabel(llmStages?.rewrite) },
    { label: 'Generate', value: generationModeLabel(generationMode) },
  ];
  const globalSearchStatuses = [
    {
      label: '本次请求',
      value: booleanStatus(globalSearchStatus?.requested),
    },
    {
      label: '服务许可',
      value: booleanStatus(globalSearchStatus?.service_enabled),
    },
    {
      label: '实际模式',
      value: retrievalModeLabel(graphRetrievalMode),
    },
  ];
  const steps = [
    '输入问题',
    'SystemAgent 路由',
    agent,
    retrievalMode === '-' ? '检索待运行' : `${retrievalModeLabel(retrievalMode)}检索`,
  ];
  if (hasTrace(trace, 'grade:llm')) steps.push('LLM Grade');
  if (hasTrace(trace, 'rewrite:llm')) steps.push('LLM Rewrite');
  if (hasTrace(trace, 'generate:llm')) steps.push('LLM Generate');

  return {
    route: routeValue,
    agent,
    confidence,
    retrievalMode: retrievalModeLabel(retrievalMode),
    globalSearchStatuses,
    globalSearchNotice: shouldShowGlobalSearchNotice(routeValue)
      ? buildGlobalSearchNotice(
          globalSearchStatus,
          graphRetrievalMode,
          routeValue,
          answer?.hybrid_branch_status ?? null,
        )
      : null,
    llmStatuses,
    fallbackReason: answer?.fallback_reason ?? '无',
    hybridBranches: answer?.hybrid_branch_status ?? null,
    runId: answer?.run_id ?? route?.run_id ?? '暂无',
    steps,
    debugChips: trace
      .filter((item) =>
        ['system:route_source', 'agent:', 'retrieve:', 'grade:', 'rewrite:', 'generate:'].some((key) =>
          item.includes(key),
        ),
      )
      .slice(0, 12),
  };
}

function buildGlobalSearchNotice(
  status: RAGAnswer['global_search_status'] | undefined,
  graphMode: string,
  route: string,
  hybridBranches: HybridBranchStatus | null,
): string | null {
  if (!status) return null;
  if (route === 'hybrid_rag' && hybridBranches) {
    const graph = hybridBranches.graphrag;
    const multimodal = hybridBranches.multimodal;
    const graphExecution = branchExecutionLabel(graph.execution);
    const multimodalSummary = multimodal.evidence_valid
      ? 'Multimodal 子分支仍提供了有效离线证据。'
      : 'Multimodal 子分支也未提供有效证据。';
    if (!status.requested) {
      return graph.evidence_valid
        ? `Hybrid 的 GraphRAG 子分支采用${retrievalModeLabel(graphMode)}；${multimodalSummary}`
        : `Hybrid 的 GraphRAG 子分支${graphExecution}但未提供正式证据；${multimodalSummary}`;
    }
    if (!status.service_enabled) {
      return `本次请求已开启 Global Search，但服务级未允许；Hybrid 的 GraphRAG 子分支${graphExecution}，${multimodalSummary}`;
    }
    if (status.executed && status.succeeded && graphMode === 'graphrag_global_search') {
      return `Hybrid 的真实 GraphRAG Global Search 已执行成功；${multimodalSummary}`;
    }
    return `Hybrid 已尝试 GraphRAG Global Search，但 GraphRAG 子分支${graphExecution}且未采用正式结果；${multimodalSummary}`;
  }
  if (!status.requested) {
    if (graphMode === 'graphrag_local_search') {
      return '本次未请求 Global Search，已采用默认的官方 GraphRAG Local Search 正式回答。Local Search 仍会使用远程 chat/embedding。';
    }
    return `本次未请求 Global Search；官方 Local Search 未成功，当前已降级为${retrievalModeLabel(graphMode)}。`;
  }
  if (!status.service_enabled) {
    return `本次请求已开启 Global Search，但服务级未允许，实际未执行 Global Search；当前模式为${retrievalModeLabel(graphMode)}。`;
  }
  if (
    status.effective_allowed &&
    status.executed &&
    status.succeeded &&
    graphMode === 'graphrag_global_search'
  ) {
    return '真实 GraphRAG Global Search 已执行成功，本次正式结论采用 Global Search 结果。';
  }
  if (status.effective_allowed && status.executed) {
    return `已尝试真实 GraphRAG Global Search，但未采用其结果，现已降级为${retrievalModeLabel(graphMode)}。`;
  }
  if (status.effective_allowed) {
    return `本次策略允许 Global Search，但实际未执行；当前模式为${retrievalModeLabel(graphMode)}。`;
  }
  return `本次请求已开启，但当前路由不需要 GraphRAG Global Search；实际模式为${retrievalModeLabel(graphMode)}。`;
}

function findGraphRetrievalMode(
  items: RetrieverResult[],
  branchModes: string[] = [],
): string {
  const modes = [
    ...items.map((item) => String(item.metadata?.retrieval_mode ?? '')),
    ...branchModes,
  ];
  return (
    modes.find((mode) =>
      [
        'graphrag_local_search',
        'graphrag_local_evidence',
        'graphrag_global_search',
        'graphrag_wrapper',
      ].includes(mode),
    ) ?? '不适用'
  );
}

function booleanStatus(value: boolean | undefined): string {
  if (value === undefined) return '未提交';
  return value ? '已开启' : '已关闭';
}

function llmStageLabel(stage: string | undefined): string {
  return {
    disabled: '未启用',
    not_needed: '未需要',
    executed: '已执行',
    fallback: '已回退',
  }[stage ?? ''] ?? '未提交';
}

function hasTrace(trace: string[], token: string): boolean {
  return trace.some((item) => item.includes(token));
}

function preview(content: string): string {
  return content.length > 200 ? `${content.slice(0, 200)}...` : content;
}

function branchExecutionLabel(execution: string): string {
  return {
    completed: '已完成',
    timeout: '超时',
    failed: '失败',
  }[execution] ?? execution;
}

function isMeaningfulQuery(value: string): boolean {
  const compact = value.replace(/\s+/g, '').toLowerCase();
  if (!compact) return false;
  if (['你好', '您好', '嗨', '哈喽', 'hello', 'hi', '在吗'].includes(compact)) {
    return false;
  }
  if (/^[\p{P}\p{S}]+$/u.test(compact)) return false;
  if (/^[a-z0-9_]{4,}$/.test(compact)) return false;
  return true;
}

function toChineseError(err: unknown): string {
  if (err instanceof TypeError) {
    return '请求失败：无法连接后端服务，请确认 FastAPI 已在 127.0.0.1:8000 启动，或检查 Vite 代理配置。';
  }
  return err instanceof Error ? err.message : '请求失败：未知错误';
}

export default App;
