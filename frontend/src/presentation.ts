export function extractAgent(trace: string[], route: string): string {
  const routeAgents: Record<string, string> = {
    naive_rag: 'NaiveTravelAgent',
    graphrag: 'GraphRAGAgent',
    multimodal_rag: 'MultimodalTravelAgent',
    hybrid_rag: 'HybridAggregator',
  };
  if (routeAgents[route]) return routeAgents[route];
  const agentTrace = trace.find((item) => item.includes('agent:') && item.includes(':start'));
  if (agentTrace?.includes('naive_travel_agent')) return 'NaiveTravelAgent';
  if (agentTrace?.includes('graphrag_agent')) return 'GraphRAGAgent';
  if (agentTrace?.includes('multimodal_travel_agent')) return 'MultimodalTravelAgent';
  return '-';
}

export function shouldShowGlobalSearchNotice(route: string): boolean {
  return route === 'graphrag' || route === 'hybrid_rag';
}

export function retrievalModeLabel(mode: string): string {
  const labels: Record<string, string> = {
    graphrag_local_search: '官方 GraphRAG Local Search',
    graphrag_global_search: '真实 GraphRAG Global Search',
    graphrag_local_evidence: '本地证据预览',
    graphrag_wrapper: '安全兜底模式',
    multi_source_candidate_aggregation: '多源候选聚合',
    none: '无',
  };
  if (mode === '-') return '-';
  if (mode === '不适用') return mode;
  return labels[mode] ?? mode;
}

export function generationModeLabel(mode: string): string {
  return {
    llm: 'LLM 生成',
    template: '模板生成',
    official_response: '官方响应',
    none: '未生成',
  }[mode] ?? mode;
}
