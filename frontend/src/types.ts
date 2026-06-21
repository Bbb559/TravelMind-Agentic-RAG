export type SourceRef = {
  source_type: string;
  source_path: string | null;
  title: string | null;
  metadata: Record<string, unknown>;
};

export type RetrieverResult = {
  content: string;
  source_type: string;
  source_path: string | null;
  title: string | null;
  score: number | null;
  metadata: Record<string, unknown>;
  retriever_name: string;
};

export type RouteDecision = {
  query: string;
  route: string;
  confidence: string;
  reason: string;
  query_type: string;
  entities: string[];
  matched_terms: string[];
  run_id?: string;
};

export type ExecutionStatus = {
  agent: string | null;
  retrieval_mode: string;
  evidence_status: 'not_run' | 'sufficient' | 'partial' | 'insufficient';
  generation_mode: 'none' | 'template' | 'llm' | 'official_response';
  llm_stages: {
    router: 'disabled' | 'not_needed' | 'executed' | 'fallback';
    grade: 'disabled' | 'not_needed' | 'executed' | 'fallback';
    rewrite: 'disabled' | 'not_needed' | 'executed' | 'fallback';
    generate: 'disabled' | 'not_needed' | 'executed' | 'fallback';
  };
};

export type HybridBranchStatus = Record<
  'graphrag' | 'multimodal',
  {
    execution: 'completed' | 'timeout' | 'failed';
    evidence_valid: boolean;
    retrieval_modes: string[];
    fallback_reason: string | null;
  }
>;

export type RAGAnswer = {
  answer: string;
  route: string;
  confidence: string;
  sources: SourceRef[];
  retrieved: RetrieverResult[];
  fallback_reason: string | null;
  trace: string[];
  execution_status?: ExecutionStatus;
  hybrid_branch_status?: HybridBranchStatus | null;
  run_id?: string;
  runtime_summary?: {
    llm_enabled: boolean;
    key_present: boolean;
  };
  global_search_status?: {
    requested: boolean;
    service_enabled: boolean;
    effective_allowed: boolean;
    executed: boolean;
    succeeded: boolean;
  };
};
