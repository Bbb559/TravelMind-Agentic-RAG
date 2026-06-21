import type { RAGAnswer, RouteDecision } from './types';

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() ?? '';
const API_BASE_URL = configuredBaseUrl.replace(/\/$/, '');

function createRunId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `travelmind-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function postJson<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const runId = createRunId();
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-TravelMind-Run-Id': runId,
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`请求失败：HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function routeQuery(query: string): Promise<RouteDecision> {
  return postJson<RouteDecision>('/api/route', { query });
}

export function runWorkflow(query: string, allowGlobalSearch = false): Promise<RAGAnswer> {
  return postJson<RAGAnswer>('/api/workflow', {
    query,
    allow_global_search: allowGlobalSearch,
  });
}
