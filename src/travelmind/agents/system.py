"""SystemAgent 上层分流与子 Agent 编排。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Any

from travelmind.answer_output import sanitize_user_answer
from travelmind.agents.graphrag import GraphRAGAgent
from travelmind.agents.multimodal import MultimodalTravelAgent
from travelmind.agents.naive import NaiveTravelAgent
from travelmind.agents.router import SystemRouter
from travelmind.config import ProjectSettings, get_settings
from travelmind.input_guard import is_invalid_input
from travelmind.llm import LLMClientProtocol, OpenAICompatibleClient
from travelmind.runtime.rag_helpers import generate_answer_text
from travelmind.schemas import RAGAnswer, RouteDecision


class SystemAgent:
    """负责上层 route，并将问题交给对应子 Agent。"""

    def __init__(
        self,
        settings: ProjectSettings | None = None,
        llm_client: LLMClientProtocol | None = None,
        *,
        graphrag_adapter=None,
        graphrag_local_adapter=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or OpenAICompatibleClient(self.settings)
        self.graphrag_adapter = graphrag_adapter
        self.graphrag_local_adapter = graphrag_local_adapter
        self.router = SystemRouter(self.settings, self.llm_client)

    def route(self, query: str) -> tuple[RouteDecision, list[str]]:
        if is_invalid_input(query):
            return (
                RouteDecision(
                    query=query,
                    route="invalid_input",
                    confidence="low",
                    reason="输入为空白、纯标点、无语义问候或随机乱码。",
                    query_type="invalid_input",
                ),
                ["workflow:start", "input:invalid"],
            )
        decision = self.router.route(query)
        trace = ["workflow:start"]
        if self.router.last_route_source == "llm":
            trace.append("system:route_source:llm")
        elif self.router.last_route_source == "rule_fallback":
            trace.append(f"system:route_source:rule_fallback:{self.router.last_fallback_reason}")
        else:
            trace.append("system:route_source:rule")
        trace.extend([f"system:route:{decision.route}", f"route:{decision.route}", f"query_type:{decision.query_type}"])
        return decision, trace

    def run(self, query: str, *, allow_global_search: bool = False) -> RAGAnswer:
        decision, trace = self.route(query)
        if decision.route == "invalid_input":
            answer = RAGAnswer(
                answer="请先输入具体旅游问题。",
                route="invalid_input",
                confidence="low",
                sources=[],
                retrieved=[],
                fallback_reason="invalid_input",
                trace=trace,
            )
        elif decision.route == "naive_rag":
            answer = NaiveTravelAgent(self.settings, self.llm_client).run(query, decision, trace)
        elif decision.route == "graphrag":
            answer = GraphRAGAgent(
                self.settings,
                self.llm_client,
                allow_global_search=allow_global_search,
                graphrag_adapter=self.graphrag_adapter,
                graphrag_local_adapter=self.graphrag_local_adapter,
            ).run(query, decision, trace)
        elif decision.route == "multimodal_rag":
            answer = MultimodalTravelAgent(self.settings, self.llm_client).run(query, decision, trace)
        elif decision.route == "hybrid_rag":
            answer = self._run_hybrid(
                query,
                decision,
                trace,
                allow_global_search=allow_global_search,
            )
        else:
            answer = RAGAnswer(
                answer="当前问题不属于 TravelMind 已接入能力，无法可靠回答。",
                route="fallback",
                confidence="low",
                sources=[],
                retrieved=[],
                fallback_reason="unsupported_query",
                trace=trace + ["fallback:unsupported_query"],
            )
        return self._finalize_answer(answer)

    def _run_hybrid(
        self,
        query: str,
        decision: RouteDecision,
        trace: list[str],
        *,
        allow_global_search: bool,
    ) -> RAGAnswer:
        branch_timeout = max(0, self.settings.hybrid_branch_timeout_seconds)
        hybrid_trace = list(trace) + ["agent:hybrid_aggregator:start"]
        graph_search_attempts = (
            2
            if (
                allow_global_search
                and self.settings.graphrag_global_search_enabled
            )
            else 1
        )
        per_search_timeout = branch_timeout / graph_search_attempts
        graph_settings = replace(
            self.settings,
            graphrag_timeout_seconds=min(
                self.settings.graphrag_timeout_seconds,
                per_search_timeout,
            ),
        )

        executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="travelmind-hybrid",
        )
        futures: dict[str, Future[RAGAnswer]] = {
            "graphrag": executor.submit(
                GraphRAGAgent(
                    graph_settings,
                    self.llm_client,
                    allow_global_search=allow_global_search,
                    graphrag_adapter=self.graphrag_adapter,
                    graphrag_local_adapter=self.graphrag_local_adapter,
                ).run,
                query,
                decision,
                list(hybrid_trace),
            ),
            "multimodal": executor.submit(
                MultimodalTravelAgent(self.settings, self.llm_client).run,
                query,
                decision,
                list(hybrid_trace),
            ),
        }
        done, _ = wait(futures.values(), timeout=branch_timeout)
        branch_answers: dict[str, RAGAnswer] = {}
        branch_statuses: dict[str, str] = {}
        for branch, future in futures.items():
            if future not in done:
                future.cancel()
                branch_statuses[branch] = "timeout"
                continue
            try:
                branch_answers[branch] = future.result()
                branch_statuses[branch] = "completed"
            except Exception:
                branch_statuses[branch] = "failed"
        executor.shutdown(wait=False, cancel_futures=True)

        final_trace = list(hybrid_trace)
        for branch in ("graphrag", "multimodal"):
            answer = branch_answers.get(branch)
            if answer is not None:
                _merge_trace(final_trace, answer.trace)
            final_trace.append(
                f"hybrid:branch:{branch}:{branch_statuses[branch]}"
            )

        graph_answer = branch_answers.get("graphrag")
        multimodal_answer = branch_answers.get("multimodal")
        graph_candidates = graph_answer.retrieved if graph_answer is not None else []
        multimodal_candidates = (
            multimodal_answer.retrieved
            if multimodal_answer is not None
            else []
        )
        graph_results = [
            item for item in graph_candidates if _result_is_valid(item)
        ]
        multimodal_results = [
            item for item in multimodal_candidates if _result_is_valid(item)
        ]
        retrieved = graph_results + multimodal_results
        sources = [item.to_source() for item in retrieved]
        final_trace.append("hybrid:multi_source_candidate_aggregation")
        graph_preview_results = [
            item
            for item in graph_candidates
            if item.metadata.get("answer_policy") == "evidence_preview_only"
        ]
        if graph_preview_results:
            final_trace.append("hybrid:graphrag_evidence_preview_only")
        valid_branches = [
            branch
            for branch, results in (
                ("graphrag", graph_results),
                ("multimodal", multimodal_results),
            )
            if results
        ]
        if len(valid_branches) == 2:
            answer_text = generate_answer_text(
                query,
                "hybrid_rag",
                "low",
                retrieved,
                final_trace,
                self.settings.llm_enabled and self.settings.llm_generate_enabled,
                self.llm_client,
                "generation_prompt.md",
                "system_agent",
                {"boundary": "hybrid_rag 当前只做多源候选聚合，不声明深度融合完成"},
            )
            fallback_reason = None
        elif len(valid_branches) == 1:
            final_trace.append("hybrid:partial_fallback")
            branch = valid_branches[0]
            branch_answer = (
                graph_answer
                if branch == "graphrag"
                else multimodal_answer
            )
            branch_label = (
                "GraphRAG 正式检索"
                if branch == "graphrag"
                else "Multimodal 离线资料"
            )
            answer_text = (
                f"当前仅基于已命中的 {branch_label} 回答，另一分支未提供有效证据。\n\n"
                f"{branch_answer.answer if branch_answer is not None else ''}"
            ).strip()
            fallback_reason = "hybrid_partial_fallback"
        else:
            answer_text = "当前各检索分支均未找到足够证据，无法生成具体旅游建议。"
            fallback_reason = "hybrid_no_usable_results"
            final_trace.append("generate:skipped:no_relevant_evidence")
        if (
            "深度融合" in answer_text
            and "不" not in answer_text
            and "未" not in answer_text
        ):
            answer_text = "当前仅完成多源候选聚合，尚未声明深度融合完成。"
        hybrid_branch_status = {
            "graphrag": _branch_status(
                branch_statuses["graphrag"],
                graph_answer,
                graph_candidates,
                bool(graph_results),
            ),
            "multimodal": _branch_status(
                branch_statuses["multimodal"],
                multimodal_answer,
                multimodal_candidates,
                bool(multimodal_results),
            ),
        }
        final_trace.append("agent:hybrid_aggregator:end")
        return RAGAnswer(
            answer=answer_text,
            route="hybrid_rag",
            confidence="low",
            sources=sources,
            retrieved=retrieved,
            fallback_reason=fallback_reason,
            trace=final_trace,
            hybrid_branch_status=hybrid_branch_status,
        )

    def _finalize_answer(self, answer: RAGAnswer) -> RAGAnswer:
        answer.answer = sanitize_user_answer(answer.answer)
        answer.execution_status = _execution_status(answer, self.settings)
        return answer


def _merge_trace(target: list[str], source: list[str]) -> None:
    for step in source:
        if step not in target:
            target.append(step)


def _result_is_valid(result) -> bool:
    explicit = result.metadata.get("evidence_valid")
    if isinstance(explicit, bool):
        return explicit
    if result.metadata.get("answer_policy") == "evidence_preview_only":
        return False
    mode = result.metadata.get("retrieval_mode")
    if mode in {"graphrag_local_search", "graphrag_global_search"}:
        return (
            result.metadata.get("graphrag_relevance") is True
            and result.metadata.get("grade") != "fail"
        )
    return bool(
        result.metadata.get("usable_for_answer")
        or result.metadata.get("grade") == "pass"
    )


def _branch_status(
    execution: str,
    answer: RAGAnswer | None,
    candidates: list,
    evidence_valid: bool,
) -> dict[str, Any]:
    return {
        "execution": execution,
        "evidence_valid": evidence_valid,
        "retrieval_modes": [
            str(item.metadata.get("retrieval_mode"))
            for item in candidates
            if item.metadata.get("retrieval_mode")
        ],
        "fallback_reason": answer.fallback_reason if answer is not None else execution,
    }


def _execution_status(
    answer: RAGAnswer,
    settings: ProjectSettings,
) -> dict[str, Any]:
    agent = {
        "invalid_input": None,
        "naive_rag": "NaiveTravelAgent",
        "graphrag": "GraphRAGAgent",
        "multimodal_rag": "MultimodalTravelAgent",
        "hybrid_rag": "HybridAggregator",
    }.get(answer.route)
    if answer.route == "invalid_input":
        retrieval_mode = "none"
        evidence_status = "not_run"
    elif answer.route == "hybrid_rag":
        retrieval_mode = "multi_source_candidate_aggregation"
        valid_count = sum(
            1
            for item in (answer.hybrid_branch_status or {}).values()
            if item.get("evidence_valid") is True
        )
        evidence_status = (
            "sufficient"
            if valid_count == 2
            else "partial"
            if valid_count == 1
            else "insufficient"
        )
    else:
        retrieval_mode = (
            str(answer.retrieved[0].metadata.get("retrieval_mode"))
            if answer.retrieved
            else "none"
        )
        evidence_status = (
            "sufficient"
            if any(_result_is_valid(item) for item in answer.retrieved)
            else "insufficient"
        )
    trace = answer.trace
    if (
        "generate:official_local_response" in trace
        or "generate:official_global_response" in trace
    ):
        generation_mode = "official_response"
    elif "generate:llm" in trace:
        generation_mode = "llm"
    elif "generate:template" in trace:
        generation_mode = "template"
    else:
        generation_mode = "none"
    return {
        "agent": agent,
        "retrieval_mode": retrieval_mode,
        "evidence_status": evidence_status,
        "generation_mode": generation_mode,
        "llm_stages": {
            "router": _llm_stage(
                settings.llm_enabled and settings.system_agent_llm_router_enabled,
                trace,
                "system:route_source:llm",
                "system:route_source:rule_fallback",
            ),
            "grade": _llm_stage(
                settings.llm_enabled and settings.llm_grade_enabled,
                trace,
                "grade:llm:",
                "grade:llm_fallback",
            ),
            "rewrite": _llm_stage(
                settings.llm_enabled and settings.llm_rewrite_enabled,
                trace,
                "rewrite:llm:",
                "rewrite:llm_fallback",
            ),
            "generate": _llm_stage(
                settings.llm_enabled and settings.llm_generate_enabled,
                trace,
                "generate:llm",
                "generate:llm_fallback",
            ),
        },
    }


def _llm_stage(
    enabled: bool,
    trace: list[str],
    executed_token: str,
    fallback_token: str,
) -> str:
    if not enabled:
        return "disabled"
    if any(step.startswith(fallback_token) for step in trace):
        return "fallback"
    if any(step.startswith(executed_token) for step in trace):
        return "executed"
    return "not_needed"
