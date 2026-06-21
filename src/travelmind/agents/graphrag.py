"""GraphRAG 概要/比较/关联子 Agent。"""

from __future__ import annotations

from travelmind.agents.base import BaseRAGAgent
from travelmind.agents.contracts import AgentPromptConfig, AgentToolSpec
from travelmind.answer_output import sanitize_user_answer
from travelmind.retrievers import GraphRAGSearchRetriever
from travelmind.runtime.rag_helpers import build_rag_answer
from travelmind.schemas import GraphState, RAGAnswer


class GraphRAGAgent(BaseRAGAgent):
    agent_name = "graphrag_agent"
    prompt_config = AgentPromptConfig("agents/graphrag_generate_response.md", "agents/graphrag_final_answer.md")
    tool_spec = AgentToolSpec(
        name="national_graphrag_retriever_tool",
        description="检索中国大陆旅游 GraphRAG 官方 Local/Global 索引。",
        input_schema={"query": "string"},
        output_schema={"results": "RetrieverResult[]"},
        boundary=(
            "官方 local_search 是默认正式链路；global_search 仅在服务开关与"
            "单次请求双重授权后优先执行；本地 evidence/wrapper 仅供预览。"
        ),
    )

    def __init__(
        self,
        *args,
        allow_global_search: bool = False,
        graphrag_adapter=None,
        graphrag_local_adapter=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.retriever = GraphRAGSearchRetriever(
            self.settings,
            adapter=graphrag_adapter,
            local_adapter=graphrag_local_adapter,
            allow_global_search=allow_global_search,
        )

    def run(
        self,
        query: str,
        decision,
        inherited_trace: list[str] | None = None,
    ) -> RAGAnswer:
        """每种官方搜索每个请求最多一次；evidence/wrapper 只作预览。"""
        state = GraphState(
            query=query,
            route_decision=decision,
            trace=list(inherited_trace or []),
        )
        state.trace.append(f"agent:{self.agent_name}:start")
        self.generate_response(state)
        self.retrieve_tool(state)

        mode = (
            state.retrieved[0].metadata.get("retrieval_mode")
            if state.retrieved
            else "graphrag_wrapper"
        )
        if mode == "graphrag_local_search":
            state.graded_results = list(state.retrieved)
            state.trace.append("grade:skipped:official_local_response")
            state.trace.append(f"agent:{self.agent_name}:generate_final_answer")
            state.trace.append("generate:official_local_response")
            decision = state.route_decision
            answer = build_rag_answer(
                state,
                decision.route if decision else "graphrag",
                decision.confidence if decision else "medium",
                state.retrieved[0].content,
                None,
            )
        elif mode == "graphrag_global_search":
            state.graded_results = self.grade_search_docs(state)
            answer = self.generate_final_answer(state)
        else:
            for result in state.retrieved:
                result.metadata.update(
                    {
                        "answer_policy": "evidence_preview_only",
                        "usable_for_answer": False,
                    }
                )
            state.graded_results = []
            state.trace.append("grade:skipped:evidence_preview_only")
            answer = self._generate_evidence_preview_answer(state, str(mode))

        state.trace.append(f"agent:{self.agent_name}:end")
        answer.answer = sanitize_user_answer(answer.answer)
        state.answer = answer
        return answer

    def _trace_retrieval_mode(self, state: GraphState) -> None:
        mode = getattr(self.retriever, "last_mode", "graphrag_wrapper")
        reason = getattr(self.retriever, "last_reason", "")
        diagnostics = getattr(self.retriever, "last_diagnostics", {})
        if diagnostics.get("global_search_called") is True:
            state.trace.append("graphrag:global_search_called")
        if diagnostics.get("global_search_succeeded") is True:
            state.trace.append("graphrag:global_search_succeeded")
        if diagnostics.get("global_search_error"):
            state.trace.append(f"graphrag:global_search_failed:{diagnostics['global_search_error']}")
        if diagnostics.get("official_local_called") is True:
            state.trace.append("graphrag:official_local_called")
        if diagnostics.get("official_local_succeeded") is True:
            state.trace.append("graphrag:official_local_succeeded")
        if diagnostics.get("official_local_error"):
            state.trace.append(f"graphrag:official_local_failed:{diagnostics['official_local_error']}")
        gate_reason = diagnostics.get("global_gate_reason")
        if gate_reason in {"global_search_disabled", "request_not_allowed"}:
            state.trace.append(f"graphrag:{gate_reason}")
        if mode == "graphrag_global_search":
            state.trace.append("graphrag:retriever_mode:global_search")
        elif mode == "graphrag_local_search":
            state.trace.append("graphrag:retriever_mode:local_search")
        elif mode == "graphrag_local_evidence":
            state.trace.append(f"graphrag:retriever_mode:local_evidence:{reason or 'global_search_unavailable'}")
        else:
            state.trace.append(f"graphrag:retriever_mode:wrapper_fallback:{reason or 'unknown'}")

    def generate_final_answer(self, state: GraphState) -> RAGAnswer:
        if state.graded_results:
            if not (
                self.settings.llm_enabled
                and self.settings.llm_generate_enabled
                and self.llm_client
            ):
                state.trace.append(f"agent:{self.agent_name}:generate_final_answer")
                state.trace.append("generate:official_global_response")
                decision = state.route_decision
                return build_rag_answer(
                    state,
                    decision.route if decision else "graphrag",
                    decision.confidence if decision else "medium",
                    state.graded_results[0].content,
                    None,
                )
            return super().generate_final_answer(state)

        state.trace.append(f"agent:{self.agent_name}:generate_final_answer")
        state.trace.append(f"agent:{self.agent_name}:prompt:{self.prompt_config.final_answer}")
        state.trace.append("generate:template")
        metadata = state.retrieved[0].metadata if state.retrieved else {}
        grade_rejected = metadata.get("llm_grade_raw") == "fail"
        reason = str(
            metadata.get("fallback_reason")
            or (
                "graphrag_grade_rejected"
                if grade_rejected
                else "graphrag_low_relevance"
            )
        )
        entities = "、".join(str(item) for item in metadata.get("query_entities", []))
        if reason == "graphrag_grade_rejected":
            answer = (
                "真实 GraphRAG 搜索结果未通过回答质量门禁，"
                "因此不采用该结果生成正式旅游结论。"
            )
        elif reason == "graphrag_low_relevance":
            answer = f"当前 GraphRAG 索引未检索到与“{entities or state.query}”核心实体明确相关的证据，因此不生成具体旅游结论。"
        else:
            answer = (
                f"当前未能执行真实 GraphRAG global_search（{reason}），"
                "且本地索引证据不足以可靠回答，因此不生成具体旅游结论。"
            )
        decision = state.route_decision
        return build_rag_answer(
            state,
            decision.route if decision else "graphrag",
            "low",
            answer,
            reason,
        )

    def _generate_evidence_preview_answer(
        self,
        state: GraphState,
        mode: str,
    ) -> RAGAnswer:
        state.trace.append(f"agent:{self.agent_name}:generate_final_answer")
        state.trace.append("generate:skipped:evidence_preview_only")
        metadata = state.retrieved[0].metadata if state.retrieved else {}
        reason = str(metadata.get("fallback_reason") or "graphrag_low_relevance")
        if mode == "graphrag_local_evidence":
            answer = (
                "官方 GraphRAG Local Search 未成功，以下仅为本地 GraphRAG 产物证据预览，"
                "不代表官方 Local Search 正式回答。"
            )
        else:
            answer = (
                "官方 GraphRAG Local Search 未成功，且未找到可用的本地证据，"
                "因此无法生成正式结论。"
            )

        decision = state.route_decision
        return build_rag_answer(
            state,
            decision.route if decision else "graphrag",
            "low",
            answer,
            reason,
        )
