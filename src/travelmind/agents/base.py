"""子 Agent 闭环基类。"""

from __future__ import annotations

from travelmind.agents.contracts import AgentPromptConfig, AgentToolSpec
from travelmind.config import ProjectSettings
from travelmind.llm import LLMClientProtocol
from travelmind.runtime.rag_helpers import build_rag_answer, generate_answer_text, grade_results, rewrite_query
from travelmind.schemas import GraphState, RAGAnswer, RouteDecision


class BaseRAGAgent:
    """基于 generate_response -> retrieve -> grade -> refine -> final_answer 闭环。"""

    agent_name = "base_agent"
    prompt_config = AgentPromptConfig("generation_prompt.md", "generation_prompt.md")
    tool_spec = AgentToolSpec("base_tool", "base retriever")

    def __init__(self, settings: ProjectSettings, llm_client: LLMClientProtocol | None = None, max_generate_times: int = 1) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self.max_generate_times = max_generate_times

    def run(self, query: str, decision: RouteDecision, inherited_trace: list[str] | None = None) -> RAGAnswer:
        state = GraphState(query=query, route_decision=decision, trace=list(inherited_trace or []))
        state.trace.append(f"agent:{self.agent_name}:start")
        while True:
            self.generate_response(state)
            self.retrieve_tool(state)
            usable = self.grade_search_docs(state)
            if usable or state.retry_count >= self.max_generate_times:
                state.graded_results = usable
                break
            self.refine_query(state)
            state.tool_query = None
            state.retry_count += 1
        answer = self.generate_final_answer(state)
        state.trace.append(f"agent:{self.agent_name}:end")
        state.answer = answer
        return answer

    def generate_response(self, state: GraphState) -> None:
        state.trace.append(f"agent:{self.agent_name}:generate_response")
        state.trace.append(f"agent:{self.agent_name}:prompt:{self.prompt_config.generate_response}")

    def retrieve_tool(self, state: GraphState) -> None:
        query = state.tool_query or state.rewritten_query or state.query
        state.trace.append(f"agent:{self.agent_name}:tool_schema:{self.tool_spec.name}")
        state.trace.append(f"agent:{self.agent_name}:retrieve_tool:{self.tool_spec.name}")
        state.retrieved = self.retriever.retrieve(query)
        state.trace.append(f"retrieve:{self.retriever.__class__.__name__}")
        self._trace_retrieval_mode(state)

    def grade_search_docs(self, state: GraphState):
        state.trace.append(f"agent:{self.agent_name}:grade_search_docs")
        usable = grade_results(
            state.query,
            state.retrieved,
            state.trace,
            self.settings.llm_enabled and self.settings.llm_grade_enabled,
            self.llm_client,
        )
        state.graded_results = usable
        return usable

    def refine_query(self, state: GraphState) -> None:
        state.trace.append(f"agent:{self.agent_name}:refine_query")
        state.rewritten_query = rewrite_query(
            state.query,
            state.trace,
            self.settings.llm_enabled and self.settings.llm_rewrite_enabled,
            self.llm_client,
        )

    def generate_final_answer(self, state: GraphState) -> RAGAnswer:
        state.trace.append(f"agent:{self.agent_name}:generate_final_answer")
        state.trace.append(f"agent:{self.agent_name}:prompt:{self.prompt_config.final_answer}")
        decision = state.route_decision
        results = state.graded_results or []
        confidence = "low" if any(item.metadata.get("confidence_policy") == "low" for item in results) else (decision.confidence if decision else "low")
        fallback_reason = None if results else (
            getattr(self.retriever, "last_evidence_reason", "")
            or "no_usable_results"
        )
        answer_text = generate_answer_text(
            state.query,
            decision.route if decision else "fallback",
            confidence,
            results,
            state.trace,
            self.settings.llm_enabled and self.settings.llm_generate_enabled,
            self.llm_client,
            self.prompt_config.final_answer,
            self.agent_name,
            self.tool_spec.to_dict(),
        )
        return build_rag_answer(state, decision.route if decision else "fallback", confidence, answer_text, fallback_reason)

    def _trace_retrieval_mode(self, state: GraphState) -> None:
        if not state.retrieved:
            return
        mode = state.retrieved[0].metadata.get("retrieval_mode")
        if mode:
            state.trace.append(f"{self.agent_name}:retriever_mode:{mode}")
