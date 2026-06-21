"""港澳台离线 Markdown 子 Agent。"""

from __future__ import annotations

from travelmind.agents.base import BaseRAGAgent
from travelmind.agents.contracts import AgentPromptConfig, AgentToolSpec
from travelmind.retrievers import MultimodalVectorMarkdownRetriever
from travelmind.runtime.rag_helpers import build_rag_answer
from travelmind.schemas import GraphState, RAGAnswer


class MultimodalTravelAgent(BaseRAGAgent):
    agent_name = "multimodal_travel_agent"
    prompt_config = AgentPromptConfig("agents/multimodal_generate_response.md", "agents/multimodal_final_answer.md")
    tool_spec = AgentToolSpec(
        name="gang_ao_tai_retriever_tool",
        description="检索港澳台离线 Markdown 旅游资料。",
        input_schema={"query": "string"},
        output_schema={"results": "RetrieverResult[]"},
        boundary="当前在线问答只消费离线 Markdown；不启用 OCR，不启用在线 VLM。",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.retriever = MultimodalVectorMarkdownRetriever(self.settings)

    def _trace_retrieval_mode(self, state: GraphState) -> None:
        mode = getattr(self.retriever, "last_mode", "markdown_keyword")
        reason = getattr(self.retriever, "last_reason", "")
        if mode == "markdown_vector":
            state.trace.append("multimodal:retriever_mode:markdown_vector")
        elif reason:
            state.trace.append(f"multimodal:retriever_mode:vector_fallback_markdown:{reason}")
        else:
            state.trace.append("multimodal:retriever_mode:markdown_keyword")

    def generate_final_answer(self, state: GraphState) -> RAGAnswer:
        if state.graded_results:
            return super().generate_final_answer(state)
        state.trace.append(f"agent:{self.agent_name}:generate_final_answer")
        state.trace.append("generate:skipped:no_relevant_evidence")
        decision = state.route_decision
        return build_rag_answer(
            state,
            decision.route if decision else "multimodal_rag",
            "low",
            "离线资料中没有找到足够证据，无法生成具体旅游建议。",
            getattr(self.retriever, "last_evidence_reason", "")
            or "no_relevant_evidence",
        )
