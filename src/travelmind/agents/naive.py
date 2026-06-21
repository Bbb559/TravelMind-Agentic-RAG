"""大陆详细问答子 Agent。"""

from __future__ import annotations

from travelmind.agents.base import BaseRAGAgent
from travelmind.agents.contracts import AgentPromptConfig, AgentToolSpec
from travelmind.retrievers import NaiveAutoTravelRetriever
from travelmind.runtime.rag_helpers import parse_json_object
from travelmind.schemas import GraphState


class NaiveTravelAgent(BaseRAGAgent):
    agent_name = "naive_travel_agent"
    prompt_config = AgentPromptConfig("agents/naive_generate_response.md", "agents/naive_final_answer.md")
    tool_spec = AgentToolSpec(
        name="national_retriever_tool",
        description="检索中国大陆旅游 CSV/FAISS 资料。",
        input_schema={"query": "string"},
        output_schema={"results": "RetrieverResult[]"},
        boundary="当前只使用 assets 下的大陆旅游知识库。",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.retriever = NaiveAutoTravelRetriever(self.settings)

    def generate_response(self, state: GraphState) -> None:
        super().generate_response(state)
        if not (
            self.settings.llm_enabled
            and self.settings.naive_agent_llm_loop_enabled
            and self.llm_client
        ):
            state.trace.append("agent:naive_travel_agent:generate_response:deterministic")
            return
        try:
            prompt = (
                "你是 NaiveTravelAgent generate_response，只输出 JSON："
                '{"need_retrieve": true, "tool_name": "national_retriever_tool", "search_query": "...", "reason": "..."}'
                f"\n用户问题：{state.query}"
            )
            data = parse_json_object(self.llm_client.generate(prompt))
            search_query = str(data.get("search_query") or "").strip()
            if data.get("tool_name") != self.tool_spec.name or data.get("need_retrieve") is not True:
                raise ValueError("invalid_tool_decision")
            if not self._safe_tool_query(state.query, search_query):
                state.trace.append("agent:naive_travel_agent:generate_response:rejected_unsafe")
                return
            state.tool_query = search_query
            state.trace.append("agent:naive_travel_agent:generate_response:llm")
        except Exception:
            state.trace.append("agent:naive_travel_agent:generate_response:fallback_deterministic")

    def _trace_retrieval_mode(self, state: GraphState) -> None:
        mode = getattr(self.retriever, "last_mode", "csv")
        reason = getattr(self.retriever, "last_reason", "")
        if mode == "faiss":
            state.trace.append("naive:retriever_mode:faiss")
        elif reason:
            state.trace.append(f"naive:retriever_mode:faiss_fallback_csv:{reason}")
        else:
            state.trace.append("naive:retriever_mode:csv")

    def _safe_tool_query(self, original: str, candidate: str) -> bool:
        if not candidate or len(candidate) > max(80, len(original) * 2):
            return False
        known = ["北京", "上海", "成都", "西安", "南京", "大理", "双廊", "丽江", "云南", "香港", "澳门", "台湾"]
        original_places = {place for place in known if place in original}
        candidate_places = {place for place in known if place in candidate}
        return not (candidate_places - original_places and original_places)
