"""Agentic RAG 兼容 facade。

默认 deterministic，可通过环境变量开启 LLM router / grade / rewrite / generate。
"""

from __future__ import annotations

from travelmind.agents import SystemAgent
from travelmind.config import ProjectSettings, get_settings
from travelmind.llm import LLMClientProtocol
from travelmind.schemas import RAGAnswer, RouteDecision


class AgenticRAGWorkflow:
    """对 CLI/API/测试保持稳定的 workflow facade。"""

    def __init__(self, settings: ProjectSettings | None = None, llm_client: LLMClientProtocol | None = None) -> None:
        self.settings = settings or get_settings()
        self.system_agent = SystemAgent(self.settings, llm_client)

    def run(self, query: str, *, allow_global_search: bool = False) -> RAGAnswer:
        return self.system_agent.run(
            query,
            allow_global_search=allow_global_search,
        )

    def route(self, query: str) -> RouteDecision:
        return self.system_agent.route(query)[0]

    def build_graph(self):  # pragma: no cover
        return None
