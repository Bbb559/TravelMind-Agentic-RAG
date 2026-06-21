"""规则 Router 与可选 LLM Router。"""

from __future__ import annotations

import json
import re
from typing import Any

from travelmind.config import ProjectSettings
from travelmind.destinations import destination_scopes, match_destination_entities
from travelmind.llm import LLMClientProtocol, load_prompt
from travelmind.schemas import RouteDecision

ROUTES = {
    "invalid_input",
    "naive_rag",
    "graphrag",
    "multimodal_rag",
    "hybrid_rag",
    "fallback",
}
CONFIDENCES = {"high", "medium", "low"}
GAT = {"香港", "澳门", "台湾", "台北"}
GRAPH_INTENTS = {
    "对比",
    "比较",
    "哪个更",
    "区别",
    "差异",
    "总结",
    "归纳",
    "概括",
    "关联",
    "共同点",
    "串成",
    "串联",
    "联游",
    "路线",
}


class SupervisorRouter:
    """默认规则路由器。"""

    def route(self, query: str) -> RouteDecision:
        text = query.strip()
        if self._unsupported(text):
            return RouteDecision(text, "fallback", "low", "当前问题不属于已接入的旅游知识库能力。", "unsupported")
        gat_hits = [term for term in GAT if term in text]
        mainland_hits = match_destination_entities(text)
        mainland_scopes = destination_scopes(mainland_hits)
        if gat_hits and mainland_hits:
            return RouteDecision(text, "hybrid_rag", "medium", "跨港澳台与大陆知识源，需要多源候选聚合。", "hybrid", gat_hits + mainland_hits, gat_hits + mainland_hits)
        if gat_hits:
            return RouteDecision(text, "multimodal_rag", "high", "命中港澳台离线 Markdown 知识库。", "gang_ao_tai", gat_hits, gat_hits)
        graph_hits = [term for term in GRAPH_INTENTS if term in text]
        if graph_hits and len(mainland_scopes) > 1:
            query_type = "route_summary" if any(term in text for term in {"串成", "串联", "联游", "路线"}) else "global_summary"
            return RouteDecision(
                text,
                "graphrag",
                "medium",
                "问题需要跨目的地比较、总结、关联或路线归纳，交给 GraphRAG。",
                query_type,
                mainland_hits,
                graph_hits,
            )
        if mainland_hits:
            return RouteDecision(
                text,
                "naive_rag",
                "high",
                "问题是明确地区或目的地的具体推荐与详细问答，优先使用大陆 CSV/FAISS RAG。",
                "detail",
                mainland_hits,
                mainland_hits,
            )
        return RouteDecision(text, "naive_rag", "high", "问题偏目的地详细问答，交给大陆 CSV/FAISS RAG。", "detail", mainland_hits, mainland_hits)

    def _unsupported(self, query: str) -> bool:
        if not query or len(query) < 2:
            return True
        blocked = ["天气", "Python", "快排", "上传", "照片", "解析", "PDF", "图片", "OCR", "VLM", "火星"]
        if any(term.lower() in query.lower() for term in blocked):
            return True
        if re.fullmatch(r"[a-zA-Z0-9_]{4,}", query):
            return True
        return False


class SystemRouter:
    """SystemAgent 使用的规则 + 可选 LLM 路由器。"""

    def __init__(self, settings: ProjectSettings, llm_client: LLMClientProtocol | None = None) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self.rule = SupervisorRouter()
        self.last_route_source = "rule"
        self.last_fallback_reason = ""

    def route(self, query: str) -> RouteDecision:
        rule_decision = self.rule.route(query)
        if rule_decision.route == "fallback":
            self.last_route_source = "rule_fallback"
            self.last_fallback_reason = "unsupported_guard"
            return rule_decision
        if not (self.settings.llm_enabled and self.settings.system_agent_llm_router_enabled and self.llm_client):
            self.last_route_source = "rule"
            self.last_fallback_reason = ""
            return rule_decision
        try:
            prompt = self._build_prompt(query)
            raw = self.llm_client.generate(prompt)
            decision = self._parse(query, raw)
        except Exception as exc:
            self.last_route_source = "rule_fallback"
            self.last_fallback_reason = f"llm_router_error:{exc.__class__.__name__}"
            return self.rule.route(query)
        if decision.confidence == "low":
            self.last_route_source = "rule_fallback"
            self.last_fallback_reason = "low_confidence"
            return rule_decision
        if self._should_apply_rule_guard(rule_decision, decision):
            decision = RouteDecision(
                query=query,
                route=rule_decision.route,
                confidence=max(decision.confidence, rule_decision.confidence, key={"low": 0, "medium": 1, "high": 2}.get),
                reason=f"LLM Router 后经规则护栏修正：{rule_decision.reason}",
                query_type=rule_decision.query_type,
                entities=rule_decision.entities,
                matched_terms=rule_decision.matched_terms,
            )
        self.last_route_source = "llm"
        self.last_fallback_reason = ""
        return decision

    def _build_prompt(self, query: str) -> str:
        try:
            base = load_prompt("routing_prompt.md")
        except Exception:
            base = "你是 TravelMind Router，只输出 JSON。"
        return f"{base}\n\n用户问题：{query}"

    def _parse(self, query: str, raw: str) -> RouteDecision:
        data = _json_object(raw)
        route = data.get("route")
        confidence = data.get("confidence")
        if route not in ROUTES:
            raise ValueError("invalid_route")
        if confidence not in CONFIDENCES:
            raise ValueError("invalid_confidence")
        entities = data.get("entities", [])
        matched_terms = data.get("matched_terms", [])
        if not isinstance(entities, list) or not isinstance(matched_terms, list):
            raise ValueError("invalid_terms")
        return RouteDecision(
            query=query,
            route=str(route),
            confidence=str(confidence),
            reason=str(data.get("reason") or "LLM Router decision"),
            query_type=str(data.get("query_type") or "llm_router"),
            entities=[str(item) for item in entities],
            matched_terms=[str(item) for item in matched_terms],
        )

    def _should_apply_rule_guard(self, rule_decision: RouteDecision, llm_decision: RouteDecision) -> bool:
        if rule_decision.route == llm_decision.route:
            return False
        return rule_decision.route != "fallback"


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("json_not_object")
    return data
