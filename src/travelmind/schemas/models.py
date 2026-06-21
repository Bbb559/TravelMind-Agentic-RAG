"""主链路共享的轻量 schema。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceRef:
    source_type: str
    source_path: str | None
    title: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_path": self.source_path,
            "title": self.title,
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class RetrieverResult:
    content: str
    source_type: str
    source_path: str | None
    title: str | None
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    retriever_name: str = ""

    def to_source(self) -> SourceRef:
        return SourceRef(
            source_type=self.source_type,
            source_path=self.source_path,
            title=self.title,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "title": self.title,
            "score": _json_safe(self.score),
            "metadata": _json_safe(self.metadata),
            "retriever_name": self.retriever_name,
        }


@dataclass
class RouteDecision:
    query: str
    route: str
    confidence: str
    reason: str
    query_type: str
    entities: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "route": self.route,
            "confidence": self.confidence,
            "reason": self.reason,
            "query_type": self.query_type,
            "entities": self.entities,
            "matched_terms": self.matched_terms,
        }


@dataclass
class RAGAnswer:
    answer: str
    route: str
    confidence: str
    sources: list[SourceRef] = field(default_factory=list)
    retrieved: list[RetrieverResult] = field(default_factory=list)
    fallback_reason: str | None = None
    trace: list[str] = field(default_factory=list)
    execution_status: dict[str, Any] = field(default_factory=dict)
    hybrid_branch_status: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "route": self.route,
            "confidence": self.confidence,
            "sources": [source.to_dict() for source in self.sources],
            "retrieved": [item.to_dict() for item in self.retrieved],
            "fallback_reason": self.fallback_reason,
            "trace": self.trace,
            "execution_status": _json_safe(self.execution_status),
            "hybrid_branch_status": _json_safe(self.hybrid_branch_status),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


@dataclass
class GraphState:
    query: str
    route_decision: RouteDecision | None = None
    retrieved: list[RetrieverResult] = field(default_factory=list)
    graded_results: list[RetrieverResult] = field(default_factory=list)
    rewritten_query: str | None = None
    tool_query: str | None = None
    answer: RAGAnswer | None = None
    retry_count: int = 0
    trace: list[str] = field(default_factory=list)
