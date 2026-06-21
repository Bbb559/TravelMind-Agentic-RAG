"""GraphRAG 真实 global_search 优先、索引证据安全降级的检索封装。"""

from __future__ import annotations

import re
from typing import Any

from travelmind.config import ProjectSettings
from travelmind.graphrag import (
    GraphRAGGlobalSearchAdapter,
    GraphRAGOfficialLocalSearchAdapter,
    assess_graphrag_relevance,
    extract_query_entities,
)
from travelmind.schemas import RetrieverResult


class GraphRAGSearchRetriever:
    """编排官方 Global/Local Search 与安全 evidence/wrapper 降级。"""

    REQUIRED = [
        "entities.parquet",
        "communities.parquet",
        "relationships.parquet",
        "community_reports.parquet",
        "text_units.parquet",
    ]

    def __init__(
        self,
        settings: ProjectSettings,
        adapter: Any | None = None,
        *,
        local_adapter: Any | None = None,
        allow_global_search: bool = False,
    ) -> None:
        self.settings = settings
        self.adapter = adapter or GraphRAGGlobalSearchAdapter(settings)
        self.local_adapter = local_adapter or GraphRAGOfficialLocalSearchAdapter(settings)
        self.allow_global_search = allow_global_search
        self.last_mode = "graphrag_wrapper"
        self.last_reason = ""
        self.last_diagnostics: dict[str, Any] = {}

    def retrieve(self, query: str) -> list[RetrieverResult]:
        missing = [name for name in self.REQUIRED if not (self.settings.graphrag_output_dir / name).exists()]
        gate_reason = self._global_search_gate_reason()
        if missing:
            return [self._wrapper_result(query, "index_missing")]

        if not gate_reason:
            global_result, global_reason = self._try_global_search(query)
            if global_result is not None:
                return [global_result]
        else:
            global_reason = gate_reason
            self.last_diagnostics = {
                **self.last_diagnostics,
                "global_gate_reason": gate_reason,
                "global_search_called": False,
                "global_search_succeeded": False,
            }

        local_result, local_reason = self._try_official_local_search(query)
        if local_result is not None:
            return [local_result]

        reason = local_reason or global_reason or "official_local_failed"
        evidence_result = self._local_evidence_result(query, reason)
        if evidence_result is not None:
            self.last_mode = "graphrag_local_evidence"
            self.last_reason = reason
            self.last_diagnostics = {
                **self.last_diagnostics,
                "adapter_mode": "local_index_evidence",
                "global_search_called": bool(self.last_diagnostics.get("global_search_called")),
                "global_search_succeeded": bool(self.last_diagnostics.get("global_search_succeeded")),
                "official_local_called": bool(self.last_diagnostics.get("official_local_called")),
                "official_local_succeeded": False,
                "official_local_error": local_reason,
                "raw_preview": evidence_result.content[:500],
                "fallback_reason": "official_local_failed",
                "quality_status": self.last_diagnostics.get(
                    "quality_status",
                    "SKIP",
                ),
            }
            evidence_result.metadata.update(
                {
                    "fallback_reason": "official_local_failed",
                    "official_local_error": local_reason,
                    "global_search_error": None if gate_reason else global_reason,
                }
            )
            return [evidence_result]
        return [
            self._wrapper_result(
                query,
                "official_local_failed",
                diagnostics={
                    **self.last_diagnostics,
                    "official_local_error": local_reason,
                    "global_search_error": None if gate_reason else global_reason,
                },
            )
        ]

    def _try_global_search(self, query: str) -> tuple[RetrieverResult | None, str]:
        ready, reason = self.adapter.readiness()
        adapter_diagnostics = dict(
            getattr(self.adapter, "last_diagnostics", {})
        )
        if ready:
            adapter_diagnostics.update(
                {
                    "global_search_called": True,
                    "global_search_succeeded": False,
                }
            )
            try:
                outcome = self.adapter.search(query)
                result = self._global_result(query, outcome)
                adapter_diagnostics.update(
                    dict(
                        outcome.get("diagnostics")
                        or getattr(self.adapter, "last_diagnostics", {})
                    )
                )
                adapter_diagnostics.update(
                    {
                        "global_search_called": True,
                        "global_search_succeeded": True,
                    }
                )
                adapter_diagnostics["global_answer_preview"] = str(
                    outcome.get("raw_preview") or ""
                )[:300]
                if (
                    result.metadata["graphrag_relevance"]
                    and _has_valid_source_summary(
                        result.metadata.get("source_summary")
                    )
                ):
                    self.last_mode = "graphrag_global_search"
                    self.last_reason = ""
                    self.last_diagnostics = {
                        **adapter_diagnostics,
                        "adapter_mode": "real_global_search",
                        "raw_preview": outcome.get("raw_preview", ""),
                        "fallback_reason": None,
                        "quality_status": "PASS",
                    }
                    return result, ""
                reason = (
                    "low_coverage"
                    if not result.metadata["graphrag_relevance"]
                    else "missing_source_summary"
                )
                adapter_diagnostics["quality_status"] = "FAIL"
                adapter_diagnostics["fallback_reason"] = reason
                adapter_diagnostics["global_search_succeeded"] = False
                if hasattr(self.adapter, "last_diagnostics"):
                    self.adapter.last_diagnostics.update(
                        {
                            "global_search_succeeded": False,
                            "fallback_reason": reason,
                        }
                    )
            except TimeoutError:
                reason = "timeout"
                adapter_diagnostics.update(
                    dict(getattr(self.adapter, "last_diagnostics", {}))
                )
                adapter_diagnostics.update(
                    {
                        "global_search_called": True,
                        "global_search_succeeded": False,
                    }
                )
            except Exception as exc:
                adapter_diagnostics.update(
                    dict(getattr(self.adapter, "last_diagnostics", {}))
                )
                adapter_diagnostics.update(
                    {
                        "global_search_called": True,
                        "global_search_succeeded": False,
                    }
                )
                reason = str(
                    adapter_diagnostics.get("fallback_reason")
                    or f"sdk_error:{exc.__class__.__name__}"
                )
        else:
            adapter_diagnostics.update(
                {
                    "global_search_called": False,
                    "global_search_succeeded": False,
                }
            )

        self.last_diagnostics = {
            **self.last_diagnostics,
            **adapter_diagnostics,
            "global_search_error": reason or "global_search_unavailable",
            "global_search_succeeded": False,
        }
        return None, reason or "global_search_unavailable"

    def _try_official_local_search(self, query: str) -> tuple[RetrieverResult | None, str]:
        ready, reason = self.local_adapter.readiness()
        diagnostics = dict(getattr(self.local_adapter, "last_diagnostics", {}))
        if not ready:
            self.last_diagnostics = {
                **self.last_diagnostics,
                **diagnostics,
                "official_local_called": False,
                "official_local_succeeded": False,
                "official_local_error": reason,
            }
            return None, reason
        try:
            outcome = self.local_adapter.search(query)
            result = self._official_local_result(query, outcome)
            if not result.metadata["graphrag_relevance"]:
                reason = "low_coverage"
            elif not _has_valid_source_summary(
                result.metadata.get("source_summary")
            ):
                reason = "missing_source_summary"
            else:
                self.last_mode = "graphrag_local_search"
                self.last_reason = ""
                self.last_diagnostics = {
                    **self.last_diagnostics,
                    **dict(outcome.get("diagnostics") or getattr(self.local_adapter, "last_diagnostics", {})),
                    "adapter_mode": "official_local_search",
                    "official_local_called": True,
                    "official_local_succeeded": True,
                    "official_local_error": None,
                    "fallback_reason": None,
                    "quality_status": "PASS",
                }
                return result, ""
            if hasattr(self.local_adapter, "last_diagnostics"):
                self.local_adapter.last_diagnostics.update(
                    {
                        "official_local_succeeded": False,
                        "official_local_error": reason,
                    }
                )
        except TimeoutError:
            reason = "timeout"
        except Exception as exc:
            reason = str(
                getattr(self.local_adapter, "last_diagnostics", {}).get(
                    "official_local_error",
                    f"sdk_error:{exc.__class__.__name__}",
                )
            )
        self.last_diagnostics = {
            **self.last_diagnostics,
            **dict(getattr(self.local_adapter, "last_diagnostics", {})),
            "official_local_called": bool(
                getattr(self.local_adapter, "last_diagnostics", {}).get(
                    "official_local_called",
                    ready,
                )
            ),
            "official_local_succeeded": False,
            "official_local_error": reason,
        }
        return None, reason

    def _global_search_gate_reason(self) -> str:
        if not self.settings.graphrag_global_search_enabled:
            return "global_search_disabled"
        if not self.allow_global_search:
            return "request_not_allowed"
        return ""

    def _global_result(self, query: str, outcome: dict[str, Any]) -> RetrieverResult:
        result = RetrieverResult(
            content=str(outcome.get("content") or "")[: self.settings.graphrag_max_context_chars],
            source_type="graphrag_index",
            source_path=str(outcome.get("source_path") or self.settings.graphrag_output_dir),
            title="GraphRAG global_search",
            score=None,
            metadata={
                "retrieval_mode": "graphrag_global_search",
                "global_search_available": True,
                "index_found": True,
                "query_entities": extract_query_entities(query),
                "max_context_chars": self.settings.graphrag_max_context_chars,
                "source_summary": list(outcome.get("source_summary") or []),
            },
            retriever_name="GraphRAGSearchRetriever",
        )
        self._apply_relevance(query, result)
        return result

    def _official_local_result(self, query: str, outcome: dict[str, Any]) -> RetrieverResult:
        result = RetrieverResult(
            content=str(outcome.get("content") or "")[: self.settings.graphrag_max_context_chars],
            source_type="graphrag_index",
            source_path=str(outcome.get("source_path") or self.settings.graphrag_output_dir),
            title="GraphRAG official local_search",
            score=None,
            metadata={
                "retrieval_mode": "graphrag_local_search",
                "official_local_search_available": True,
                "global_search_available": False,
                "index_found": True,
                "query_entities": extract_query_entities(query),
                "max_context_chars": self.settings.graphrag_max_context_chars,
                "source_summary": list(outcome.get("source_summary") or []),
                "answer_policy": "official_search_response",
            },
            retriever_name="GraphRAGSearchRetriever",
        )
        self._apply_relevance(query, result)
        return result

    def _local_evidence_result(self, query: str, fallback_reason: str) -> RetrieverResult | None:
        import pandas as pd

        query_entities = extract_query_entities(query)
        if not query_entities:
            return None

        output = self.settings.graphrag_output_dir
        evidence: list[tuple[float, str, str, str]] = []
        table_specs = [
            ("text_units.parquet", ("text",)),
            ("community_reports.parquet", ("title", "summary", "full_content")),
            ("entities.parquet", ("title", "description")),
            ("relationships.parquet", ("source", "target", "description")),
        ]
        for filename, columns in table_specs:
            path = output / filename
            frame = pd.read_parquet(path)
            available = [column for column in columns if column in frame.columns]
            if not available:
                continue
            for _, row in frame.iterrows():
                text = "\n".join(str(row.get(column, "")) for column in available if str(row.get(column, "")).strip())
                for entity in query_entities:
                    if entity not in text:
                        continue
                    snippet = _entity_window(text, entity)
                    quality = _evidence_quality(snippet)
                    if quality <= 0:
                        continue
                    table_bonus = 2.0 if filename == "text_units.parquet" else 1.0
                    heading_bonus = 4.0 if entity in snippet[:100] else 0.0
                    evidence.append((quality + table_bonus + heading_bonus, filename, entity, snippet))

        if not evidence:
            return None

        evidence.sort(key=lambda item: (item[0], len(item[3])), reverse=True)
        selected: list[tuple[float, str, str, str]] = []
        for entity in query_entities:
            candidate = next((item for item in evidence if item[2] == entity), None)
            if candidate is not None:
                selected.append(candidate)

        if not selected:
            return None

        content_parts = [f"[{filename}:{entity}]\n{snippet}" for _, filename, entity, snippet in selected]
        content = "\n\n".join(content_parts)[: self.settings.graphrag_max_context_chars]
        result = RetrieverResult(
            content=content,
            source_type="graphrag_index",
            source_path=str(output / selected[0][1]),
            title="GraphRAG 索引证据检索",
            score=None,
            metadata={
                "retrieval_mode": "graphrag_local_evidence",
                "global_search_available": False,
                "index_found": True,
                "query_entities": query_entities,
                "evidence_tables": sorted({filename for _, filename, _, _ in selected}),
                "evidence_quality": True,
                "fallback_reason": fallback_reason,
                "confidence_policy": "low",
                "answer_policy": "evidence_preview_only",
            },
            retriever_name="GraphRAGSearchRetriever",
        )
        self._apply_relevance(query, result)
        return result if result.metadata["graphrag_relevance"] else None

    def _apply_relevance(self, query: str, result: RetrieverResult) -> None:
        assessment = assess_graphrag_relevance(query, result)
        result.metadata.update(
            {
                "graphrag_relevance": assessment.relevant,
                "relevance_reason": assessment.reason,
                "query_entities": assessment.query_entities,
                "matched_entities": assessment.matched_entities,
                "entity_coverage": assessment.coverage,
            }
        )
        result.score = assessment.coverage

    def _wrapper_result(
        self,
        query: str,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> RetrieverResult:
        self.last_mode = "graphrag_wrapper"
        self.last_reason = reason
        existing = [name for name in self.REQUIRED if (self.settings.graphrag_output_dir / name).exists()]
        self.last_diagnostics = {
            **(diagnostics or {}),
            "adapter_mode": "wrapper",
            "raw_preview": "",
            "fallback_reason": reason,
            "quality_status": (diagnostics or {}).get("quality_status", "SKIP"),
        }
        return RetrieverResult(
            content=f"GraphRAG 索引未找到与问题核心实体明确相关的可回答证据。问题：{query}",
            source_type="graphrag_index",
            source_path=str(self.settings.graphrag_output_dir),
            title="GraphRAG 安全兜底",
            score=0.0,
            metadata={
                "retrieval_mode": "graphrag_wrapper",
                "global_search_available": False,
                "index_found": bool(existing),
                "existing_files": existing,
                "fallback_reason": reason,
                "graphrag_relevance": False,
                "query_entities": extract_query_entities(query),
                "matched_entities": [],
                "entity_coverage": 0.0,
                "confidence_policy": "low",
                "answer_policy": "evidence_preview_only",
                "official_local_error": (diagnostics or {}).get(
                    "official_local_error"
                ),
                "global_search_error": (diagnostics or {}).get(
                    "global_search_error"
                ),
            },
            retriever_name="GraphRAGSearchRetriever",
        )


def _entity_window(text: str, entity: str, radius: int = 550) -> str:
    guide_starts = [match.start() for match in re.finditer(r"[\u4e00-\u9fff]{2,45}旅游攻略：", text)]
    for position, start in enumerate(guide_starts):
        end = guide_starts[position + 1] if position + 1 < len(guide_starts) else len(text)
        section = text[start:end].strip()
        if entity in section[:120]:
            return section[:1200]

    index = text.find(entity)
    start = max(0, index - min(radius, 320))
    next_guide_start = next((position for position in guide_starts if position > index), None)
    end = next_guide_start if next_guide_start is not None else min(len(text), index + len(entity) + radius)
    snippet = text[start:end].strip()
    if start > 0:
        sentence_start = max(snippet.find("。"), snippet.find("\n"))
        if 0 <= sentence_start < 160:
            snippet = snippet[sentence_start + 1 :].lstrip()
    return snippet[:1200]


def _evidence_quality(text: str) -> float:
    if not text.strip():
        return 0.0
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    visible_count = len(re.findall(r"\S", text))
    cjk_ratio = cjk_count / visible_count if visible_count else 0.0
    travel_hits = sum(
        term in text
        for term in ("旅游", "景点", "路线", "交通", "住宿", "风景", "古城", "公园", "出发", "推荐")
    )
    replacement_penalty = text.count("�") * 0.25
    score = cjk_ratio * 5 + min(travel_hits, 5) - replacement_penalty
    return score if cjk_ratio >= 0.25 or travel_hits >= 2 else 0.0


def _has_valid_source_summary(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(
        isinstance(item, dict)
        and bool(str(item.get("section") or "").strip())
        and isinstance(item.get("row_count"), int)
        and item["row_count"] > 0
        for item in value
    )
