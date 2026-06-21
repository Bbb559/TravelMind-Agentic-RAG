"""大陆旅游 CSV / FAISS 优先检索。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from travelmind.config import ProjectSettings
from travelmind.destinations import entity_is_mentioned, match_destination_entities
from travelmind.schemas import RetrieverResult
from travelmind.travel_intent import classify_travel_intent, fields_for_intent


SECTION_NAMES = {
    "交通安排",
    "住宿推荐",
    "必打卡景点",
    "美食推荐",
    "实用小贴士",
    "旅行感悟",
}


class NaiveTravelRetriever:
    """基于 `assets/travel_guide.csv` 的保底检索。"""

    def __init__(self, settings: ProjectSettings) -> None:
        self.settings = settings

    def retrieve(self, query: str) -> list[RetrieverResult]:
        rows = self._load_rows()
        entities = match_destination_entities(query)
        intent = classify_travel_intent(query)
        required_fields = fields_for_intent(intent)
        scored: list[tuple[int, dict[str, str], list[str]]] = []
        for row in rows:
            text = " ".join(row.values())
            matched_entities = [
                entity
                for entity in entities
                if entity_is_mentioned(entity, text)
            ]
            if not entities or len(matched_entities) != len(entities):
                continue
            available_fields = [
                field
                for field in required_fields
                if str(row.get(field) or "").strip()
            ]
            if not available_fields:
                continue
            title = str(row.get("目的地") or "")
            exact_title_hits = sum(
                1
                for entity in entities
                if entity_is_mentioned(entity, title)
            )
            score = 100 * len(matched_entities) + 10 * exact_title_hits + len(available_fields)
            scored.append((score, row, matched_entities))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._to_result(
                row,
                float(score),
                query=query,
                intent=intent,
                matched_entities=matched_entities,
            )
            for score, row, matched_entities in scored[:4]
        ]

    def _load_rows(self) -> list[dict[str, str]]:
        for encoding in ("utf-8-sig", "gbk", "gb18030"):
            try:
                with self.settings.travel_csv_path.open("r", encoding=encoding, newline="") as file:
                    return list(csv.DictReader(file))
            except UnicodeDecodeError:
                continue
        with self.settings.travel_csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
            return list(csv.DictReader(file))

    def _to_result(
        self,
        row: dict[str, str],
        score: float,
        *,
        query: str,
        intent: str,
        matched_entities: list[str],
    ) -> RetrieverResult:
        title = row.get("目的地") or "travel_guide"
        sections = {
            key: str(value).strip()
            for key, value in row.items()
            if key in SECTION_NAMES and str(value or "").strip()
        }
        selected_sections = {
            key: value
            for key, value in sections.items()
            if key in fields_for_intent(intent)
        }
        content = "\n".join(
            f"{key}: {value}"
            for key, value in selected_sections.items()
        )
        return RetrieverResult(
            content=content,
            source_type="csv",
            source_path=str(self.settings.travel_csv_path),
            title=title,
            score=score,
            metadata={
                "retrieval_mode": "csv",
                "destination": title,
                "sections": selected_sections,
                "query_intent": intent,
                "matched_entities": matched_entities,
                "matched_intents": [intent],
                "evidence_valid": True,
                "evidence_reason": "entity_and_intent_covered",
                "query": query,
            },
            retriever_name="NaiveTravelRetriever",
        )


class NaiveAutoTravelRetriever:
    """优先尝试 FAISS，失败自动回退 CSV。"""

    def __init__(self, settings: ProjectSettings, faiss_loader: Any | None = None) -> None:
        self.settings = settings
        self.csv_retriever = NaiveTravelRetriever(settings)
        self.faiss_loader = faiss_loader
        self.last_mode = "csv"
        self.last_reason = ""
        self.last_evidence_reason = ""

    def retrieve(self, query: str) -> list[RetrieverResult]:
        index_dir = self.settings.faiss_index_dir
        reason = self._faiss_unavailable_reason(index_dir)
        if reason:
            return self._fallback(query, reason)
        try:
            results = self._retrieve_faiss(query, index_dir)
        except Exception as exc:
            return self._fallback(query, f"faiss_error:{exc.__class__.__name__}")
        if not results:
            return self._fallback(query, "faiss_no_relevant_evidence")
        self.last_mode = "faiss"
        self.last_reason = ""
        self.last_evidence_reason = "entity_and_intent_covered"
        return results

    def _faiss_unavailable_reason(self, index_dir: Path) -> str | None:
        try:
            index_dir.resolve().relative_to(self.settings.assets_dir.resolve())
        except ValueError:
            return "unsafe_index_path"
        if not (index_dir / "index.faiss").exists() or not (index_dir / "index.pkl").exists():
            return "index_missing"
        if not self.settings.embedding_api_key:
            return "missing_embedding_key"
        return None

    def _retrieve_faiss(self, query: str, index_dir: Path) -> list[RetrieverResult]:
        if self.faiss_loader is not None:
            docs_scores = self.faiss_loader(query)
        else:
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_community.vectorstores import FAISS

            embeddings = DashScopeEmbeddings(
                model=self.settings.embedding_model,
                dashscope_api_key=self.settings.embedding_api_key,
            )
            vectorstore = FAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            docs_scores = vectorstore.similarity_search_with_score(query, k=4)
        results: list[RetrieverResult] = []
        entities = match_destination_entities(query)
        intent = classify_travel_intent(query)
        for index, item in enumerate(docs_scores):
            if isinstance(item, tuple):
                doc, raw_score = item
            else:
                doc, raw_score = item, None
            metadata = dict(getattr(doc, "metadata", {}) or {})
            content = str(getattr(doc, "page_content", doc))
            title = metadata.get("目的地") or metadata.get("destination") or metadata.get("title") or f"faiss_result_{index}"
            searchable = f"{title}\n{content}\n{metadata}"
            matched_entities = [
                entity
                for entity in entities
                if entity_is_mentioned(entity, searchable)
            ]
            if not entities or len(matched_entities) != len(entities):
                continue
            sections = _parse_sections(content, metadata)
            selected_sections = {
                key: value
                for key, value in sections.items()
                if key in fields_for_intent(intent)
            }
            if not selected_sections:
                continue
            metadata.update(
                {
                    "retrieval_mode": "faiss",
                    "index_found": True,
                    "faiss_index_dir": str(index_dir),
                    "embedding_model": self.settings.embedding_model,
                    "raw_score": raw_score,
                    "sections": selected_sections,
                    "query_intent": intent,
                    "matched_entities": matched_entities,
                    "matched_intents": [intent],
                    "evidence_valid": True,
                    "evidence_reason": "entity_and_intent_covered",
                }
            )
            results.append(
                RetrieverResult(
                    content="\n".join(
                        f"{key}: {value}"
                        for key, value in selected_sections.items()
                    ),
                    source_type="csv",
                    source_path=str(self.settings.travel_csv_path),
                    title=str(title),
                    score=None if raw_score is None else float(raw_score),
                    metadata=metadata,
                    retriever_name="NaiveFaissTravelRetriever",
                )
            )
        return results

    def _fallback(self, query: str, reason: str) -> list[RetrieverResult]:
        self.last_mode = "faiss_fallback_csv"
        self.last_reason = reason
        results = self.csv_retriever.retrieve(query)
        self.last_evidence_reason = (
            "entity_and_intent_covered"
            if results
            else _empty_reason(query)
        )
        for result in results:
            result.metadata.update(
                {
                    "retrieval_mode": "csv",
                    "faiss_fallback_reason": reason,
                    "index_found": (self.settings.faiss_index_dir / "index.faiss").exists(),
                    "dependencies_available": True,
                    "embedding_key_present": bool(self.settings.embedding_api_key),
                }
            )
        return results


def _parse_sections(content: str, metadata: dict[str, Any]) -> dict[str, str]:
    existing = metadata.get("sections")
    if isinstance(existing, dict):
        return {
            str(key): str(value)
            for key, value in existing.items()
            if str(key) in SECTION_NAMES and str(value).strip()
        }
    sections: dict[str, str] = {}
    for line in content.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in SECTION_NAMES and value.strip():
            sections[key.strip()] = value.strip()
    return sections


def _empty_reason(query: str) -> str:
    if any(term in query for term in ("新加坡", "圣淘沙")):
        return "destination_not_covered"
    if match_destination_entities(query):
        return "intent_evidence_missing"
    return "no_relevant_evidence"
