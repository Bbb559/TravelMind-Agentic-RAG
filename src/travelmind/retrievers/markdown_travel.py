"""港澳台离线 Markdown 检索。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from travelmind.config import ProjectSettings
from travelmind.schemas import RetrieverResult

REGION_FILES = {
    "香港": ("xianggang", "xianggang.md"),
    "澳门": ("aomen", "aomen.md"),
    "台湾": ("taiwan", "taiwan.md"),
    "台北": ("taiwan", "taiwan.md"),
}
TOPIC_TERMS = ("迪士尼", "大三巴", "101", "亲子", "景点", "交通", "美食")


def _region_for_query(query: str) -> tuple[str, str] | None:
    for key, value in REGION_FILES.items():
        if key in query:
            return value
    return None


class MultimodalMarkdownRetriever:
    """基于 canonical Markdown 文件的关键词检索。"""

    def __init__(self, settings: ProjectSettings) -> None:
        self.settings = settings

    def retrieve(self, query: str) -> list[RetrieverResult]:
        targets = [_region_for_query(query)] if _region_for_query(query) else list(dict.fromkeys(REGION_FILES.values()))
        results: list[RetrieverResult] = []
        for target in targets:
            if target is None:
                continue
            region, filename = target
            path = self.settings.multimodal_markdown_dir / region / filename
            if not path.exists():
                continue
            chunks = self._chunks(path)
            for index, (heading, content) in enumerate(chunks):
                score = self._score(query, heading + "\n" + content)
                if score > 0:
                    results.append(self._result(path, region, heading, content, index, float(score), "markdown_keyword"))
        results.sort(key=lambda item: item.score or 0.0, reverse=True)
        return results[:4]

    def _chunks(self, path: Path) -> list[tuple[str, str]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks: list[tuple[str, list[str]]] = []
        current_heading = path.stem
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith("#"):
                if current:
                    chunks.append((current_heading, current))
                    current = []
                current_heading = line.lstrip("#").strip() or current_heading
            else:
                current.append(line)
        if current:
            chunks.append((current_heading, current))
        return [(heading, "\n".join(lines).strip()[:1200]) for heading, lines in chunks if "\n".join(lines).strip()]

    def _score(self, query: str, text: str) -> int:
        terms = [*TOPIC_TERMS, "香港", "澳门", "台湾", "台北"]
        return sum(1 for term in terms if term in query and term in text)

    def _result(
        self,
        path: Path,
        region: str,
        heading: str,
        content: str,
        chunk_index: int,
        score: float,
        mode: str,
    ) -> RetrieverResult:
        return RetrieverResult(
            content=content,
            source_type="pdf_markdown",
            source_path=str(path),
            title=heading,
            score=score,
            metadata={
                "retrieval_mode": mode,
                "region": region,
                "heading": heading,
                "chunk_index": chunk_index,
                "source_path": str(path),
                "matched_entities": [region],
                "matched_intents": ["multimodal_topic"],
                "evidence_valid": True,
                "evidence_reason": "region_and_topic_covered",
            },
            retriever_name="MultimodalMarkdownRetriever",
        )


class MultimodalVectorMarkdownRetriever:
    """优先使用 Markdown FAISS，失败回退关键词 Markdown。"""

    def __init__(self, settings: ProjectSettings, vector_loader: Any | None = None) -> None:
        self.settings = settings
        self.keyword = MultimodalMarkdownRetriever(settings)
        self.vector_loader = vector_loader
        self.last_mode = "markdown_keyword"
        self.last_reason = ""
        self.last_evidence_reason = ""

    def retrieve(self, query: str) -> list[RetrieverResult]:
        reason = self._unavailable_reason()
        if reason:
            return self._fallback(query, reason)
        try:
            results = self._retrieve_vector(query)
        except Exception as exc:
            return self._fallback(query, f"vector_error:{exc.__class__.__name__}")
        if not results:
            return self._fallback(query, "vector_empty")
        self.last_mode = "markdown_vector"
        self.last_reason = ""
        self.last_evidence_reason = "region_and_topic_covered"
        return results

    def _unavailable_reason(self) -> str | None:
        index_dir = self.settings.multimodal_markdown_dir
        try:
            index_dir.resolve().relative_to(self.settings.assets_dir.resolve())
        except ValueError:
            return "unsafe_index_path"
        if not (index_dir / "index.faiss").exists() or not (index_dir / "index.pkl").exists():
            return "index_missing"
        if not self.settings.embedding_api_key:
            return "missing_embedding_key"
        return None

    def _retrieve_vector(self, query: str) -> list[RetrieverResult]:
        if self.vector_loader is not None:
            docs_scores = self.vector_loader(query)
        else:
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_community.vectorstores import FAISS

            embeddings = DashScopeEmbeddings(
                model=self.settings.embedding_model,
                dashscope_api_key=self.settings.embedding_api_key,
            )
            vectorstore = FAISS.load_local(
                str(self.settings.multimodal_markdown_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            docs_scores = vectorstore.similarity_search_with_score(query, k=4)
        results: list[RetrieverResult] = []
        expected_region = (_region_for_query(query) or (None, ""))[0]
        required_topics = [
            term
            for term in TOPIC_TERMS
            if term in query
        ]
        for index, item in enumerate(docs_scores):
            doc, raw_score = item if isinstance(item, tuple) else (item, None)
            metadata = dict(getattr(doc, "metadata", {}) or {})
            content = str(getattr(doc, "page_content", doc))
            source_path = metadata.get("source_path") or metadata.get("source")
            region = metadata.get("region") or self._region_from_source(source_path)
            if not region:
                continue
            heading = metadata.get("heading") or metadata.get("title") or self._infer_heading(content)
            searchable = f"{heading}\n{content}"
            if expected_region and region != expected_region:
                continue
            if required_topics and not any(
                term in searchable
                for term in required_topics
            ):
                continue
            if not source_path:
                region_dir = {"xianggang": "xianggang.md", "aomen": "aomen.md", "taiwan": "taiwan.md"}.get(
                    region, "xianggang.md"
                )
                source_path = str(self.settings.multimodal_markdown_dir / str(region) / region_dir)
            metadata.update(
                {
                    "retrieval_mode": "markdown_vector",
                    "raw_score": raw_score,
                    "score": None if raw_score is None else 1.0 / (1.0 + abs(float(raw_score))),
                    "embedding_model": self.settings.embedding_model,
                    "index_found": True,
                    "source_path": source_path,
                    "region": region,
                    "heading": heading,
                    "matched_entities": [str(region)],
                    "matched_intents": required_topics or ["multimodal_topic"],
                    "evidence_valid": True,
                    "evidence_reason": "region_and_topic_covered",
                }
            )
            results.append(
                RetrieverResult(
                    content=content,
                    source_type="pdf_markdown",
                    source_path=str(source_path),
                    title=str(heading),
                    score=metadata["score"],
                    metadata=metadata,
                    retriever_name="MultimodalVectorMarkdownRetriever",
                )
            )
        return results

    def _fallback(self, query: str, reason: str) -> list[RetrieverResult]:
        self.last_mode = "vector_fallback_markdown"
        self.last_reason = reason
        results = self.keyword.retrieve(query)
        self.last_evidence_reason = (
            "region_and_topic_covered"
            if results
            else "no_relevant_evidence"
        )
        for result in results:
            result.metadata.update(
                {
                    "retrieval_mode": "markdown_keyword",
                    "vector_fallback_reason": reason,
                    "index_found": (self.settings.multimodal_markdown_dir / "index.faiss").exists(),
                    "dependencies_available": True,
                    "embedding_key_present": bool(self.settings.embedding_api_key),
                }
            )
        return results

    def _region_from_source(self, source_path: str | None) -> str | None:
        if source_path:
            lowered = source_path.lower()
            for region in ("xianggang", "aomen", "taiwan"):
                if region in lowered:
                    return region
        return None

    def _infer_heading(self, content: str) -> str:
        for line in content.splitlines():
            if line.strip():
                return line.strip().lstrip("#")[:60]
        return "markdown_vector_chunk"
