"""GraphRAG 查询实体提取与结果相关性判断。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from travelmind.destinations import entity_aliases, match_destination_entities
from travelmind.schemas import RetrieverResult

GAT_ENTITIES = ("香港", "澳门", "台湾", "台北")


@dataclass(frozen=True)
class RelevanceAssessment:
    relevant: bool
    query_entities: list[str]
    matched_entities: list[str]
    coverage: float
    reason: str


def extract_query_entities(query: str) -> list[str]:
    """提取问题中明确出现的大陆省市和主要目的地实体。"""
    entities = match_destination_entities(query)
    return [
        *[entity for entity in GAT_ENTITIES if entity in query],
        *entities,
    ]


def assess_graphrag_relevance(query: str, result: RetrieverResult) -> RelevanceAssessment:
    """基于核心实体覆盖判断 GraphRAG 证据是否足以支持回答。"""
    metadata_entities = result.metadata.get("query_entities")
    query_entities = (
        [str(item) for item in metadata_entities if str(item).strip()]
        if isinstance(metadata_entities, list)
        else extract_query_entities(query)
    )
    text = f"{result.title or ''}\n{result.content}"
    matched_entities = [
        entity
        for entity in query_entities
        if _has_substantive_entity_coverage(entity, text)
    ]
    coverage = len(matched_entities) / len(query_entities) if query_entities else 0.0

    if not query_entities:
        return RelevanceAssessment(False, [], [], 0.0, "no_core_entities")

    relevant = coverage == 1.0
    reason = "entity_coverage_sufficient" if relevant else "core_entity_missing"
    return RelevanceAssessment(relevant, query_entities, matched_entities, coverage, reason)


def _has_substantive_entity_coverage(entity: str, text: str) -> bool:
    aliases = entity_aliases(entity)
    clauses = [
        clause.strip()
        for clause in re.split(r"[\n，,；;。！？!?]+", text)
        if clause.strip()
    ]
    if any(
        _is_broad_missing_information_clause(alias, clause)
        for clause in clauses
        for alias in aliases
        if alias in clause
    ):
        return False
    for clause in clauses:
        for alias in aliases:
            if alias not in clause:
                continue
            if not _is_missing_information_clause(alias, clause):
                return True
    return False


def _is_missing_information_clause(alias: str, clause: str) -> bool:
    clause = clause.replace("*", "")
    escaped = re.escape(alias)
    information = r"(?:资料|信息|证据|内容|数据|记录)"
    missing = r"(?:没有|未找到|找不到|缺少|暂无|未提供|未检索到|未包含|不包含|不含)"
    patterns = (
        rf"{missing}.{{0,32}}(?:关于|有关|针对)?\s*{escaped}.{{0,32}}{information}",
        rf"(?:关于|有关|针对)?\s*{escaped}.{{0,32}}{missing}.{{0,32}}{information}",
        rf"{escaped}.{{0,12}}{information}.{{0,12}}(?:缺失|不足|不可用|为空|不存在)",
        rf"(?:无法|不能).{{0,20}}(?:对|就|针对)?\s*{escaped}.{{0,20}}(?:评估|回答|判断|比较|分析|得出结论)",
        rf"(?:无法|不能).{{0,20}}(?:评估|回答|判断|比较|分析).{{0,12}}{escaped}",
    )
    return any(re.search(pattern, clause) for pattern in patterns)


def _is_broad_missing_information_clause(alias: str, clause: str) -> bool:
    clause = clause.replace("*", "")
    escaped = re.escape(alias)
    missing = r"(?:没有|未找到|找不到|缺少|暂无|未提供|未检索到|未包含|不包含|不含)"
    qualifier = r"(?:任何|实质性|详细|可用|相关|具体)"
    information = r"(?:资料|信息|证据|内容|数据|记录)"
    parenthetical = r"(?:[（(][^）)]{0,30}[）)])?"
    patterns = (
        rf"{missing}.{{0,24}}(?:关于|有关|针对)?\s*{escaped}{parenthetical}(?:的)?{qualifier}{information}",
        rf"(?:关于|有关|针对)?\s*{escaped}{parenthetical}(?:的)?{qualifier}{information}.{{0,20}}{missing}",
    )
    return any(re.search(pattern, clause) for pattern in patterns)
