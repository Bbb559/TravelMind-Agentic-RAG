"""Agentic RAG 运行期评分、改写与生成辅助。"""

from __future__ import annotations

import json
import re
from typing import Any

from travelmind.graphrag import assess_graphrag_relevance
from travelmind.llm import LLMClientProtocol, load_prompt
from travelmind.schemas import GraphState, RAGAnswer, RetrieverResult


def parse_json_object(raw: str) -> dict[str, Any]:
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


def deterministic_grade(result: RetrieverResult) -> str:
    mode = result.metadata.get("retrieval_mode")
    if mode in {"graphrag_local_search", "graphrag_global_search"}:
        return "pass" if result.metadata.get("graphrag_relevance") is True else "fail"
    if mode == "graphrag_local_evidence":
        return "weak" if result.metadata.get("graphrag_relevance") is True else "fail"
    if mode in {"faiss", "markdown_vector"}:
        return "pass"
    if mode in {"csv", "markdown_keyword"} and (result.score or 0) > 0:
        return "pass"
    if result.source_type == "graphrag_index":
        return "weak"
    return "fail"


def grade_results(
    query: str,
    results: list[RetrieverResult],
    trace: list[str],
    llm_enabled: bool,
    llm_client: LLMClientProtocol | None,
) -> list[RetrieverResult]:
    if not results:
        trace.append("grade:fail:no_results")
        return []
    for result in results:
        grade = deterministic_grade(result)
        result.metadata.update({"grade": grade, "grade_source": "deterministic", "usable_for_answer": grade == "pass"})
    grades = ",".join(str(item.metadata.get("grade")) for item in results)
    if not (llm_enabled and llm_client):
        trace.append(f"grade:deterministic:{grades}")
        return _usable_with_cautious(query, results)
    try:
        prompt = _build_grading_prompt(query, results)
        data = parse_json_object(llm_client.generate(prompt))
        by_index = data.get("results", [])
        if not isinstance(by_index, list):
            raise ValueError("invalid_results")
        for item in by_index:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", -1))
            if 0 <= index < len(results):
                grade = str(item.get("grade", "weak"))
                if grade not in {"pass", "weak", "fail"}:
                    grade = "weak"
                retrieval_mode = results[index].metadata.get("retrieval_mode")
                if retrieval_mode in {"graphrag_wrapper", "community_reports_preview"} and grade == "pass":
                    grade = "weak"
                if retrieval_mode in {
                    "graphrag_local_search",
                    "graphrag_global_search",
                    "graphrag_local_evidence",
                }:
                    assessment = assess_graphrag_relevance(query, results[index])
                    results[index].metadata.update(
                        {
                            "graphrag_relevance": assessment.relevant,
                            "relevance_reason": assessment.reason,
                            "query_entities": assessment.query_entities,
                            "matched_entities": assessment.matched_entities,
                            "entity_coverage": assessment.coverage,
                        }
                    )
                    if not assessment.relevant:
                        grade = "fail"
                raw_grade = grade
                cautiously_recoverable_mode = results[index].metadata.get(
                    "retrieval_mode"
                ) in {
                    "faiss",
                    "markdown_vector",
                }
                if (
                    grade == "fail"
                    and cautiously_recoverable_mode
                    and _related(query, results[index])
                ):
                    grade = "weak"
                usable = bool(item.get("usable_for_answer", grade == "pass"))
                results[index].metadata.update(
                    {
                        "grade": grade,
                        "grade_source": "llm",
                        "llm_grade": grade,
                        "llm_grade_raw": raw_grade,
                        "llm_grade_reason": str(item.get("reason", "")),
                        "usable_for_answer": usable and grade == "pass",
                    }
                )
        grades = ",".join(str(item.metadata.get("grade")) for item in results)
        trace.append(f"grade:llm:{grades}")
        return _usable_with_cautious(query, results)
    except Exception as exc:
        trace.append(f"grade:llm_fallback_deterministic:{grades}:{exc.__class__.__name__}")
        return _usable_with_cautious(query, results)


def rewrite_query(
    query: str,
    trace: list[str],
    llm_enabled: bool,
    llm_client: LLMClientProtocol | None,
) -> str:
    deterministic = f"{query} 旅游 攻略"
    if not (llm_enabled and llm_client):
        trace.append(f"rewrite:deterministic:{deterministic}")
        return deterministic
    try:
        prompt = f"{load_prompt('rewrite_prompt.md')}\n\n原问题：{query}"
        data = parse_json_object(llm_client.generate(prompt))
        rewritten = str(data.get("rewritten_query") or "").strip()
        if not _safe_rewrite(query, rewritten):
            trace.append(f"rewrite:llm_rejected_unsafe:{deterministic}")
            return deterministic
        trace.append(f"rewrite:llm:{rewritten}")
        return rewritten
    except Exception:
        trace.append(f"rewrite:llm_fallback_deterministic:{deterministic}")
        return deterministic


def generate_answer_text(
    query: str,
    route: str,
    confidence: str,
    results: list[RetrieverResult],
    trace: list[str],
    llm_enabled: bool,
    llm_client: LLMClientProtocol | None,
    prompt_name: str = "generation_prompt.md",
    agent_name: str | None = None,
    tool_spec: dict[str, Any] | None = None,
) -> str:
    template = _template_answer(query, route, confidence, results)
    if not (llm_enabled and llm_client and results):
        trace.append("generate:template")
        return template
    try:
        prompt = build_generation_prompt(query, route, confidence, results, trace, prompt_name, agent_name, tool_spec)
        answer = llm_client.generate(prompt).strip()
        if not answer:
            raise ValueError("empty_answer")
        trace.append("generate:llm")
        return answer
    except Exception:
        trace.append("generate:llm_fallback_template")
        return template


def build_rag_answer(state: GraphState, route: str, confidence: str, answer_text: str, fallback_reason: str | None) -> RAGAnswer:
    retrieved = state.graded_results or state.retrieved
    sources = [item.to_source() for item in retrieved]
    return RAGAnswer(
        answer=answer_text,
        route=route,
        confidence=confidence,
        sources=sources,
        retrieved=retrieved,
        fallback_reason=fallback_reason,
        trace=state.trace,
    )


def build_generation_prompt(
    query: str,
    route: str,
    confidence: str,
    results: list[RetrieverResult],
    trace: list[str],
    prompt_name: str,
    agent_name: str | None,
    tool_spec: dict[str, Any] | None,
) -> str:
    try:
        base = load_prompt(prompt_name)
    except Exception:
        base = load_prompt("generation_prompt.md")
    payload = {
        "query": query,
        "route": route,
        "confidence": confidence,
        "agent_name": agent_name,
        "tool_spec": tool_spec,
        "sources": [item.to_source().to_dict() for item in results],
        "retrieved": [
            {
                "title": item.title,
                "source_type": item.source_type,
                "content": item.content[:900],
                "metadata": _json_safe(item.metadata),
            }
            for item in results
        ],
        "trace": trace[-20:],
    }
    return f"{base}\n\n只输出 answer 文本，不输出 JSON。\n\n上下文：\n{json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)}"


def _build_grading_prompt(query: str, results: list[RetrieverResult]) -> str:
    try:
        base = load_prompt("grading_prompt.md")
    except Exception:
        base = "判断检索结果是否支持问题，只输出 JSON。"
    payload = [
        {
            "index": index,
            "title": result.title,
            "content": result.content[:700],
            "metadata": _json_safe(result.metadata),
        }
        for index, result in enumerate(results)
    ]
    return f"{base}\n\n问题：{query}\n检索结果：{json.dumps(payload, ensure_ascii=False)}"


def _usable_with_cautious(query: str, results: list[RetrieverResult]) -> list[RetrieverResult]:
    usable: list[RetrieverResult] = []
    unsupported = any(term in query for term in ["天气", "Python", "快排", "上传", "照片", "PDF", "火星", "OCR", "VLM"])
    for result in results:
        grade = result.metadata.get("grade")
        mode = result.metadata.get("retrieval_mode")
        if result.metadata.get("usable_for_answer") and grade == "pass":
            usable.append(result)
            continue
        if (
            grade == "weak"
            and not unsupported
            and mode
            in {
                "faiss",
                "markdown_vector",
                "graphrag_local_search",
                "graphrag_global_search",
            }
            and _related(query, result)
        ):
            result.metadata.update(
                {
                    "answer_policy": "cautious_source_backed",
                    "usable_for_answer": True,
                    "confidence_policy": "low",
                }
            )
            usable.append(result)
    return usable


def _related(query: str, result: RetrieverResult) -> bool:
    if result.source_type == "graphrag_index":
        return assess_graphrag_relevance(query, result).relevant
    text = f"{result.title or ''}\n{result.content}\n{result.metadata}".lower()
    important = ["大理", "双廊", "丽江", "成都", "西安", "南京", "上海", "北京", "香港", "澳门", "台湾", "台北", "迪士尼", "101"]
    hits = [term for term in important if term in query and term.lower() in text]
    return bool(hits) or (result.score is not None and result.score > 0)


def _template_answer(query: str, route: str, confidence: str, results: list[RetrieverResult]) -> str:
    if not results:
        return f"当前资料不足，无法可靠回答“{query}”。"
    if route == "naive_rag":
        return _naive_template_answer(query, results)
    if route == "multimodal_rag":
        return _multimodal_template_answer(results)
    if route == "graphrag" and results[0].metadata.get("retrieval_mode") == "graphrag_wrapper":
        return "当前只检测到 GraphRAG 资产，尚未取得 global_search 结果，因此不生成具体景点结论。"
    if route == "hybrid_rag":
        return "当前为多源候选聚合结果，尚未声明深度融合完成。请结合下方来源进一步判断。"
    prefix = "基于已检索资料，"
    if confidence == "low" or any(item.metadata.get("answer_policy") == "cautious_source_backed" for item in results):
        prefix = "资料相关但置信度较低，仅基于候选来源谨慎回答："
    snippets = "；".join((item.content.replace("\n", " ")[:160] for item in results[:2]))
    return f"{prefix}{snippets}"


def _naive_template_answer(query: str, results: list[RetrieverResult]) -> str:
    intent = str(results[0].metadata.get("query_intent") or "general")
    merged: dict[str, str] = {}
    for result in results:
        sections = result.metadata.get("sections")
        if not isinstance(sections, dict):
            continue
        for key, value in sections.items():
            text = str(value).strip()
            if text and key not in merged:
                merged[str(key)] = text

    if intent == "itinerary":
        parts = [
            _natural_paragraph(_itinerary_lead(query), merged.get("必打卡景点")),
            _natural_paragraph("交通方面，可以参考：", merged.get("交通安排")),
            _natural_paragraph("游览时还需要注意：", merged.get("实用小贴士")),
        ]
    elif intent == "transport":
        parts = [
            _natural_paragraph("交通方面，可以参考：", merged.get("交通安排")),
            _natural_paragraph("出行前还需要注意：", merged.get("实用小贴士")),
        ]
    elif intent == "attractions":
        parts = [
            _natural_paragraph("景点选择上，可以优先考虑：", merged.get("必打卡景点")),
            _natural_paragraph("游览时还需要注意：", merged.get("实用小贴士")),
        ]
    elif intent == "accommodation":
        parts = [
            _natural_paragraph("住宿方面，可以参考：", merged.get("住宿推荐")),
            _natural_paragraph("预订和入住时还需要注意：", merged.get("实用小贴士")),
        ]
    elif intent == "food":
        parts = [
            _natural_paragraph("当地饮食可以优先了解：", merged.get("美食推荐")),
            _natural_paragraph("体验当地美食时还需要注意：", merged.get("实用小贴士")),
        ]
    else:
        parts = [
            _natural_paragraph("交通方面，可以参考：", merged.get("交通安排")),
            _natural_paragraph("景点选择上，可以优先考虑：", merged.get("必打卡景点")),
            _natural_paragraph("住宿方面，可以参考：", merged.get("住宿推荐")),
            _natural_paragraph("当地饮食可以优先了解：", merged.get("美食推荐")),
            _natural_paragraph("行程中还需要注意：", merged.get("实用小贴士")),
        ]
    answer = "\n\n".join(part for part in parts if part)
    return answer or "当前资料仅命中目的地，但缺少与问题意图对应的证据，无法可靠回答。"


def _multimodal_template_answer(results: list[RetrieverResult]) -> str:
    prefix = "基于已检索资料，"
    max_chars = 1200
    sentences: list[str] = []
    seen: set[str] = set()
    for result in results[:3]:
        if result.metadata.get("evidence_valid") is not True:
            continue
        normalized = re.sub(r"\s+", " ", result.content).strip()
        for match in re.finditer(r"[^。！？.!?]+[。！？.!?]+", normalized):
            sentence = match.group(0).strip()
            key = re.sub(r"\s+", "", sentence)
            if not sentence or key in seen:
                continue
            candidate = "".join(sentences) + sentence
            if len(prefix) + len(candidate) > max_chars:
                continue
            seen.add(key)
            sentences.append(sentence)
    if not sentences:
        return "离线资料中没有找到可整理为完整结论的有效证据，无法生成具体旅游建议。"
    return prefix + "".join(sentences)


def _itinerary_lead(query: str) -> str:
    if "半天" in query:
        return "如果安排半天游，可以优先考虑："
    if "一天" in query or "一日" in query:
        return "如果安排一天游，可以优先考虑："
    if "几日游" in query or re.search(r"[两二三四五六七八九\d]+[天日]游", query):
        return "如果安排多日行程，可以优先考虑："
    return "游玩时可以优先考虑："


def _natural_paragraph(lead: str, content: str | None) -> str:
    if not content:
        return ""
    normalized = re.sub(r"[ \t]+", " ", content).strip()
    if len(normalized) > 1200:
        complete = normalized[:1200]
        boundary = max(complete.rfind(mark) for mark in "。！？.!?")
        if boundary >= 0:
            normalized = complete[: boundary + 1]
    return f"{lead}{normalized}"


def _safe_rewrite(query: str, rewritten: str) -> bool:
    if not rewritten or len(rewritten) > max(80, len(query) * 2):
        return False
    regions = ["香港", "澳门", "台湾", "台北"]
    for region in regions:
        if region in query and region not in rewritten:
            return False
    return True


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
