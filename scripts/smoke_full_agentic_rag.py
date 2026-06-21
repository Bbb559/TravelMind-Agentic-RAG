"""全链路 Agentic RAG smoke。

不输出任何 key，只输出 readiness 和 case 结果。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travelmind.config import get_settings
from travelmind.graphrag import (
    GraphRAGGlobalSearchAdapter,
    GraphRAGOfficialLocalSearchAdapter,
)
from travelmind.graphs import AgenticRAGWorkflow


CASES = [
    ("贵州荔波小七孔怎么玩比较合适？", "naive_rag"),
    ("成都都江堰适合怎么玩？", "naive_rag"),
    ("大理双廊怎么安排半天游？", "naive_rag"),
    ("新加坡圣淘沙怎么玩比较合适？", "naive_rag"),
    ("阳朔有哪些必打卡景点？", "naive_rag"),
    ("大理到双廊怎么去？", "naive_rag"),
    ("成都有什么美食？", "naive_rag"),
    ("西安有哪些历史景点？", "naive_rag"),
    ("对比西安和南京的人文景点", "graphrag"),
    ("阳朔和张家界哪个更适合看山水风景？", "graphrag"),
    ("大理、丽江、香格里拉适合怎么串成一条云南路线？", "graphrag"),
    ("香港迪士尼怎么玩？", "multimodal_rag"),
    ("澳门大三巴牌坊在哪里？", "multimodal_rag"),
    ("台北101有什么看点？", "multimodal_rag"),
    ("台湾有哪些适合亲子游的地方？", "multimodal_rag"),
    ("台湾和西安哪个更适合亲子游？", "hybrid_rag"),
    ("香港和成都哪个更适合周末游？", "hybrid_rag"),
    ("澳门和南京的人文景点有什么区别？", "hybrid_rag"),
    ("qwxjkp", "invalid_input"),
    ("今天天气怎么样？", "fallback"),
    ("帮我写一段 Python 快排", "fallback"),
    ("上传一张景区照片并询问适合的游玩路线", "fallback"),
    ("现在解析这个 PDF 里的图片和文字", "fallback"),
    ("火星三日游怎么安排？", "fallback"),
]


def readiness() -> dict[str, bool]:
    settings = get_settings()
    return {
        "llm_ready": bool(settings.llm_api_key),
        "embedding_ready": bool(settings.embedding_api_key),
        "naive_faiss_ready": bool(
            settings.embedding_api_key
            and (settings.faiss_index_dir / "index.faiss").exists()
            and (settings.faiss_index_dir / "index.pkl").exists()
        ),
        "graphrag_ready": GraphRAGGlobalSearchAdapter(settings).readiness()[0],
        "graphrag_official_local_ready": GraphRAGOfficialLocalSearchAdapter(
            settings
        ).readiness()[0],
        "graphrag_local_evidence_ready": all(
            (settings.graphrag_output_dir / name).exists()
            for name in ("text_units.parquet", "entities.parquet", "relationships.parquet")
        ),
        "multimodal_vector_ready": bool(
            settings.embedding_api_key
            and (settings.multimodal_markdown_dir / "index.faiss").exists()
            and (settings.multimodal_markdown_dir / "index.pkl").exists()
        ),
    }


def extract(answer: dict[str, Any]) -> dict[str, Any]:
    trace = answer["trace"]
    retrieved = answer["retrieved"]
    metadata = retrieved[0]["metadata"] if retrieved else {}
    retrieval_modes = [
        item.get("metadata", {}).get("retrieval_mode")
        for item in retrieved
        if isinstance(item, dict) and isinstance(item.get("metadata"), dict)
    ]
    return {
        "route_source": _first(trace, "system:route_source:"),
        "agent_name": _agent(trace),
        "retrieval_mode": metadata.get("retrieval_mode"),
        "retrieval_modes": retrieval_modes,
        "global_search_available": metadata.get("global_search_available"),
        "graphrag_relevance": metadata.get("graphrag_relevance"),
        "entity_coverage": metadata.get("entity_coverage"),
        "grade_status": _first(trace, "grade:"),
        "rewrite_status": _first(trace, "rewrite:"),
        "generate_status": _first(trace, "generate:"),
        "fallback_reason": answer.get("fallback_reason") or metadata.get("fallback_reason"),
        "execution_status": answer.get("execution_status"),
        "hybrid_branch_status": answer.get("hybrid_branch_status"),
        "evidence_valid": [
            item.get("metadata", {}).get("evidence_valid")
            for item in retrieved
            if isinstance(item, dict) and isinstance(item.get("metadata"), dict)
        ],
    }


def classify(query: str, expected: str, answer: dict[str, Any], ready: dict[str, bool], strict: bool) -> tuple[str, str]:
    info = extract(answer)
    if answer["route"] != expected:
        return "FAIL", f"route_mismatch:{answer['route']}!= {expected}"
    execution_status = answer.get("execution_status")
    if (
        not isinstance(execution_status, dict)
        or set(execution_status)
        != {
            "agent",
            "retrieval_mode",
            "evidence_status",
            "generation_mode",
            "llm_stages",
        }
    ):
        return "FAIL", "execution_status_not_safe"
    if expected == "invalid_input":
        if (
            answer.get("answer") != "请先输入具体旅游问题。"
            or answer.get("fallback_reason") != "invalid_input"
            or answer.get("retrieved")
            or execution_status.get("agent") is not None
            or execution_status.get("retrieval_mode") != "none"
            or execution_status.get("generation_mode") != "none"
        ):
            return "FAIL", "invalid_input_contract_mismatch"
        if any(
            step.startswith(
                ("system:route", "agent:", "retrieve:", "grade:", "rewrite:", "generate:")
            )
            for step in answer.get("trace", [])
        ):
            return "FAIL", "invalid_input_executed_workflow"
        return "PASS", ""
    unsupported_terms = ["天气", "Python", "快排", "上传", "照片", "PDF", "火星", "OCR", "VLM"]
    if expected == "fallback":
        if any(term in answer["answer"] for term in ["可以上传", "天气是", "代码如下", "PDF解析完成"]):
            return "FAIL", "unsupported_capability_claim"
        if answer["route"] == "fallback" and answer.get("fallback_reason"):
            return "PASS", ""
        return "PASS", ""
    if expected == "hybrid_rag" and "深度融合完成" in answer["answer"] and "未" not in answer["answer"] and "不" not in answer["answer"]:
        return "FAIL", "hybrid_claims_deep_fusion"
    if expected == "hybrid_rag" and (
        "agent:hybrid_aggregator:start" not in answer["trace"]
        or "agent:hybrid_aggregator:end" not in answer["trace"]
    ):
        return "FAIL", "hybrid_aggregator_trace_missing"
    if expected == "hybrid_rag" and not isinstance(
        answer.get("hybrid_branch_status"),
        dict,
    ):
        return "FAIL", "hybrid_branch_status_missing"
    if any(
        marker in str(answer.get("answer", ""))
        for marker in ("[Data:", "Reports(", "Entities(", "Relationships(", "Sources(")
    ):
        return "FAIL", "internal_reference_leaked"
    if "graphrag_global_search" in info["retrieval_modes"]:
        return "FAIL", "unexpected_paid_global_search"
    if "graphrag_local_search" in info["retrieval_modes"]:
        return "FAIL", "unexpected_paid_local_search"
    if strict and ready["llm_ready"] and "system:route_source:llm" not in answer["trace"]:
        return "FAIL", "router_not_llm"
    source_safe_fallback = (
        expected in {"naive_rag", "multimodal_rag"}
        and not answer.get("retrieved")
        and info["fallback_reason"]
        in {
            "destination_not_covered",
            "entity_coverage_failed",
            "intent_evidence_missing",
            "no_relevant_evidence",
        }
    )
    if source_safe_fallback:
        if any(step.startswith("generate:llm") for step in answer["trace"]):
            return "FAIL", "no_evidence_generated_answer"
        return "PASS", ""
    graph_preview = (
        expected == "graphrag"
        and info["retrieval_mode"]
        in {"graphrag_local_evidence", "graphrag_wrapper"}
    )
    if graph_preview:
        if any(
            item.startswith(("grade:llm", "generate:llm"))
            for item in answer["trace"]
        ):
            return "FAIL", "local_evidence_formal_answer"
        if not info["fallback_reason"]:
            return "FAIL", "graphrag_preview_missing_fallback"
        if (
            "grade:skipped:evidence_preview_only" not in answer["trace"]
            or "generate:skipped:evidence_preview_only" not in answer["trace"]
        ):
            return "FAIL", "graphrag_preview_boundary_missing"
        if not _is_safe_graph_fallback(answer.get("answer", "")):
            return "FAIL", "graphrag_preview_unsafe_answer"
        return "PASS", ""
    graph_safe_fallback = (
        expected == "graphrag"
        and bool(info["fallback_reason"])
        and _is_safe_graph_fallback(answer.get("answer", ""))
    )
    if strict and ready["llm_ready"] and "generate:llm" not in answer["trace"] and not graph_safe_fallback:
        return "FAIL", "generate_not_llm"
    if expected == "naive_rag":
        if info["retrieval_mode"] not in {"faiss", "csv"}:
            return "FAIL", f"naive_unsafe_mode:{info['retrieval_mode']}"
        if not info["evidence_valid"] or not all(
            value is True
            for value in info["evidence_valid"]
        ):
            return "FAIL", "naive_invalid_evidence"
    if expected == "graphrag":
        if info["retrieval_mode"] == "community_reports_preview":
            return "FAIL", "graphrag_preview_masquerading_as_search"
        if info["retrieval_mode"] in {
            "graphrag_local_search",
            "graphrag_global_search",
            "graphrag_local_evidence",
        }:
            if info["graphrag_relevance"] is not True:
                return "FAIL", "graphrag_low_relevance_answer"
        elif graph_safe_fallback:
            return "PASS", ""
        else:
            return "FAIL", f"graphrag_unsafe_mode:{info['retrieval_mode']}"
    if expected == "multimodal_rag":
        if info["retrieval_mode"] not in {"markdown_vector", "markdown_keyword"}:
            return "FAIL", f"multimodal_unsafe_mode:{info['retrieval_mode']}"
        if not info["evidence_valid"] or not all(
            value is True
            for value in info["evidence_valid"]
        ):
            return "FAIL", "multimodal_invalid_evidence"
    return "PASS", ""


def run(strict: bool, limit: int | None) -> dict[str, Any]:
    for name in (
        "TRAVELMIND_LLM_API_KEY",
        "TRAVELMIND_EMBEDDING_API_KEY",
        "TRAVELMIND_GRAPHRAG_LLM_API_KEY",
        "TRAVELMIND_GRAPHRAG_LLM_BASE_URL",
        "TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL",
        "TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL",
    ):
        os.environ[name] = ""
    os.environ["TRAVELMIND_LLM_ENABLED"] = "true"
    os.environ["TRAVELMIND_SYSTEM_AGENT_LLM_ROUTER_ENABLED"] = "true"
    os.environ["TRAVELMIND_LLM_GENERATE_ENABLED"] = "true"
    os.environ["TRAVELMIND_LLM_GRADE_ENABLED"] = "true"
    os.environ["TRAVELMIND_LLM_REWRITE_ENABLED"] = "true"
    os.environ["TRAVELMIND_NAIVE_AGENT_LLM_LOOP_ENABLED"] = "true"
    os.environ["TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED"] = "false"
    os.environ["TRAVELMIND_GRAPHRAG_CONFIG_DIR"] = str(
        ROOT / ".runtime" / "smoke_disabled_graphrag_config"
    )
    get_settings.cache_clear()
    workflow = AgenticRAGWorkflow()
    ready = readiness()
    selected = CASES if limit is None else CASES[:limit]
    rows = []
    for query, expected in selected:
        answer = workflow.run(query, allow_global_search=False).to_dict()
        status, reason = classify(query, expected, answer, ready, strict)
        info = extract(answer)
        rows.append(
            {
                "query": query,
                "expected_route": expected,
                "actual_route": answer["route"],
                **info,
                "sources_count": len(answer["sources"]),
                "retrieved_count": len(answer["retrieved"]),
                "trace": answer["trace"],
                "answer": answer["answer"],
                "pass_or_fail": status,
                "failure_reason": reason,
            }
        )
    return {"readiness": ready, "cases": rows}


def _first(trace: list[str], prefix: str) -> str | None:
    return next((item for item in trace if item.startswith(prefix)), None)


def _agent(trace: list[str]) -> str | None:
    if "agent:hybrid_aggregator:start" in trace:
        return "hybrid_aggregator"
    for item in trace:
        if item.startswith("agent:") and item.endswith(":start"):
            return item.split(":")[1]
    return None


def _is_safe_graph_fallback(answer: str) -> bool:
    return any(
        phrase in answer
        for phrase in (
            "不生成具体旅游结论",
            "无法可靠回答",
            "当前资料不足",
            "未检索到",
            "索引未覆盖",
            "正式结论",
            "仅供预览",
            "Global Search 未开启",
            "官方 GraphRAG Local Search 未成功",
            "不代表官方 Local Search 正式回答",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    strict = args.strict or not args.non_strict
    payload = run(strict=strict, limit=args.limit)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if any(case["pass_or_fail"] == "FAIL" for case in payload["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
