"""验证 GraphRAG 2.7 官方 global_search 调用与结果质量的专项 smoke。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travelmind.agents.graphrag import GraphRAGAgent
from travelmind.config import get_settings
from travelmind.graphrag import GraphRAGGlobalSearchAdapter
from travelmind.retrievers import GraphRAGSearchRetriever
from travelmind.schemas import RouteDecision


DEFAULT_QUERIES = [
    "阳朔和张家界哪个更适合看山水风景？",
    "凤凰古城和平遥古城哪个更适合人文古城游？",
    "大理、丽江、香格里拉适合怎么串成一条云南路线？",
]


def classify_case(
    readiness: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    retrieval_mode: str | None,
    global_search_available: bool,
    graphrag_relevance: bool | None,
) -> tuple[str, str]:
    prerequisites_ready = all(
        readiness.get(key) is True
        for key in ("config_ready", "key_present", "index_ready")
    )
    if not prerequisites_ready:
        return "WARN", "SKIP"
    if diagnostics.get("global_search_called") is not True:
        return "FAIL", "SKIP"
    if diagnostics.get("global_search_succeeded") is not True:
        return "FAIL", "SKIP"
    quality_passed = (
        retrieval_mode == "graphrag_global_search"
        and global_search_available is True
        and graphrag_relevance is True
    )
    return "PASS", "PASS" if quality_passed else "FAIL"


def classify_three_layers(
    readiness: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    retrieval_mode: str | None,
    global_search_available: bool,
    graphrag_relevance: bool | None,
    answer: dict[str, Any],
) -> tuple[str, str, str]:
    invocation_status, retrieval_quality_status = classify_case(
        readiness,
        diagnostics,
        retrieval_mode=retrieval_mode,
        global_search_available=global_search_available,
        graphrag_relevance=graphrag_relevance,
    )
    answer_status = _classify_answer(
        invocation_status,
        retrieval_quality_status,
        answer,
    )
    return invocation_status, retrieval_quality_status, answer_status


def _classify_answer(
    invocation_status: str,
    retrieval_quality_status: str,
    answer: dict[str, Any],
) -> str:
    text = str(answer.get("answer") or "").strip()
    if not text or _contains_unsafe_diagnostics(text):
        return "FAIL"
    if invocation_status == "PASS" and retrieval_quality_status == "PASS":
        return (
            "PASS"
            if answer.get("sources") and answer.get("fallback_reason") in {None, ""}
            else "FAIL"
        )
    safe_fallback = answer.get("confidence") == "low" or any(
        phrase in text
        for phrase in (
            "不生成具体旅游结论",
            "无法可靠回答",
            "当前资料不足",
            "未检索到",
            "索引未覆盖",
        )
    )
    return "PASS" if safe_fallback else "FAIL"


def _contains_unsafe_diagnostics(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "api_key",
            "qwen_api_key",
            "sk-",
            "traceback",
            "global_search_error:",
        )
    )


def run(
    queries: list[str] | None = None,
    settings=None,
    *,
    allow_paid_global_search: bool = False,
) -> dict[str, Any]:
    base_settings = settings or get_settings()
    if not allow_paid_global_search:
        raise PermissionError("allow_paid_global_search_required")
    if not base_settings.graphrag_global_search_enabled:
        raise PermissionError("global_search_service_disabled")
    runtime_settings = replace(
        base_settings,
        llm_enabled=False,
        llm_generate_enabled=False,
        llm_grade_enabled=False,
        llm_rewrite_enabled=False,
        system_agent_llm_router_enabled=False,
        naive_agent_llm_loop_enabled=False,
    )
    selected = queries or DEFAULT_QUERIES
    cases = [_run_case(runtime_settings, query) for query in selected]
    summary = summarize_cases(cases)
    return {
        "objective": "verify_real_graphrag_api_global_search_invocation",
        **summary,
        "quality_review_recommended": summary["recommended_action"] == "QUALITY_REVIEW",
        "cases": cases,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, str]:
    invocation_statuses = [case["invocation_status"] for case in cases]
    retrieval_statuses = [case["retrieval_quality_status"] for case in cases]
    answer_statuses = [case["answer_status"] for case in cases]
    overall_invocation_status = _overall_invocation_status(invocation_statuses)
    overall_retrieval_quality_status = _overall_retrieval_status(retrieval_statuses)
    overall_answer_status = "FAIL" if "FAIL" in answer_statuses else "PASS"
    overall_status, recommended_action = _audit_gate(
        overall_invocation_status,
        overall_retrieval_quality_status,
        overall_answer_status,
    )
    return {
        "overall_invocation_status": overall_invocation_status,
        "overall_retrieval_quality_status": overall_retrieval_quality_status,
        "overall_answer_status": overall_answer_status,
        "overall_status": overall_status,
        "recommended_action": recommended_action,
    }


def _run_case(settings, query: str) -> dict[str, Any]:
    adapter = GraphRAGGlobalSearchAdapter(settings)
    readiness = adapter.readiness_report()
    agent = GraphRAGAgent(
        settings,
        None,
        max_generate_times=0,
        allow_global_search=True,
        graphrag_adapter=adapter,
    )
    agent.retriever = GraphRAGSearchRetriever(
        settings,
        adapter=adapter,
        allow_global_search=True,
    )
    decision = RouteDecision(
        query=query,
        route="graphrag",
        confidence="medium",
        reason="paid global_search audit",
        query_type="global_search_audit",
    )
    answer = agent.run(
        query,
        decision,
        ["workflow:start", "system:route:graphrag", "route:graphrag"],
    )
    result = answer.retrieved[0] if answer.retrieved else None
    metadata = result.metadata if result else {}
    diagnostics = agent.retriever.last_diagnostics
    retrieval_mode = metadata.get("retrieval_mode")
    global_search_available = metadata.get("global_search_available") is True
    graphrag_relevance = metadata.get("graphrag_relevance")
    invocation_status, retrieval_quality_status, answer_status = classify_three_layers(
        readiness,
        diagnostics,
        retrieval_mode=retrieval_mode,
        global_search_available=global_search_available,
        graphrag_relevance=graphrag_relevance,
        answer=answer.to_dict(),
    )
    return {
        "query": query,
        "graph_config_path": readiness.get("config_path"),
        "graph_output_path": readiness.get("graph_output_path"),
        "config_ready": readiness.get("config_ready"),
        "key_present": readiness.get("key_present"),
        "index_ready": readiness.get("index_ready"),
        "chat_model": readiness.get("chat_model"),
        "embedding_model": readiness.get("embedding_model"),
        "global_search_called": diagnostics.get("global_search_called", False),
        "global_search_succeeded": diagnostics.get("global_search_succeeded", False),
        "retrieval_mode": retrieval_mode,
        "global_search_available": global_search_available,
        "graphrag_relevance": graphrag_relevance,
        "fallback_reason": diagnostics.get("fallback_reason") or metadata.get("fallback_reason"),
        "elapsed_ms": diagnostics.get("elapsed_ms"),
        "raw_result_type": diagnostics.get("raw_result_type"),
        "raw_result_length": diagnostics.get("raw_result_length", 0),
        "invocation_status": invocation_status,
        "retrieval_quality_status": retrieval_quality_status,
        "quality_status": retrieval_quality_status,
        "answer_status": answer_status,
        "trace": answer.trace,
        "top_sources": [source.to_dict() for source in answer.sources[:3]],
        "answer_preview": answer.answer[:300],
        "answer_confidence": answer.confidence,
        "answer_fallback_reason": answer.fallback_reason,
    }


def _overall_invocation_status(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _overall_retrieval_status(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if statuses and all(status == "PASS" for status in statuses):
        return "PASS"
    return "SKIP"


def _audit_gate(
    invocation_status: str,
    retrieval_quality_status: str,
    answer_status: str,
) -> tuple[str, str]:
    if invocation_status == "FAIL" or answer_status == "FAIL":
        return "FAIL", "FIX_REQUIRED"
    if invocation_status == "WARN":
        return "WARN", "ENVIRONMENT_REQUIRED"
    if retrieval_quality_status == "FAIL":
        return "PASS_WITH_QUALITY_GAP", "QUALITY_REVIEW"
    return "PASS", "READY"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-paid-global-search", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(
            args.query or None,
            allow_paid_global_search=args.allow_paid_global_search,
        )
    except PermissionError as exc:
        print(
            json.dumps(
                {
                    "status": "REFUSED",
                    "reason": str(exc),
                    "paid_global_search_called": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
