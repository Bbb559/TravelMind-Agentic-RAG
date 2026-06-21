"""HTTP 层 Agentic RAG smoke。

用于验证 FastAPI 契约、Agent 路由、证据边界和默认无付费调用。
默认不读取、不输出任何 key。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


CASES = [
    (" ！！！ ", "invalid_input", "invalid_input"),
    ("贵州荔波小七孔怎么玩比较合适？", "naive_rag", "naive_valid"),
    ("新加坡圣淘沙怎么玩比较合适？", "naive_rag", "no_evidence"),
    ("对比西安和南京的人文景点", "graphrag", "graphrag_relevant_or_safe"),
    ("香港迪士尼怎么玩？", "multimodal_rag", "multimodal_valid"),
    ("台湾和云南哪个更适合亲子游？", "hybrid_rag", None),
]
DEFAULT_TIMEOUT_SECONDS = 300


def post_workflow(base_url: str, query: str, timeout: int) -> dict[str, Any]:
    body = json.dumps(
        {"query": query, "allow_global_search": False},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/workflow",
        data=body,
        headers={"Content-Type": "application/json", "X-TravelMind-Run-Id": "api-smoke"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - local smoke URL
        return json.loads(response.read().decode("utf-8"))


def extract_status(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace") or []
    retrieved = payload.get("retrieved") or []
    retrieval_modes = [
        item.get("metadata", {}).get("retrieval_mode")
        for item in retrieved
        if isinstance(item, dict) and isinstance(item.get("metadata"), dict)
    ]
    return {
        "route": payload.get("route"),
        "runtime_summary": payload.get("runtime_summary"),
        "global_search_status": payload.get("global_search_status"),
        "route_source_llm": any("system:route_source:llm" in item for item in trace),
        "grade_llm": any("grade:llm" in item for item in trace),
        "generate_llm": any("generate:llm" in item for item in trace),
        "naive_faiss": any("naive:retriever_mode:faiss" in item for item in trace) or "faiss" in retrieval_modes,
        "graphrag_global_search": any("graphrag:retriever_mode:global_search" in item for item in trace)
        or "graphrag_global_search" in retrieval_modes,
        "graphrag_local_search": any(
            "graphrag:retriever_mode:local_search" in item for item in trace
        )
        or "graphrag_local_search" in retrieval_modes,
        "multimodal_vector": any("multimodal:retriever_mode:markdown_vector" in item for item in trace)
        or "markdown_vector" in retrieval_modes,
        "retrieval_modes": retrieval_modes,
        "graphrag_relevance": any(
            item.get("metadata", {}).get("graphrag_relevance") is True
            for item in retrieved
            if isinstance(item, dict)
        ),
        "fallback_reason": payload.get("fallback_reason")
        or next(
            (
                item.get("metadata", {}).get("fallback_reason")
                for item in retrieved
                if isinstance(item, dict) and isinstance(item.get("metadata"), dict)
            ),
            None,
        ),
        "execution_status": payload.get("execution_status"),
        "hybrid_branch_status": payload.get("hybrid_branch_status"),
        "evidence_valid": [
            item.get("metadata", {}).get("evidence_valid")
            for item in retrieved
            if isinstance(item, dict) and isinstance(item.get("metadata"), dict)
        ],
    }


def classify_case(payload: dict[str, Any], expected_route: str, expected_mode: str | None) -> tuple[str, str]:
    status = extract_status(payload)
    if status["route"] != expected_route:
        return "FAIL", f"route_mismatch:{status['route']}!={expected_route}"
    execution_status = status.get("execution_status")
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
    runtime_summary = status.get("runtime_summary")
    if not isinstance(runtime_summary, dict) or set(runtime_summary) != {"llm_enabled", "key_present"}:
        return "FAIL", "runtime_summary_not_safe"
    if not all(isinstance(value, bool) for value in runtime_summary.values()):
        return "FAIL", "runtime_summary_not_boolean"
    llm_runtime_ready = bool(
        runtime_summary["llm_enabled"]
        and runtime_summary["key_present"]
    )
    global_search_status = status.get("global_search_status")
    if (
        not isinstance(global_search_status, dict)
        or set(global_search_status)
        != {
            "requested",
            "service_enabled",
            "effective_allowed",
            "executed",
            "succeeded",
        }
        or not all(isinstance(value, bool) for value in global_search_status.values())
    ):
        return "FAIL", "global_search_status_not_safe"
    if (
        global_search_status["requested"]
        or global_search_status["effective_allowed"]
        or global_search_status["executed"]
        or global_search_status["succeeded"]
    ):
        return "FAIL", "paid_global_search_not_disabled"
    if "graphrag_global_search" in status["retrieval_modes"]:
        return "FAIL", "unexpected_paid_global_search"
    if "graphrag_local_search" in status["retrieval_modes"]:
        return "FAIL", "unexpected_paid_local_search"
    if expected_route == "invalid_input":
        if (
            payload.get("answer") != "请先输入具体旅游问题。"
            or payload.get("retrieved")
            or payload.get("fallback_reason") != "invalid_input"
            or execution_status.get("agent") is not None
            or execution_status.get("generation_mode") != "none"
        ):
            return "FAIL", "invalid_input_contract_mismatch"
        return "PASS", ""
    if expected_route == "hybrid_rag":
        trace = payload.get("trace") or []
        if (
            "agent:hybrid_aggregator:start" not in trace
            or "agent:hybrid_aggregator:end" not in trace
        ):
            return "FAIL", "hybrid_aggregator_trace_missing"
        if not isinstance(status.get("hybrid_branch_status"), dict):
            return "FAIL", "hybrid_branch_status_missing"
    if any(
        marker in str(payload.get("answer", ""))
        for marker in ("[Data:", "Reports(", "Entities(", "Relationships(", "Sources(")
    ):
        return "FAIL", "internal_reference_leaked"
    if llm_runtime_ready and not status["route_source_llm"]:
        return "FAIL", "router_not_llm"
    if expected_mode == "no_evidence":
        if (
            status["retrieval_modes"]
            or status["fallback_reason"]
            not in {
                "destination_not_covered",
                "entity_coverage_failed",
                "intent_evidence_missing",
                "no_relevant_evidence",
            }
            or status["generate_llm"]
        ):
            return "FAIL", "no_evidence_contract_mismatch"
        return "PASS", ""
    graph_preview = (
        expected_route == "graphrag"
        and any(
            mode in {"graphrag_local_evidence", "graphrag_wrapper"}
            for mode in status["retrieval_modes"]
        )
    )
    if graph_preview:
        trace = payload.get("trace") or []
        if any(
            item.startswith(("grade:llm", "generate:llm"))
            for item in trace
        ):
            return "FAIL", "local_evidence_formal_answer"
        if not status["fallback_reason"]:
            return "FAIL", "graphrag_preview_missing_fallback"
        if (
            "grade:skipped:evidence_preview_only" not in trace
            or "generate:skipped:evidence_preview_only" not in trace
        ):
            return "FAIL", "graphrag_preview_boundary_missing"
        if not _is_safe_graph_fallback(str(payload.get("answer", ""))):
            return "FAIL", "graphrag_preview_unsafe_answer"
        return "PASS", ""
    if llm_runtime_ready and not status["grade_llm"]:
        return "FAIL", "grade_not_llm"
    graph_safe_fallback = (
        expected_route == "graphrag"
        and bool(status["fallback_reason"])
        and _is_safe_graph_fallback(str(payload.get("answer", "")))
    )
    if (
        llm_runtime_ready
        and not status["generate_llm"]
        and not graph_safe_fallback
    ):
        return "FAIL", "generate_not_llm"
    if expected_mode == "naive_valid":
        if not set(status["retrieval_modes"]) <= {"faiss", "csv"}:
            return "FAIL", f"naive_unsafe_mode:{status['retrieval_modes']}"
        if not status["evidence_valid"] or not all(
            value is True
            for value in status["evidence_valid"]
        ):
            return "FAIL", "naive_invalid_evidence"
    if expected_mode == "graphrag_relevant_or_safe":
        preview_mode = "community_reports_preview" in status["retrieval_modes"]
        real_mode = any(
            mode
            in {
                "graphrag_local_search",
                "graphrag_global_search",
                "graphrag_local_evidence",
            }
            for mode in status["retrieval_modes"]
        )
        if preview_mode:
            return "FAIL", "graphrag_preview_masquerading_as_search"
        if real_mode and not status["graphrag_relevance"]:
            return "FAIL", "graphrag_low_relevance_answer"
        if not real_mode and not graph_safe_fallback:
            return "FAIL", f"graphrag_unsafe_mode:{status['retrieval_modes']}"
    if expected_mode == "multimodal_valid":
        if not set(status["retrieval_modes"]) <= {
            "markdown_vector",
            "markdown_keyword",
        }:
            return "FAIL", f"multimodal_unsafe_mode:{status['retrieval_modes']}"
        if not status["evidence_valid"] or not all(
            value is True
            for value in status["evidence_valid"]
        ):
            return "FAIL", "multimodal_invalid_evidence"
    if expected_route == "graphrag" and _looks_like_raw_english_snippet(payload.get("answer", "")):
        return "FAIL", "graphrag_answer_looks_like_raw_english_snippet"
    return "PASS", ""


def _looks_like_raw_english_snippet(answer: str) -> bool:
    ascii_letters = sum(1 for char in answer if char.isascii() and char.isalpha())
    cjk_chars = sum(1 for char in answer if "\u4e00" <= char <= "\u9fff")
    return ascii_letters > 80 and ascii_letters > cjk_chars * 2


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


def run(base_url: str, timeout: int) -> dict[str, Any]:
    rows = []
    for query, expected_route, expected_mode in CASES:
        try:
            payload = post_workflow(base_url, query, timeout)
            pass_or_fail, failure_reason = classify_case(payload, expected_route, expected_mode)
            status = extract_status(payload)
            rows.append(
                {
                    "query": query,
                    "expected_route": expected_route,
                    "actual_route": payload.get("route"),
                    "runtime_summary": status["runtime_summary"],
                    "global_search_status": status["global_search_status"],
                    "retrieval_modes": status["retrieval_modes"],
                    "route_source_llm": status["route_source_llm"],
                    "grade_llm": status["grade_llm"],
                    "generate_llm": status["generate_llm"],
                    "answer_preview": str(payload.get("answer", ""))[:180],
                    "pass_or_fail": pass_or_fail,
                    "failure_reason": failure_reason,
                }
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "query": query,
                    "expected_route": expected_route,
                    "actual_route": None,
                    "pass_or_fail": "FAIL",
                    "failure_reason": f"http_error:{exc.__class__.__name__}",
                }
            )
    return {"base_url": base_url, "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run(args.base_url, args.timeout)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if any(item["pass_or_fail"] == "FAIL" for item in payload["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
