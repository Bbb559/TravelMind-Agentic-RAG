"""显式验证 GraphRAG 2.7 官方 local_search 的三题专项 smoke。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travelmind.agents import SystemAgent
from travelmind.config import get_settings
from travelmind.graphrag import GraphRAGOfficialLocalSearchAdapter


DEFAULT_QUERIES = [
    "阳朔和张家界哪个更适合看山水风景？",
    "凤凰古城和平遥古城哪个更适合人文古城游？",
    "大理、丽江、香格里拉适合怎么串成一条云南路线？",
]


def classify_case(
    answer: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    trace = list(answer.get("trace") or [])
    retrieved = list(answer.get("retrieved") or [])
    metadata = retrieved[0].get("metadata", {}) if retrieved else {}
    if answer.get("route") != "graphrag":
        return "FAIL", "route_not_graphrag"
    if not any(
        item == "agent:graphrag_agent:start"
        for item in trace
    ):
        return "FAIL", "agent_not_graphrag"
    if metadata.get("retrieval_mode") != "graphrag_local_search":
        return "FAIL", "official_local_mode_missing"
    if diagnostics.get("official_local_called") is not True:
        return "FAIL", "official_local_not_called"
    if diagnostics.get("official_local_succeeded") is not True:
        return "FAIL", "official_local_not_succeeded"
    if metadata.get("graphrag_relevance") is not True:
        return "FAIL", "official_local_low_coverage"
    if not metadata.get("source_summary"):
        return "FAIL", "official_local_source_summary_missing"
    if answer.get("fallback_reason") not in {None, ""}:
        return "FAIL", "unexpected_fallback"
    if "graphrag:global_search_called" in trace:
        return "FAIL", "unexpected_global_search"
    if not str(answer.get("answer") or "").strip():
        return "FAIL", "empty_answer"
    if _contains_unsafe_output(answer):
        return "FAIL", "unsafe_output"
    return "PASS", ""


def run(
    queries: list[str] | None = None,
    settings=None,
    *,
    allow_paid_local_search: bool = False,
    adapter_factory: Callable[[Any], Any] = GraphRAGOfficialLocalSearchAdapter,
) -> dict[str, Any]:
    if not allow_paid_local_search:
        raise PermissionError("allow_paid_local_search_required")

    base_settings = settings or get_settings()
    runtime_settings = replace(
        base_settings,
        llm_enabled=False,
        llm_generate_enabled=False,
        llm_grade_enabled=False,
        llm_rewrite_enabled=False,
        system_agent_llm_router_enabled=False,
        naive_agent_llm_loop_enabled=False,
        graphrag_global_search_enabled=False,
    )
    selected = queries or DEFAULT_QUERIES
    rows = []
    for query in selected:
        adapter = adapter_factory(runtime_settings)
        readiness = adapter.readiness_report()
        answer = SystemAgent(
            runtime_settings,
            None,
            graphrag_local_adapter=adapter,
        ).run(query, allow_global_search=False)
        payload = answer.to_dict()
        diagnostics = dict(adapter.last_diagnostics)
        status, reason = classify_case(payload, diagnostics)
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        if (
            runtime_settings.graphrag_llm_base_url
            and runtime_settings.graphrag_llm_base_url in serialized
        ):
            status, reason = "FAIL", "base_url_leak"
        metadata = (
            payload["retrieved"][0].get("metadata", {})
            if payload["retrieved"]
            else {}
        )
        rows.append(
            {
                "query": query,
                "route": payload["route"],
                "agent": _agent(payload["trace"]),
                "config_ready": readiness.get("config_ready"),
                "key_present": readiness.get("key_present"),
                "index_ready": readiness.get("index_ready"),
                "vector_store_ready": readiness.get("vector_store_ready"),
                "vector_table_ready": readiness.get("vector_table_ready"),
                "official_local_called": diagnostics.get(
                    "official_local_called",
                    False,
                ),
                "official_local_succeeded": diagnostics.get(
                    "official_local_succeeded",
                    False,
                ),
                "official_local_error": diagnostics.get(
                    "official_local_error"
                ),
                "retrieval_mode": metadata.get("retrieval_mode"),
                "entity_coverage": metadata.get("entity_coverage"),
                "source_summary": metadata.get("source_summary") or [],
                "global_search_executed": (
                    "graphrag:global_search_called" in payload["trace"]
                ),
                "fallback_reason": payload.get("fallback_reason"),
                "elapsed_ms": diagnostics.get("elapsed_ms"),
                "answer_preview": str(payload.get("answer") or "")[:300],
                "pass_or_fail": status,
                "failure_reason": reason,
            }
        )
    return {
        "objective": "verify_real_graphrag_api_local_search_invocation",
        "overall_status": (
            "PASS"
            if rows and all(item["pass_or_fail"] == "PASS" for item in rows)
            else "FAIL"
        ),
        "cases": rows,
    }


def _agent(trace: list[str]) -> str | None:
    for item in trace:
        if item.startswith("agent:") and item.endswith(":start"):
            return item.split(":")[1]
    return None


def _contains_unsafe_output(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=False, default=str).lower()
    return any(
        marker in text
        for marker in (
            "api_key",
            "traceback",
            "sk-",
            "sdk_error:",
        )
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-paid-local-search", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid_local_search:
        print(
            json.dumps(
                {
                    "status": "REFUSED",
                    "reason": "allow_paid_local_search_required",
                    "paid_local_search_called": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    payload = run(
        args.query or None,
        allow_paid_local_search=True,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
