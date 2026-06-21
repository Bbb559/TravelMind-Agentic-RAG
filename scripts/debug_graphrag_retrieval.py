"""诊断 GraphRAG 资产覆盖、检索结果和相关性评分，不输出任何密钥。"""

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
from travelmind.agents.router import SupervisorRouter
from travelmind.config import get_settings
from travelmind.graphrag import extract_query_entities
from travelmind.schemas import RouteDecision


DEFAULT_QUERIES = [
    "贵州有哪些适合自然风光游的地方？",
    "云南有哪些适合慢旅行的目的地？",
    "阳朔和张家界哪个更适合看山水风景？",
    "凤凰古城和平遥古城哪个更适合人文古城游？",
    "大理、丽江、香格里拉适合怎么串成一条云南路线？",
    "从上海出发有哪些景点适合周末去？",
]

TABLE_COLUMNS = {
    "community_reports.parquet": ("title", "summary", "full_content"),
    "text_units.parquet": ("text",),
    "entities.parquet": ("title", "description"),
    "relationships.parquet": ("source", "target", "description"),
}


def inspect_parquet_assets(query: str) -> dict[str, Any]:
    import pandas as pd

    settings = get_settings()
    entities = extract_query_entities(query)
    tables: dict[str, Any] = {}
    for filename, preferred_columns in TABLE_COLUMNS.items():
        path = settings.graphrag_output_dir / filename
        if not path.exists():
            tables[filename] = {"exists": False, "rows": 0, "columns": [], "entity_hits": {}}
            continue
        frame = pd.read_parquet(path)
        columns = [column for column in preferred_columns if column in frame.columns]
        combined = (
            frame[columns].astype(str).agg("\n".join, axis=1)
            if columns
            else pd.Series([""] * len(frame), dtype=str)
        )
        tables[filename] = {
            "exists": True,
            "rows": len(frame),
            "columns": list(frame.columns),
            "entity_hits": {
                entity: int(combined.str.contains(entity, regex=False, na=False).sum())
                for entity in entities
            },
        }
    return tables


def diagnose_query(
    query: str,
    *,
    allow_paid_global_search: bool = False,
    allow_paid_local_search: bool = False,
) -> dict[str, Any]:
    base_settings = get_settings()
    settings = (
        base_settings
        if allow_paid_global_search or allow_paid_local_search
        else replace(
            base_settings,
            graphrag_llm_api_key="",
            graphrag_llm_base_url="",
            graphrag_llm_chat_model="",
            graphrag_llm_embedding_model="",
        )
    )
    route_decision = SupervisorRouter().route(query)
    diagnostic_decision = RouteDecision(
        query=query,
        route="graphrag",
        confidence="medium",
        reason="GraphRAG 诊断强制执行",
        query_type=route_decision.query_type,
        entities=route_decision.entities,
        matched_terms=route_decision.matched_terms,
    )
    effective_allowed = bool(
        settings.graphrag_global_search_enabled
        and allow_paid_global_search
    )
    agent = GraphRAGAgent(
        settings,
        None,
        max_generate_times=0,
        allow_global_search=allow_paid_global_search,
    )
    config_readiness = agent.retriever.adapter.readiness_report()
    local_readiness = agent.retriever.local_adapter.readiness_report()
    answer = agent.run(query, diagnostic_decision)
    result = answer.retrieved[0] if answer.retrieved else None
    metadata = result.metadata if result else {}
    diagnostics = agent.retriever.last_diagnostics
    if metadata.get("retrieval_mode") == "graphrag_local_search":
        invocation_status = (
            "PASS"
            if diagnostics.get("official_local_called")
            and diagnostics.get("official_local_succeeded")
            else "FAIL"
        )
        quality_status = (
            "PASS"
            if metadata.get("graphrag_relevance") is True
            and bool(metadata.get("source_summary"))
            else "FAIL"
        )
    elif not effective_allowed:
        invocation_status = "SKIP"
        quality_status = (
            "PASS"
            if metadata.get("retrieval_mode") == "graphrag_local_evidence"
            and metadata.get("graphrag_relevance") is True
            else "SKIP"
        )
    elif not config_readiness["ready"]:
        invocation_status = "WARN"
        quality_status = "SKIP"
    elif diagnostics.get("global_search_called") and diagnostics.get("global_search_succeeded"):
        invocation_status = "PASS"
        quality_status = (
            "PASS"
            if metadata.get("retrieval_mode") == "graphrag_global_search"
            and metadata.get("graphrag_relevance") is True
            else "FAIL"
        )
    else:
        invocation_status = "FAIL"
        quality_status = "SKIP"
    return {
        "query": query,
        "route_decision": route_decision.to_dict(),
        "agent_name": agent.agent_name,
        "graphrag_search_query": query,
        "graph_output_path": str(settings.graphrag_output_dir),
        "graph_config_path": str(settings.graphrag_config_dir),
        "config_readiness": {
            "config_ready": config_readiness["config_ready"],
            "key_present": config_readiness["key_present"],
            "index_ready": config_readiness["index_ready"],
            "reason": config_readiness["reason"],
        },
        "official_local_readiness": {
            "config_ready": local_readiness["config_ready"],
            "key_present": local_readiness["key_present"],
            "index_ready": local_readiness["index_ready"],
            "vector_store_ready": local_readiness["vector_store_ready"],
            "vector_table_ready": local_readiness["vector_table_ready"],
            "reason": local_readiness["reason"],
        },
        "official_local_status": {
            "authorized": allow_paid_local_search,
            "executed": bool(diagnostics.get("official_local_called", False)),
            "succeeded": bool(diagnostics.get("official_local_succeeded", False)),
            "error": diagnostics.get("official_local_error"),
        },
        "global_search_status": {
            "requested": allow_paid_global_search,
            "service_enabled": settings.graphrag_global_search_enabled,
            "effective_allowed": effective_allowed,
            "executed": bool(diagnostics.get("global_search_called", False)),
            "succeeded": bool(diagnostics.get("global_search_succeeded", False)),
        },
        "models": {
            "chat": config_readiness["chat_model"],
            "embedding": config_readiness["embedding_model"],
        },
        "parquet": inspect_parquet_assets(query),
        "adapter_diagnostics": diagnostics,
        "global_search_called": diagnostics.get("global_search_called", False),
        "global_search_succeeded": diagnostics.get("global_search_succeeded", False),
        "global_search_elapsed_ms": diagnostics.get("elapsed_ms"),
        "raw_result_type": diagnostics.get("raw_result_type"),
        "raw_result_length": diagnostics.get("raw_result_length", 0),
        "source_summary": metadata.get("source_summary") or [],
        "invocation_status": invocation_status,
        "quality_status": quality_status,
        "normalized_result": _safe_result_summary(result),
        "top_retrieved": {
            "title": result.title if result else None,
            "source_path": result.source_path if result else None,
            "preview": result.content[:500] if result else "",
            "score": result.score if result else None,
        },
        "grade": metadata.get("grade"),
        "llm_grade": metadata.get("llm_grade"),
        "grade_reason": metadata.get("llm_grade_reason") or metadata.get("relevance_reason"),
        "usable_for_answer": metadata.get("usable_for_answer"),
        "final_confidence": answer.confidence,
        "fallback_reason": answer.fallback_reason,
        "final_answer": answer.answer,
        "trace": answer.trace,
    }


def _safe_result_summary(result) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = result.to_dict()
    content = str(payload.pop("content", ""))
    payload["content_preview"] = content[:500]
    payload["content_length"] = len(content)
    return payload


def run(
    queries: list[str],
    *,
    allow_paid_global_search: bool = False,
    allow_paid_local_search: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    query_results = [
        diagnose_query(
            query,
            allow_paid_global_search=allow_paid_global_search,
            allow_paid_local_search=allow_paid_local_search,
        )
        for query in queries
    ]
    return {
        "graph_output_path": str(settings.graphrag_output_dir),
        "graph_config_path": str(settings.graphrag_config_dir),
        "global_search_status": {
            "requested": allow_paid_global_search,
            "service_enabled": settings.graphrag_global_search_enabled,
            "effective_allowed": bool(
                allow_paid_global_search
                and settings.graphrag_global_search_enabled
            ),
            "executed": any(
                item["global_search_status"]["executed"]
                for item in query_results
            ),
            "succeeded": any(
                item["global_search_status"]["succeeded"]
                for item in query_results
            ),
        },
        "official_local_status": {
            "authorized": allow_paid_local_search,
            "executed": any(
                item["official_local_status"]["executed"]
                for item in query_results
            ),
            "succeeded": any(
                item["official_local_status"]["succeeded"]
                for item in query_results
            ),
        },
        "queries": query_results,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-paid-global-search", action="store_true")
    parser.add_argument("--allow-paid-local-search", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    if (
        args.allow_paid_global_search
        and not settings.graphrag_global_search_enabled
    ):
        print(
            json.dumps(
                {
                    "status": "REFUSED",
                    "reason": "global_search_service_disabled",
                    "paid_global_search_called": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    payload = run(
        args.query or DEFAULT_QUERIES,
        allow_paid_global_search=args.allow_paid_global_search,
        allow_paid_local_search=args.allow_paid_local_search,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
