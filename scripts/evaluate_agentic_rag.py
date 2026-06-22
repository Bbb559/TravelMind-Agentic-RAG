"""Run the versioned TravelMind quality benchmark.

The offline suite always disables remote LLM, embedding, Official Local, and
Global Search calls. Paid Official Local evaluation requires an explicit flag.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from travelmind.agents import SystemAgent
from travelmind.api import app
from travelmind.config import get_settings
from travelmind.evaluation import (
    EvaluationBundle,
    load_evaluation_bundle,
    manual_faithfulness_metrics,
    percentile,
    route_classification_metrics,
    validate_evaluation_bundle,
    workflow_effect_metrics,
)
from travelmind.graphrag import GraphRAGOfficialLocalSearchAdapter


SAFE_ENVIRONMENT = {
    "TRAVELMIND_LLM_ENABLED": "false",
    "TRAVELMIND_LLM_GENERATE_ENABLED": "false",
    "TRAVELMIND_LLM_GRADE_ENABLED": "false",
    "TRAVELMIND_LLM_REWRITE_ENABLED": "false",
    "TRAVELMIND_SYSTEM_AGENT_LLM_ROUTER_ENABLED": "false",
    "TRAVELMIND_NAIVE_AGENT_LLM_LOOP_ENABLED": "false",
    "TRAVELMIND_LLM_API_KEY": "",
    "TRAVELMIND_EMBEDDING_API_KEY": "",
    "TRAVELMIND_GRAPHRAG_LLM_API_KEY": "",
    "TRAVELMIND_GRAPHRAG_LLM_BASE_URL": "",
    "TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL": "",
    "TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL": "",
    "TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED": "false",
    "TRAVELMIND_RUN_LOG_ENABLED": "false",
    "TRAVELMIND_RUNTIME_PROFILE": "",
}


@contextmanager
def offline_environment() -> Iterator[None]:
    previous = {
        name: os.environ.get(name)
        for name in SAFE_ENVIRONMENT
    }
    os.environ.update(SAFE_ENVIRONMENT)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


def run_offline_evaluation(
    *,
    route_cases: Sequence[dict[str, Any]],
    workflow_cases: Sequence[dict[str, Any]],
    repeats: int = 3,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("evaluation_repeats_must_be_positive")

    route_rows: list[dict[str, Any]] = []
    workflow_rows: list[dict[str, Any]] = []
    route_latencies: list[tuple[str, float]] = []
    workflow_latencies: list[tuple[str, float]] = []
    workflow_agent_latencies: list[tuple[str, float]] = []
    paid_search_calls = 0

    with offline_environment(), TestClient(app) as client:
        client.post("/api/route", json={"query": "你好"})
        client.post("/api/workflow", json={"query": "你好"})

        for case in route_cases:
            payload: dict[str, Any] = {}
            samples: list[float] = []
            for _ in range(repeats):
                started = perf_counter()
                response = client.post(
                    "/api/route",
                    json={
                        "query": case["query"],
                        "allow_global_search": False,
                    },
                )
                elapsed_ms = (perf_counter() - started) * 1000
                response.raise_for_status()
                payload = response.json()
                samples.append(elapsed_ms)
                route_latencies.append(
                    (str(payload.get("route") or "unknown"), elapsed_ms)
                )
            route_rows.append(
                {
                    "id": case["id"],
                    "expected_route": case["expected_route"],
                    "actual_route": payload.get("route"),
                    "correct": payload.get("route") == case["expected_route"],
                    "latency_ms": _latency_summary(samples),
                }
            )

        workflow_payloads: list[dict[str, Any]] = []
        for case in workflow_cases:
            payload = {}
            samples = []
            for _ in range(repeats):
                started = perf_counter()
                response = client.post(
                    "/api/workflow",
                    json={
                        "query": case["query"],
                        "allow_global_search": False,
                    },
                )
                elapsed_ms = (perf_counter() - started) * 1000
                response.raise_for_status()
                payload = response.json()
                samples.append(elapsed_ms)
                workflow_latencies.append(
                    (str(payload.get("route") or "unknown"), elapsed_ms)
                )
                workflow_agent_latencies.append(
                    (
                        str(
                            (
                                payload.get("execution_status")
                                or {}
                            ).get("agent")
                            or "no_agent"
                        ),
                        elapsed_ms,
                    )
                )
                paid_search_calls += int(
                    bool(
                        (payload.get("global_search_status") or {}).get(
                            "executed"
                        )
                    )
                )
                paid_search_calls += sum(
                    1
                    for item in payload.get("retrieved") or []
                    if (item.get("metadata") or {}).get("retrieval_mode")
                    in {
                        "graphrag_local_search",
                        "graphrag_global_search",
                    }
                )
            workflow_payloads.append(payload)
            workflow_rows.append(
                {
                    "id": case["id"],
                    "expected_route": case["expected_route"],
                    "answerable": bool(case.get("answerable")),
                    "latency_ms": _latency_summary(samples),
                    "payload": _safe_payload(payload),
                }
            )

    expected = [str(case["expected_route"]) for case in route_cases]
    predicted = [str(row["actual_route"]) for row in route_rows]
    route_metrics = route_classification_metrics(expected, predicted)
    workflow_metrics = workflow_effect_metrics(
        workflow_cases,
        workflow_payloads,
    )
    all_latencies = [
        value
        for _, value in route_latencies + workflow_latencies
    ]
    return {
        "suite": "offline",
        "benchmark_version": "v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment_summary(repeats),
        "route_metrics": route_metrics,
        "workflow_metrics": workflow_metrics,
        "latency_ms": {
            "overall": _latency_summary(all_latencies),
            "route_api": _latency_summary(
                [value for _, value in route_latencies]
            ),
            "workflow_api": _latency_summary(
                [value for _, value in workflow_latencies]
            ),
            "by_route": _latency_by_route(
                workflow_latencies
            ),
            "by_agent": _latency_by_route(
                workflow_agent_latencies
            ),
        },
        "paid_search_calls": paid_search_calls,
        "route_cases": route_rows,
        "workflow_cases": workflow_rows,
    }


def run_paid_local_evaluation(
    cases: Sequence[dict[str, Any]],
    *,
    allow_paid_local_search: bool,
    adapter_factory: Callable[[Any], Any] = (
        GraphRAGOfficialLocalSearchAdapter
    ),
) -> dict[str, Any]:
    if not allow_paid_local_search:
        raise PermissionError("allow_paid_local_search_required")

    settings = replace(
        get_settings(),
        llm_enabled=False,
        llm_generate_enabled=False,
        llm_grade_enabled=False,
        llm_rewrite_enabled=False,
        system_agent_llm_router_enabled=False,
        naive_agent_llm_loop_enabled=False,
        graphrag_global_search_enabled=False,
    )
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        adapter = adapter_factory(settings)
        started = perf_counter()
        answer = SystemAgent(
            settings,
            None,
            graphrag_local_adapter=adapter,
        ).run(str(case["query"]), allow_global_search=False)
        elapsed_ms = (perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        payload = answer.to_dict()
        retrieved = payload.get("retrieved") or []
        metadata = (
            retrieved[0].get("metadata") or {}
            if retrieved
            else {}
        )
        diagnostics = dict(getattr(adapter, "last_diagnostics", {}) or {})
        global_executed = "graphrag:global_search_called" in (
            payload.get("trace") or []
        )
        succeeded = bool(
            payload.get("route") == case.get("expected_route")
            and metadata.get("retrieval_mode") == "graphrag_local_search"
            and metadata.get("graphrag_relevance") is True
            and metadata.get("source_summary")
            and diagnostics.get("official_local_called") is True
            and diagnostics.get("official_local_succeeded") is True
            and not global_executed
            and not payload.get("fallback_reason")
        )
        rows.append(
            {
                "id": case["id"],
                "route": payload.get("route"),
                "succeeded": succeeded,
                "retrieval_mode": metadata.get("retrieval_mode"),
                "entity_coverage": metadata.get("entity_coverage"),
                "source_summary_count": len(
                    metadata.get("source_summary") or []
                ),
                "official_local_error": diagnostics.get(
                    "official_local_error"
                ),
                "global_search_executed": global_executed,
                "fallback_reason": payload.get("fallback_reason"),
                "latency_ms": elapsed_ms,
            }
        )
    successes = sum(bool(row["succeeded"]) for row in rows)
    return {
        "suite": "paid-local",
        "benchmark_version": "v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sample_count": len(rows),
        "success_count": successes,
        "success_rate": (
            successes / len(rows)
            if rows
            else 0.0
        ),
        "global_search_calls": sum(
            bool(row["global_search_executed"])
            for row in rows
        ),
        "latency_ms": _latency_summary(latencies),
        "cases": rows,
    }


def render_markdown_summary(result: dict[str, Any]) -> str:
    suite = result.get("suite")
    if suite == "offline":
        route = result["route_metrics"]
        workflow = result["workflow_metrics"]
        latency = result["latency_ms"]["workflow_api"]
        return (
            "# TravelMind Offline Evaluation\n\n"
            f"- Benchmark: `{result['benchmark_version']}`\n"
            f"- Samples: {route['sample_count']} route + "
            f"{workflow['sample_count']} workflow\n"
            f"- Route Accuracy: {route['accuracy']:.1%}\n"
            f"- Route Macro-F1: {route['macro_f1']:.3f}\n"
            f"- Evidence Hit@3: {workflow['evidence_hit_at_3']:.1%}\n"
            f"- Safe Refusal Rate: "
            f"{workflow['safe_refusal_rate']:.1%}\n"
            f"- Unsafe Generation Rate: "
            f"{workflow['unsafe_generation_rate']:.1%}\n"
            f"- Workflow latency P50/P95: "
            f"{latency['p50']:.1f}/{latency['p95']:.1f} ms\n"
            f"- Paid search calls: {result['paid_search_calls']}\n"
        )
    if suite == "paid-local":
        return (
            "# TravelMind Paid Official Local Evaluation\n\n"
            f"- Samples: {result['sample_count']}\n"
            f"- Success: {result['success_count']}/"
            f"{result['sample_count']} ({result['success_rate']:.1%})\n"
            f"- Global Search calls: {result['global_search_calls']}\n"
            f"- Latency P50/P95: {result['latency_ms']['p50']:.1f}/"
            f"{result['latency_ms']['p95']:.1f} ms\n"
        )
    if suite == "manual":
        return (
            "# TravelMind Manual Faithfulness Evaluation\n\n"
            f"- Reviewed answers: {result['answer_count']}\n"
            f"- Claim Support Rate: "
            f"{result['claim_support_rate']:.1%}\n"
            f"- Answer Hallucination Rate: "
            f"{result['answer_hallucination_rate']:.1%}\n"
        )
    raise ValueError("unsupported_evaluation_suite")


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": payload.get("route"),
        "fallback_reason": payload.get("fallback_reason"),
        "execution_status": payload.get("execution_status"),
        "global_search_status": payload.get("global_search_status"),
        "retrieved": [
            {
                "metadata": {
                    key: (item.get("metadata") or {}).get(key)
                    for key in (
                        "retrieval_mode",
                        "evidence_valid",
                        "evidence_reason",
                        "matched_entities",
                        "matched_intents",
                        "entity_coverage",
                    )
                    if key in (item.get("metadata") or {})
                }
            }
            for item in payload.get("retrieved") or []
        ],
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0}
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def _latency_by_route(
    rows: Sequence[tuple[str, float]],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = {}
    for route, value in rows:
        grouped.setdefault(route, []).append(value)
    return {
        route: _latency_summary(values)
        for route, values in sorted(grouped.items())
    }


def _environment_summary(repeats: int) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "mode": "offline_no_remote_credentials",
        "repeats_per_case": repeats,
        "single_threaded": True,
    }


def _write_result(
    result: dict[str, Any],
    *,
    json_path: Path | None,
    markdown_path: Path | None,
) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_markdown_summary(result),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("offline", "paid-local", "manual"),
        default="offline",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "evals" / "v1",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--allow-paid-local-search",
        action="store_true",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()

    bundle: EvaluationBundle = load_evaluation_bundle(args.dataset_dir)
    validate_evaluation_bundle(bundle)
    if args.suite == "offline":
        result = run_offline_evaluation(
            route_cases=bundle.route_cases,
            workflow_cases=bundle.workflow_cases,
            repeats=args.repeats,
        )
    elif args.suite == "paid-local":
        if not args.allow_paid_local_search:
            print(
                json.dumps(
                    {
                        "status": "REFUSED",
                        "reason": "allow_paid_local_search_required",
                        "paid_local_search_called": False,
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        result = run_paid_local_evaluation(
            bundle.paid_local_cases,
            allow_paid_local_search=True,
        )
    else:
        try:
            metrics = manual_faithfulness_metrics(
                bundle.manual_annotations
            )
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "status": "REFUSED",
                        "reason": str(exc),
                        "manual_metrics_published": False,
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        result = {
            "suite": "manual",
            "benchmark_version": "v1",
            "generated_at": datetime.now(UTC).isoformat(
                timespec="seconds"
            ),
            **metrics,
        }

    _write_result(
        result,
        json_path=args.output_json,
        markdown_path=args.output_markdown,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
