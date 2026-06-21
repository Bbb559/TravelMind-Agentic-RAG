"""FastAPI 入口。"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from travelmind.config import get_settings
from travelmind.data import inventory
from travelmind.graphs import AgenticRAGWorkflow

app = FastAPI(title="TravelMind API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-TravelMind-Run-Id"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_LOG_DIR = PROJECT_ROOT / ".runtime" / "run_logs"


class QueryRequest(BaseModel):
    query: str
    allow_global_search: bool = False


def _safe_run_id(raw: str | None) -> str:
    candidate = (raw or "").strip()
    if not candidate:
        return uuid.uuid4().hex
    candidate = re.sub(r"[^a-zA-Z0-9_.-]", "-", candidate)[:80].strip(".-")
    return candidate or uuid.uuid4().hex


def _write_run_log(endpoint: str, run_id: str, payload: dict[str, Any]) -> None:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{run_id}_{endpoint}.json"
    log_payload = {
        "run_id": run_id,
        "endpoint": endpoint,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "payload": payload,
    }
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _with_run_log(request: Request, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _safe_run_id(request.headers.get("X-TravelMind-Run-Id"))
    response_payload = dict(payload)
    response_payload["run_id"] = run_id
    if not get_settings().run_log_enabled:
        return response_payload
    try:
        _write_run_log(endpoint, run_id, response_payload)
    except OSError:
        # 本地调试日志不能成为 API 响应的硬依赖。
        pass
    return response_payload


def runtime_summary() -> dict[str, bool]:
    settings = get_settings()
    return {
        "llm_enabled": bool(settings.llm_enabled),
        "key_present": bool(settings.llm_api_key),
    }


def global_search_status(
    *,
    requested: bool,
    route: str,
    trace: list[str],
) -> dict[str, bool]:
    service_enabled = bool(get_settings().graphrag_global_search_enabled)
    executed = "graphrag:global_search_called" in trace
    succeeded = "graphrag:global_search_succeeded" in trace
    return {
        "requested": requested,
        "service_enabled": service_enabled,
        "effective_allowed": bool(
            requested
            and service_enabled
            and route in {"graphrag", "hybrid_rag"}
        ),
        "executed": executed,
        "succeeded": succeeded,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/inventory")
def api_inventory() -> dict:
    return inventory()


@app.post("/api/route")
def api_route(body: QueryRequest, request: Request) -> dict:
    payload = AgenticRAGWorkflow().route(body.query).to_dict()
    return _with_run_log(request, "route", payload)


@app.post("/api/workflow")
def api_workflow(body: QueryRequest, request: Request) -> dict:
    payload = AgenticRAGWorkflow().run(
        body.query,
        allow_global_search=body.allow_global_search,
    ).to_dict()
    payload["runtime_summary"] = runtime_summary()
    payload["global_search_status"] = global_search_status(
        requested=body.allow_global_search,
        route=str(payload.get("route") or ""),
        trace=list(payload.get("trace") or []),
    )
    return _with_run_log(request, "workflow", payload)
