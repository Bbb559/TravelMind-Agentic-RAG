"""启动普通 HTTP smoke 使用的安全 FastAPI 进程。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def prepare_safe_environment() -> None:
    for name in (
        "TRAVELMIND_LLM_API_KEY",
        "TRAVELMIND_EMBEDDING_API_KEY",
        "TRAVELMIND_GRAPHRAG_LLM_API_KEY",
        "TRAVELMIND_GRAPHRAG_LLM_BASE_URL",
        "TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL",
        "TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL",
    ):
        os.environ[name] = ""
    os.environ["TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED"] = "false"
    os.environ["TRAVELMIND_GRAPHRAG_CONFIG_DIR"] = str(
        ROOT / ".runtime" / "smoke_disabled_graphrag_config"
    )
    os.environ["TRAVELMIND_LLM_ENABLED"] = "false"
    os.environ["TRAVELMIND_SYSTEM_AGENT_LLM_ROUTER_ENABLED"] = "false"
    os.environ["TRAVELMIND_LLM_GENERATE_ENABLED"] = "false"
    os.environ["TRAVELMIND_LLM_GRADE_ENABLED"] = "false"
    os.environ["TRAVELMIND_LLM_REWRITE_ENABLED"] = "false"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    prepare_safe_environment()
    import uvicorn

    uvicorn.run(
        "travelmind.api:app",
        host=args.host,
        port=args.port,
        log_level="error",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
