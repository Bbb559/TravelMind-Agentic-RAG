"""TravelMind 项目配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class ProjectSettings:
    assets_dir: Path
    travel_csv_path: Path
    faiss_index_dir: Path
    graphrag_config_dir: Path
    graphrag_output_dir: Path
    multimodal_markdown_dir: Path
    llm_enabled: bool
    llm_generate_enabled: bool
    llm_grade_enabled: bool
    llm_rewrite_enabled: bool
    system_agent_llm_router_enabled: bool
    naive_agent_llm_loop_enabled: bool
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    embedding_api_key: str
    embedding_model: str
    graphrag_llm_api_key: str
    graphrag_llm_base_url: str
    graphrag_llm_chat_model: str
    graphrag_llm_embedding_model: str
    graphrag_timeout_seconds: int
    graphrag_max_context_chars: int
    graphrag_global_search_enabled: bool
    hybrid_branch_timeout_seconds: int
    run_log_enabled: bool

    @classmethod
    def load(cls, env_path: Path | None = None) -> "ProjectSettings":
        root = Path(__file__).resolve().parents[3]
        resolved_env_path = env_path or (root / ".env")
        if load_dotenv is not None and resolved_env_path.exists():
            load_dotenv(resolved_env_path, override=False, encoding="utf-8-sig")

        assets_dir = _path_env("TRAVELMIND_ASSETS_DIR", root / "assets").resolve()
        runtime_profile = os.getenv("TRAVELMIND_RUNTIME_PROFILE", "").strip().lower()
        demo_profile_enabled = runtime_profile == "full_agentic_demo"
        return cls(
            assets_dir=assets_dir,
            travel_csv_path=_path_env("TRAVELMIND_TRAVEL_CSV", assets_dir / "travel_guide.csv").resolve(),
            faiss_index_dir=_path_env("TRAVELMIND_FAISS_INDEX_DIR", assets_dir / "faiss_index").resolve(),
            graphrag_config_dir=_path_env(
                "TRAVELMIND_GRAPHRAG_CONFIG_DIR", assets_dir / "graphrag_runtime"
            ).resolve(),
            graphrag_output_dir=_path_env(
                "TRAVELMIND_GRAPHRAG_OUTPUT_DIR", assets_dir / "graphrag_output"
            ).resolve(),
            multimodal_markdown_dir=_path_env(
                "TRAVELMIND_MULTIMODAL_MARKDOWN_DIR", assets_dir / "result_markdown"
            ).resolve(),
            llm_enabled=_bool_env("TRAVELMIND_LLM_ENABLED", demo_profile_enabled),
            llm_generate_enabled=_bool_env("TRAVELMIND_LLM_GENERATE_ENABLED", demo_profile_enabled),
            llm_grade_enabled=_bool_env("TRAVELMIND_LLM_GRADE_ENABLED", demo_profile_enabled),
            llm_rewrite_enabled=_bool_env("TRAVELMIND_LLM_REWRITE_ENABLED", demo_profile_enabled),
            system_agent_llm_router_enabled=_bool_env(
                "TRAVELMIND_SYSTEM_AGENT_LLM_ROUTER_ENABLED", demo_profile_enabled
            ),
            naive_agent_llm_loop_enabled=_bool_env("TRAVELMIND_NAIVE_AGENT_LLM_LOOP_ENABLED", demo_profile_enabled),
            llm_model=os.getenv("TRAVELMIND_LLM_MODEL") or "deepseek-chat",
            llm_base_url=os.getenv("TRAVELMIND_LLM_BASE_URL") or "https://api.deepseek.com",
            llm_api_key=os.getenv("TRAVELMIND_LLM_API_KEY") or "",
            embedding_api_key=os.getenv("TRAVELMIND_EMBEDDING_API_KEY") or "",
            embedding_model=os.getenv("TRAVELMIND_EMBEDDING_MODEL") or "text-embedding-v3",
            graphrag_llm_api_key=os.getenv("TRAVELMIND_GRAPHRAG_LLM_API_KEY") or "",
            graphrag_llm_base_url=os.getenv("TRAVELMIND_GRAPHRAG_LLM_BASE_URL") or "",
            graphrag_llm_chat_model=os.getenv("TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL") or "",
            graphrag_llm_embedding_model=os.getenv("TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL") or "",
            graphrag_timeout_seconds=int(os.getenv("TRAVELMIND_GRAPHRAG_TIMEOUT_SECONDS") or "180"),
            graphrag_max_context_chars=int(os.getenv("TRAVELMIND_GRAPHRAG_MAX_CONTEXT_CHARS") or "6000"),
            graphrag_global_search_enabled=_bool_env(
                "TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED",
                False,
            ),
            hybrid_branch_timeout_seconds=int(
                os.getenv("TRAVELMIND_HYBRID_BRANCH_TIMEOUT_SECONDS") or "20"
            ),
            run_log_enabled=_bool_env("TRAVELMIND_RUN_LOG_ENABLED", False),
        )


@lru_cache(maxsize=1)
def get_settings() -> ProjectSettings:
    return ProjectSettings.load()
