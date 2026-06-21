import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from travelmind.config import ProjectSettings


def clean_env() -> dict[str, str | None]:
    keys = [
        "TRAVELMIND_LLM_ENABLED",
        "TRAVELMIND_LLM_GENERATE_ENABLED",
        "TRAVELMIND_LLM_GRADE_ENABLED",
        "TRAVELMIND_LLM_REWRITE_ENABLED",
        "TRAVELMIND_SYSTEM_AGENT_LLM_ROUTER_ENABLED",
        "TRAVELMIND_NAIVE_AGENT_LLM_LOOP_ENABLED",
        "TRAVELMIND_GRAPHRAG_LLM_API_KEY",
        "TRAVELMIND_GRAPHRAG_LLM_BASE_URL",
        "TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL",
        "TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL",
        "TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED",
        "TRAVELMIND_HYBRID_BRANCH_TIMEOUT_SECONDS",
        "TRAVELMIND_RUN_LOG_ENABLED",
        "TRAVELMIND_RUNTIME_PROFILE",
    ]
    old = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    return old


def restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_settings(**overrides) -> ProjectSettings:
    from travelmind.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    defaults = {
        "llm_enabled": False,
        "llm_generate_enabled": False,
        "llm_grade_enabled": False,
        "llm_rewrite_enabled": False,
        "system_agent_llm_router_enabled": False,
        "naive_agent_llm_loop_enabled": False,
        "llm_api_key": "",
        "embedding_api_key": "",
        "graphrag_llm_api_key": "",
        "graphrag_llm_base_url": "",
        "graphrag_llm_chat_model": "",
        "graphrag_llm_embedding_model": "",
        "graphrag_global_search_enabled": False,
    }
    defaults.update(overrides)
    return replace(settings, **defaults)


class FakeLLM:
    def __init__(self, responses=None, error: Exception | None = None):
        self.responses = list(responses or [])
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return '{"route":"naive_rag","confidence":"high","reason":"fake","query_type":"fake","entities":[],"matched_terms":[]}'


def fake_doc(content: str, **metadata):
    return SimpleNamespace(page_content=content, metadata=metadata)
