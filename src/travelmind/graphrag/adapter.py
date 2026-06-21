"""真实 GraphRAG global_search API 的隔离适配器。"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Awaitable, Callable

from travelmind.config import ProjectSettings
from travelmind.graphrag.official_result import (
    normalize_official_response,
    summarize_official_context,
)


class GraphRAGGlobalSearchAdapter:
    """集中处理 GraphRAG 依赖、配置、超时与 API 参数适配。"""

    REQUIRED = ("entities.parquet", "communities.parquet", "community_reports.parquet")
    _environment_lock = Lock()

    def __init__(
        self,
        settings: ProjectSettings,
        global_search_callable: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.global_search_callable = global_search_callable
        self.last_diagnostics: dict[str, Any] = self._empty_diagnostics()

    def readiness_report(self) -> dict[str, Any]:
        dependency_ready = importlib.util.find_spec("graphrag") is not None
        key_present = bool(self.settings.graphrag_llm_api_key)
        base_url_present = bool(self.settings.graphrag_llm_base_url)
        chat_model_present = bool(self.settings.graphrag_llm_chat_model)
        embedding_model_present = bool(self.settings.graphrag_llm_embedding_model)
        missing_index_files = [
            name for name in self.REQUIRED if not (self.settings.graphrag_output_dir / name).exists()
        ]
        index_ready = not missing_index_files
        config_path = self._config_path()
        config_ready = False
        config_reason = ""
        chat_model = self.settings.graphrag_llm_chat_model
        embedding_model = self.settings.graphrag_llm_embedding_model

        if config_path is None:
            config_reason = "missing_config"
        elif (
            dependency_ready
            and key_present
            and base_url_present
            and chat_model_present
            and embedding_model_present
        ):
            try:
                config = self._load_config()
                chat_config = config.get_language_model_config(config.global_search.chat_model_id)
                embedding_config = config.get_language_model_config("default_embedding_model")
                chat_model = chat_config.model
                embedding_model = embedding_config.model
                config_ready = True
            except Exception as exc:
                config_reason = f"invalid_config:{exc.__class__.__name__}"
        elif not dependency_ready:
            config_reason = "missing_dependencies"

        if not dependency_ready:
            reason = "missing_dependencies"
        elif not index_ready:
            reason = "index_missing"
        elif not key_present:
            reason = "missing_key"
        elif not base_url_present:
            reason = "missing_base_url"
        elif not chat_model_present:
            reason = "missing_chat_model"
        elif not embedding_model_present:
            reason = "missing_embedding_model"
        elif not config_ready:
            reason = config_reason or "invalid_config"
        else:
            reason = ""
        return {
            "ready": not reason,
            "reason": reason,
            "dependency_ready": dependency_ready,
            "config_ready": config_ready,
            "key_present": key_present,
            "base_url_present": base_url_present,
            "chat_model_present": chat_model_present,
            "embedding_model_present": embedding_model_present,
            "index_ready": index_ready,
            "missing_index_files": missing_index_files,
            "config_path": str(config_path) if config_path else None,
            "graph_output_path": str(self.settings.graphrag_output_dir),
            "chat_model": chat_model,
            "embedding_model": embedding_model,
            "api_base": self.settings.graphrag_llm_base_url,
        }

    def readiness(self) -> tuple[bool, str]:
        report = self.readiness_report()
        return bool(report["ready"]), str(report["reason"])

    def search(self, query: str) -> dict[str, Any]:
        report = self.readiness_report()
        self.last_diagnostics = {
            **self._empty_diagnostics(),
            "config_ready": report["config_ready"],
            "key_present": report["key_present"],
            "index_ready": report["index_ready"],
            "chat_model": report["chat_model"],
            "embedding_model": report["embedding_model"],
        }
        if not report["ready"]:
            reason = str(report["reason"])
            self.last_diagnostics["fallback_reason"] = reason
            raise RuntimeError(reason)
        import asyncio

        started = time.perf_counter()
        try:
            outcome = asyncio.run(self._search_async(query))
        except TimeoutError:
            self.last_diagnostics.update(
                {
                    "elapsed_ms": _elapsed_ms(started),
                    "fallback_reason": "timeout",
                }
            )
            raise
        except Exception as exc:
            text = str(exc)
            reason = (
                text.removeprefix("global_search_")
                if text.startswith("global_search_")
                else f"sdk_error:{exc.__class__.__name__}"
            )
            self.last_diagnostics.update(
                {
                    "elapsed_ms": _elapsed_ms(started),
                    "fallback_reason": reason,
                }
            )
            raise
        self.last_diagnostics.update(
            {
                "global_search_succeeded": True,
                "elapsed_ms": _elapsed_ms(started),
                "raw_result_type": outcome["raw_result_type"],
                "raw_result_length": outcome["raw_result_length"],
                "fallback_reason": None,
            }
        )
        outcome["diagnostics"] = dict(self.last_diagnostics)
        return outcome

    async def _search_async(self, query: str) -> dict[str, Any]:
        import asyncio
        import pandas as pd

        config = self._load_config()
        output = self.settings.graphrag_output_dir
        entities = pd.read_parquet(output / "entities.parquet")
        communities = pd.read_parquet(output / "communities.parquet")
        reports = pd.read_parquet(output / "community_reports.parquet")
        if self.global_search_callable is None:
            import graphrag.api as api

            global_search_callable = api.global_search
        else:
            global_search_callable = self.global_search_callable
        self.last_diagnostics["global_search_called"] = True
        try:
            payload = await asyncio.wait_for(
                global_search_callable(
                    config=config,
                    entities=entities,
                    communities=communities,
                    community_reports=reports,
                    community_level=2,
                    dynamic_community_selection=False,
                    response_type="Multiple Paragraphs",
                    query=query,
                    verbose=False,
                ),
                timeout=self.settings.graphrag_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("global_search:timeout") from exc
        try:
            text, context = normalize_official_response(
                payload,
                prefix="global_search",
            )
        except ValueError as exc:
            reason = str(exc).removeprefix("global_search:")
            raise ValueError(f"global_search_{reason}") from exc
        self.last_diagnostics.update(
            {
                "raw_result_type": type(payload[0]).__name__,
                "raw_result_length": len(text),
            }
        )
        source_summary = summarize_official_context(context)
        if not source_summary:
            raise ValueError("global_search_missing_source_summary")
        return {
            "content": text[: self.settings.graphrag_max_context_chars],
            "raw_preview": text[:500],
            "raw_result_type": type(payload[0]).__name__,
            "raw_result_length": len(text),
            "source_path": str(output / "community_reports.parquet"),
            "context_type": type(context).__name__,
            "source_summary": source_summary,
        }

    def _config_path(self) -> Path | None:
        for name in ("settings.yaml", "travelmind_runtime.yaml"):
            path = self.settings.graphrag_config_dir / name
            if path.exists():
                return path
        return None

    def _load_config(self, validation_only: bool = False):
        from graphrag.config.load_config import load_config

        config_path = self._config_path()
        if config_path is None:
            raise RuntimeError("missing_config")
        placeholder = "validation-placeholder" if validation_only else ""
        api_key = self.settings.graphrag_llm_api_key or placeholder
        env = {
            "TRAVELMIND_GRAPHRAG_LLM_API_KEY": api_key,
            "TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL": (
                self.settings.graphrag_llm_chat_model or placeholder
            ),
            "TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL": (
                self.settings.graphrag_llm_embedding_model or placeholder
            ),
            "TRAVELMIND_GRAPHRAG_LLM_BASE_URL": (
                self.settings.graphrag_llm_base_url
                or ("https://example.invalid/v1" if validation_only else "")
            ),
        }
        with self._temporary_environment(env):
            return load_config(self.settings.graphrag_config_dir, config_path)

    @classmethod
    @contextmanager
    def _temporary_environment(cls, values: dict[str, str]):
        with cls._environment_lock:
            previous = {key: os.environ.get(key) for key in values}
            try:
                os.environ.update(values)
                yield
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    @staticmethod
    def _empty_diagnostics() -> dict[str, Any]:
        return {
            "global_search_called": False,
            "global_search_succeeded": False,
            "elapsed_ms": None,
            "raw_result_type": None,
            "raw_result_length": 0,
            "fallback_reason": None,
        }


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _safe_length(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return 0
