"""GraphRAG 2.7 官方 local_search API 隔离适配器。"""

from __future__ import annotations

import importlib.util
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


class GraphRAGOfficialLocalSearchAdapter:
    """集中处理官方 Local Search 的配置、资产、超时和安全结果。"""

    REQUIRED = (
        "entities.parquet",
        "communities.parquet",
        "community_reports.parquet",
        "text_units.parquet",
        "relationships.parquet",
    )
    _environment_lock = Lock()

    def __init__(
        self,
        settings: ProjectSettings,
        local_search_callable: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.local_search_callable = local_search_callable
        self.last_diagnostics: dict[str, Any] = self._empty_diagnostics()

    def readiness_report(self) -> dict[str, Any]:
        dependency_ready = importlib.util.find_spec("graphrag") is not None
        key_present = bool(self.settings.graphrag_llm_api_key)
        base_url_present = bool(self.settings.graphrag_llm_base_url)
        chat_model_present = bool(self.settings.graphrag_llm_chat_model)
        embedding_model_present = bool(self.settings.graphrag_llm_embedding_model)
        missing_index_files = [
            name
            for name in self.REQUIRED
            if not (self.settings.graphrag_output_dir / name).exists()
        ]
        vector_store_path = self.settings.graphrag_output_dir / "lancedb"
        vector_store_ready = vector_store_path.is_dir()
        vector_table_ready = (
            vector_store_path / "default-entity-description.lance"
        ).is_dir()
        config_path = self._config_path()
        config_ready = False
        config_reason = ""

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
                vector_ids = list(config.vector_store)
                vector_store_ready = bool(
                    vector_ids
                    and Path(config.vector_store[vector_ids[0]].db_uri).resolve()
                    == vector_store_path.resolve()
                    and vector_store_ready
                )
                config.get_language_model_config(config.local_search.chat_model_id)
                config.get_language_model_config(config.local_search.embedding_model_id)
                config_ready = True
            except Exception as exc:
                config_reason = f"invalid_config:{exc.__class__.__name__}"
        elif not dependency_ready:
            config_reason = "missing_dependencies"

        if not dependency_ready:
            reason = "missing_dependencies"
        elif missing_index_files:
            reason = "index_missing"
        elif not vector_store_ready:
            reason = "vector_store_missing"
        elif not vector_table_ready:
            reason = "vector_table_missing"
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
            "index_ready": not missing_index_files,
            "missing_index_files": missing_index_files,
            "vector_store_ready": vector_store_ready,
            "vector_table_ready": vector_table_ready,
            "config_path": str(config_path) if config_path else None,
            "graph_output_path": str(self.settings.graphrag_output_dir),
            "vector_store_path": str(vector_store_path),
            "chat_model": self.settings.graphrag_llm_chat_model,
            "embedding_model": self.settings.graphrag_llm_embedding_model,
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
            "vector_store_ready": report["vector_store_ready"],
            "vector_table_ready": report["vector_table_ready"],
            "chat_model": report["chat_model"],
            "embedding_model": report["embedding_model"],
        }
        if not report["ready"]:
            reason = str(report["reason"])
            self.last_diagnostics["official_local_error"] = reason
            raise RuntimeError(reason)

        import asyncio

        started = time.perf_counter()
        try:
            outcome = asyncio.run(self._search_async(query))
        except TimeoutError:
            self.last_diagnostics.update(
                {
                    "elapsed_ms": _elapsed_ms(started),
                    "official_local_error": "timeout",
                }
            )
            raise
        except Exception as exc:
            reason = _safe_error_code(exc, "official_local")
            self.last_diagnostics.update(
                {
                    "elapsed_ms": _elapsed_ms(started),
                    "official_local_error": reason,
                }
            )
            raise

        self.last_diagnostics.update(
            {
                "official_local_succeeded": True,
                "elapsed_ms": _elapsed_ms(started),
                "raw_result_type": outcome["raw_result_type"],
                "raw_result_length": outcome["raw_result_length"],
                "source_summary_count": len(outcome["source_summary"]),
                "official_local_error": None,
            }
        )
        outcome["diagnostics"] = dict(self.last_diagnostics)
        return outcome

    async def _search_async(self, query: str) -> dict[str, Any]:
        import asyncio
        import pandas as pd

        config = self._load_config()
        output = self.settings.graphrag_output_dir
        covariates_path = output / "covariates.parquet"
        callable_ = self.local_search_callable
        if callable_ is None:
            import graphrag.api as api

            callable_ = api.local_search

        self.last_diagnostics["official_local_called"] = True
        try:
            payload = await asyncio.wait_for(
                callable_(
                    config=config,
                    entities=pd.read_parquet(output / "entities.parquet"),
                    communities=pd.read_parquet(output / "communities.parquet"),
                    community_reports=pd.read_parquet(output / "community_reports.parquet"),
                    text_units=pd.read_parquet(output / "text_units.parquet"),
                    relationships=pd.read_parquet(output / "relationships.parquet"),
                    covariates=(
                        pd.read_parquet(covariates_path)
                        if covariates_path.exists()
                        else None
                    ),
                    community_level=2,
                    response_type="Multiple Paragraphs",
                    query=query,
                    verbose=False,
                ),
                timeout=self.settings.graphrag_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("official_local:timeout") from exc

        text, context = normalize_official_response(
            payload,
            prefix="official_local",
        )
        source_summary = summarize_official_context(context)
        if not source_summary:
            raise ValueError("official_local:missing_source_summary")
        return {
            "content": text[: self.settings.graphrag_max_context_chars],
            "raw_preview": text[:500],
            "raw_result_type": type(payload[0]).__name__,
            "raw_result_length": len(text),
            "source_path": str(output / "lancedb"),
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
        values = self._config_environment(validation_only=validation_only)
        with self._temporary_environment(values):
            initial = load_config(self.settings.graphrag_config_dir, config_path)
            vector_ids = list(initial.vector_store)
            if not vector_ids:
                raise RuntimeError("vector_store_missing")
            overrides = {
                "output.base_dir": str(self.settings.graphrag_output_dir),
                f"vector_store.{vector_ids[0]}.db_uri": str(
                    self.settings.graphrag_output_dir / "lancedb"
                ),
            }
            return load_config(
                self.settings.graphrag_config_dir,
                config_path,
                cli_overrides=overrides,
            )

    def _config_environment(self, *, validation_only: bool) -> dict[str, str]:
        placeholder = "validation-placeholder" if validation_only else ""
        return {
            "TRAVELMIND_GRAPHRAG_LLM_API_KEY": (
                self.settings.graphrag_llm_api_key or placeholder
            ),
            "TRAVELMIND_GRAPHRAG_LLM_BASE_URL": (
                self.settings.graphrag_llm_base_url
                or ("https://example.invalid/v1" if validation_only else "")
            ),
            "TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL": (
                self.settings.graphrag_llm_chat_model or placeholder
            ),
            "TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL": (
                self.settings.graphrag_llm_embedding_model or placeholder
            ),
        }

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
            "official_local_called": False,
            "official_local_succeeded": False,
            "official_local_error": None,
            "elapsed_ms": None,
            "raw_result_type": None,
            "raw_result_length": 0,
            "source_summary_count": 0,
        }


def _safe_error_code(exc: Exception, prefix: str) -> str:
    text = str(exc)
    marker = f"{prefix}:"
    if text.startswith(marker):
        return text[len(marker) :]
    if text in {
        "missing_config",
        "missing_key",
        "missing_base_url",
        "missing_chat_model",
        "missing_embedding_model",
        "index_missing",
        "vector_store_missing",
        "vector_table_missing",
    }:
        return text
    return f"sdk_error:{exc.__class__.__name__}"


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))
