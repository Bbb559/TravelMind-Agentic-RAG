import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _helpers import ROOT, clean_env, restore_env, test_settings
from travelmind.config import ProjectSettings, get_settings


class SettingsAssetsTests(unittest.TestCase):
    def setUp(self):
        self.settings = test_settings()

    def test_hybrid_branch_timeout_defaults_to_twenty_seconds(self):
        self.assertEqual(self.settings.hybrid_branch_timeout_seconds, 20)

    def test_settings_loads_bom_prefixed_env_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "TRAVELMIND_LLM_API_KEY=bom-safe-key\n"
                "TRAVELMIND_LLM_MODEL=bom-safe-model\n",
                encoding="utf-8-sig",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = ProjectSettings.load(env_path=env_path)

        self.assertEqual(settings.llm_api_key, "bom-safe-key")
        self.assertEqual(settings.llm_model, "bom-safe-model")

    def test_noncanonical_llm_aliases_are_not_loaded(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "noncanonical-key",
                "QWEN_API_KEY": "noncanonical-embedding-key",
                "TRAVELMIND_CHAT_MODEL": "noncanonical-model",
                "TRAVELMIND_LLM_API_KEY": "",
                "TRAVELMIND_EMBEDDING_API_KEY": "",
                "TRAVELMIND_LLM_MODEL": "",
            },
            clear=True,
        ):
            settings = ProjectSettings.load(env_path=Path("missing.env"))

        self.assertEqual(settings.llm_api_key, "")
        self.assertEqual(settings.embedding_api_key, "")
        self.assertEqual(settings.llm_model, "deepseek-chat")

    def test_assets_dir_is_project_assets(self):
        self.assertEqual(self.settings.assets_dir, ROOT / "assets")

    def test_csv_path_uses_assets_not_root_datasets(self):
        self.assertEqual(self.settings.travel_csv_path, ROOT / "assets" / "travel_guide.csv")
        self.assertNotEqual(self.settings.travel_csv_path, ROOT / "datasets" / "travel_guide.csv")

    def test_faiss_path_uses_assets_not_root(self):
        self.assertEqual(self.settings.faiss_index_dir, ROOT / "assets" / "faiss_index")
        self.assertNotEqual(self.settings.faiss_index_dir, ROOT / "faiss_index")

    def test_graphrag_paths_use_assets(self):
        self.assertEqual(self.settings.graphrag_output_dir, ROOT / "assets" / "graphrag_output")
        self.assertEqual(self.settings.graphrag_config_dir, ROOT / "assets" / "graphrag_runtime")

    def test_multimodal_markdown_path_uses_assets(self):
        self.assertEqual(self.settings.multimodal_markdown_dir, ROOT / "assets" / "result_markdown")

    def test_main_csv_exists(self):
        self.assertTrue(self.settings.travel_csv_path.exists())

    def test_main_faiss_index_exists(self):
        self.assertTrue((self.settings.faiss_index_dir / "index.faiss").exists())
        self.assertTrue((self.settings.faiss_index_dir / "index.pkl").exists())

    def test_multimodal_vector_index_exists(self):
        self.assertTrue((self.settings.multimodal_markdown_dir / "index.faiss").exists())
        self.assertTrue((self.settings.multimodal_markdown_dir / "index.pkl").exists())

    def test_graphrag_core_assets_exist(self):
        for name in ["entities.parquet", "relationships.parquet", "community_reports.parquet", "text_units.parquet"]:
            self.assertTrue((self.settings.graphrag_output_dir / name).exists(), name)

    def test_graphrag_runtime_config_exists(self):
        self.assertTrue((self.settings.graphrag_config_dir / "travelmind_runtime.yaml").exists())

    def test_gang_ao_pdf_assets_exist(self):
        pdfs = list((self.settings.assets_dir / "gang_ao_pdf").glob("*.pdf"))
        self.assertGreaterEqual(len(pdfs), 3)

    def test_core_python_files_are_not_empty(self):
        core = [
            ROOT / "src" / "travelmind" / "agents" / "system.py",
            ROOT / "src" / "travelmind" / "retrievers" / "naive_travel.py",
            ROOT / "src" / "travelmind" / "runtime" / "rag_helpers.py",
            ROOT / "src" / "travelmind" / "api.py",
            ROOT / "src" / "travelmind" / "cli.py",
        ]
        for path in core:
            self.assertGreater(path.stat().st_size, 500, str(path))

    def test_default_runtime_profile_keeps_llm_disabled(self):
        old = clean_env()
        try:
            get_settings.cache_clear()
            settings = get_settings()
            self.assertFalse(settings.llm_enabled)
            self.assertFalse(settings.llm_generate_enabled)
            self.assertFalse(settings.llm_grade_enabled)
            self.assertFalse(settings.llm_rewrite_enabled)
            self.assertFalse(settings.system_agent_llm_router_enabled)
            self.assertFalse(settings.naive_agent_llm_loop_enabled)
            self.assertFalse(settings.graphrag_global_search_enabled)
            self.assertFalse(settings.run_log_enabled)
        finally:
            restore_env(old)
            get_settings.cache_clear()

    def test_full_agentic_demo_profile_enables_llm_switches(self):
        old = clean_env()
        try:
            os.environ["TRAVELMIND_RUNTIME_PROFILE"] = "full_agentic_demo"
            get_settings.cache_clear()
            settings = get_settings()
            self.assertTrue(settings.llm_enabled)
            self.assertTrue(settings.llm_generate_enabled)
            self.assertTrue(settings.llm_grade_enabled)
            self.assertTrue(settings.llm_rewrite_enabled)
            self.assertTrue(settings.system_agent_llm_router_enabled)
            self.assertTrue(settings.naive_agent_llm_loop_enabled)
            self.assertFalse(settings.graphrag_global_search_enabled)
        finally:
            restore_env(old)
            get_settings.cache_clear()

    def test_explicit_false_overrides_full_agentic_demo_profile(self):
        old = clean_env()
        try:
            os.environ["TRAVELMIND_RUNTIME_PROFILE"] = "full_agentic_demo"
            os.environ["TRAVELMIND_LLM_GRADE_ENABLED"] = "false"
            get_settings.cache_clear()
            settings = get_settings()
            self.assertTrue(settings.llm_enabled)
            self.assertFalse(settings.llm_grade_enabled)
        finally:
            restore_env(old)
            get_settings.cache_clear()

    def test_global_search_requires_independent_explicit_service_switch(self):
        old = clean_env()
        try:
            os.environ["TRAVELMIND_RUNTIME_PROFILE"] = "full_agentic_demo"
            os.environ["TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED"] = "true"
            get_settings.cache_clear()
            settings = get_settings()
            self.assertTrue(settings.llm_enabled)
            self.assertTrue(settings.graphrag_global_search_enabled)
        finally:
            restore_env(old)
            get_settings.cache_clear()

    def test_graphrag_uses_only_new_independent_llm_variable_names(self):
        with patch.dict(
            os.environ,
            {
                "TRAVELMIND_GRAPHRAG_LLM_API_KEY": "new-key",
                "TRAVELMIND_GRAPHRAG_LLM_BASE_URL": "https://example.invalid/v1",
                "TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL": "new-chat",
                "TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL": "new-embedding",
            },
        ):
            get_settings.cache_clear()
            try:
                settings = get_settings()
            finally:
                get_settings.cache_clear()

        self.assertEqual(settings.graphrag_llm_api_key, "new-key")
        self.assertEqual(
            settings.graphrag_llm_base_url,
            "https://example.invalid/v1",
        )
        self.assertEqual(settings.graphrag_llm_chat_model, "new-chat")
        self.assertEqual(
            settings.graphrag_llm_embedding_model,
            "new-embedding",
        )
