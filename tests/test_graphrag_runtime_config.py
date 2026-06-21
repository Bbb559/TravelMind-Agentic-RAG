import os
import unittest
from unittest.mock import patch

from _helpers import test_settings
from travelmind.graphrag import GraphRAGGlobalSearchAdapter


class GraphRAGRuntimeConfigTests(unittest.TestCase):
    def test_runtime_config_is_secret_free_and_loadable_by_graphrag_27(self):
        settings = test_settings(
            graphrag_llm_api_key="test-placeholder",
            graphrag_llm_base_url="https://example.invalid/v1",
            graphrag_llm_chat_model="chat-placeholder",
            graphrag_llm_embedding_model="embedding-placeholder",
        )
        config_path = settings.graphrag_config_dir / "travelmind_runtime.yaml"
        text = config_path.read_text(encoding="utf-8")

        self.assertNotIn("sk-", text)
        self.assertNotIn("test-placeholder", text)

        env = {
            "TRAVELMIND_GRAPHRAG_LLM_API_KEY": "test-placeholder",
            "TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL": "chat-placeholder",
            "TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL": "embedding-placeholder",
            "TRAVELMIND_GRAPHRAG_LLM_BASE_URL": "https://example.invalid/v1",
        }
        with patch.dict(os.environ, env, clear=False):
            report = GraphRAGGlobalSearchAdapter(settings).readiness_report()

        self.assertTrue(report["config_ready"])
        self.assertTrue(report["index_ready"])
        self.assertTrue(report["key_present"])
        self.assertEqual(report["chat_model"], "chat-placeholder")
        self.assertEqual(report["embedding_model"], "embedding-placeholder")


if __name__ == "__main__":
    unittest.main()
