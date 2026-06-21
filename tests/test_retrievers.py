import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from _helpers import fake_doc, test_settings
from travelmind.retrievers.graphrag_wrapper import GraphRAGSearchRetriever
from travelmind.retrievers.markdown_travel import MultimodalMarkdownRetriever, MultimodalVectorMarkdownRetriever
from travelmind.retrievers.naive_travel import NaiveAutoTravelRetriever, NaiveTravelRetriever


class RetrieverTests(unittest.TestCase):
    def test_csv_retriever_returns_real_results(self):
        results = NaiveTravelRetriever(test_settings()).retrieve("成都有什么美食？")
        self.assertTrue(results)
        self.assertEqual(results[0].source_type, "csv")

    def test_csv_retriever_marks_csv_mode(self):
        result = NaiveTravelRetriever(test_settings()).retrieve("大理到双廊怎么去？")[0]
        self.assertEqual(result.metadata["retrieval_mode"], "csv")

    def test_naive_auto_fallbacks_without_embedding_key(self):
        retriever = NaiveAutoTravelRetriever(test_settings(embedding_api_key=""))
        results = retriever.retrieve("大理到双廊怎么去？")
        self.assertEqual(retriever.last_mode, "faiss_fallback_csv")
        self.assertEqual(results[0].metadata["retrieval_mode"], "csv")

    def test_naive_auto_fake_faiss_success(self):
        retriever = NaiveAutoTravelRetriever(
            test_settings(embedding_api_key="fake"),
            faiss_loader=lambda query: [
                (
                    fake_doc(
                        "交通安排: 从大理前往双廊可乘班车或拼车。",
                        destination="大理双廊",
                    ),
                    0.2,
                )
            ],
        )
        result = retriever.retrieve("大理到双廊怎么去？")[0]
        self.assertEqual(retriever.last_mode, "faiss")
        self.assertEqual(result.metadata["retrieval_mode"], "faiss")

    def test_naive_auto_unsafe_index_fallbacks(self):
        settings = replace(test_settings(embedding_api_key="fake"), faiss_index_dir=Path("D:/unsafe/faiss_index"))
        retriever = NaiveAutoTravelRetriever(settings)
        retriever.retrieve("大理")
        self.assertEqual(retriever.last_reason, "unsafe_index_path")

    def test_multimodal_keyword_hongkong(self):
        results = MultimodalMarkdownRetriever(test_settings()).retrieve("香港迪士尼怎么玩？")
        self.assertTrue(results)
        self.assertEqual(results[0].source_type, "pdf_markdown")

    def test_multimodal_keyword_macau(self):
        results = MultimodalMarkdownRetriever(test_settings()).retrieve("澳门大三巴牌坊在哪里？")
        self.assertTrue(results)
        self.assertTrue(any("aomen" in (item.source_path or "") for item in results))

    def test_multimodal_keyword_taiwan(self):
        results = MultimodalMarkdownRetriever(test_settings()).retrieve("台北101有什么看点？")
        self.assertTrue(results)
        self.assertTrue(any("taiwan" in (item.source_path or "") for item in results))

    def test_multimodal_keyword_zero_match_returns_no_evidence(self):
        results = MultimodalMarkdownRetriever(test_settings()).retrieve(
            "新加坡圣淘沙怎么玩比较合适？"
        )

        self.assertEqual(results, [])

    def test_multimodal_vector_fallbacks_without_key(self):
        retriever = MultimodalVectorMarkdownRetriever(test_settings(embedding_api_key=""))
        results = retriever.retrieve("香港迪士尼怎么玩？")
        self.assertEqual(retriever.last_mode, "vector_fallback_markdown")
        self.assertEqual(results[0].metadata["retrieval_mode"], "markdown_keyword")

    def test_multimodal_vector_fake_success_keeps_source_metadata(self):
        retriever = MultimodalVectorMarkdownRetriever(
            test_settings(embedding_api_key="fake"),
            vector_loader=lambda query: [(fake_doc("香港 迪士尼", source_path="assets/result_markdown/xianggang/xianggang.md", region="xianggang", heading="迪士尼"), 0.3)],
        )
        result = retriever.retrieve("香港迪士尼怎么玩？")[0]
        self.assertEqual(retriever.last_mode, "markdown_vector")
        self.assertEqual(result.metadata["retrieval_mode"], "markdown_vector")
        self.assertIn("source_path", result.metadata)
        self.assertIn("region", result.metadata)
        self.assertIn("heading", result.metadata)

    def test_multimodal_vector_rejects_wrong_region_even_with_score(self):
        retriever = MultimodalVectorMarkdownRetriever(
            test_settings(embedding_api_key="fake"),
            vector_loader=lambda query: [
                (
                    fake_doc(
                        "台北101观景建议",
                        source_path="assets/result_markdown/taiwan/taiwan.md",
                        region="taiwan",
                        heading="台北101",
                    ),
                    0.1,
                )
            ],
        )

        results = retriever.retrieve("香港迪士尼怎么玩？")

        self.assertTrue(all(item.metadata.get("region") == "xianggang" for item in results))
        self.assertTrue(all(item.metadata.get("evidence_valid") is True for item in results))

    def test_multimodal_vector_without_region_metadata_falls_back_to_keyword(self):
        retriever = MultimodalVectorMarkdownRetriever(
            test_settings(embedding_api_key="fake"),
            vector_loader=lambda query: [
                (fake_doc("台北101观景建议", **{"Header 1": "台北·攻略"}), 0.1)
            ],
        )

        results = retriever.retrieve("香港迪士尼怎么玩？")

        self.assertEqual(retriever.last_mode, "vector_fallback_markdown")
        self.assertEqual(retriever.last_reason, "vector_empty")
        self.assertTrue(results)
        self.assertTrue(
            all(item.metadata.get("retrieval_mode") == "markdown_keyword" for item in results)
        )
        self.assertTrue(
            all(item.metadata.get("region") == "xianggang" for item in results)
        )

    def test_multimodal_vector_unsafe_path_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            unsafe = Path(tmp) / "index"
            unsafe.mkdir()
            (unsafe / "index.faiss").write_text("x")
            (unsafe / "index.pkl").write_text("x")
            settings = replace(test_settings(embedding_api_key="fake"), multimodal_markdown_dir=unsafe)
            retriever = MultimodalVectorMarkdownRetriever(settings)
            retriever.retrieve("香港")
            self.assertEqual(retriever.last_reason, "unsafe_index_path")

    def test_graphrag_fallback_without_key(self):
        retriever = GraphRAGSearchRetriever(test_settings(graphrag_llm_api_key=""))
        result = retriever.retrieve("对比西安和南京的人文景点")[0]
        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertFalse(result.metadata["global_search_available"])

    def test_graphrag_key_without_full_config_does_not_claim_global_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "travelmind_runtime.yaml").write_text(
                "global_search:\n  chat_model_id: default_chat_model\n",
                encoding="utf-8",
            )
            settings = replace(
                test_settings(
                    graphrag_llm_api_key="fake",
                    graphrag_llm_base_url="https://example.invalid/v1",
                    graphrag_llm_chat_model="chat-placeholder",
                    graphrag_llm_embedding_model="embedding-placeholder",
                    graphrag_global_search_enabled=True,
                ),
                graphrag_config_dir=config_dir,
            )
            result = GraphRAGSearchRetriever(
                settings,
                allow_global_search=True,
            ).retrieve(
                "对比西安和南京的人文景点"
            )[0]

        self.assertNotEqual(result.metadata["retrieval_mode"], "graphrag_global_search")
        self.assertFalse(result.metadata["global_search_available"])

    def test_graphrag_missing_config_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(
                test_settings(
                    graphrag_llm_api_key="fake",
                    graphrag_llm_base_url="https://example.invalid/v1",
                    graphrag_llm_chat_model="chat-placeholder",
                    graphrag_llm_embedding_model="embedding-placeholder",
                    graphrag_global_search_enabled=True,
                ),
                graphrag_config_dir=Path(tmp),
            )
            result = GraphRAGSearchRetriever(
                settings,
                allow_global_search=True,
            ).retrieve("北京周边")[0]
            self.assertEqual(result.metadata["fallback_reason"], "official_local_failed")
            self.assertEqual(result.metadata["official_local_error"], "missing_config")
            self.assertEqual(result.metadata["global_search_error"], "missing_config")

    def test_retriever_results_to_source(self):
        result = NaiveTravelRetriever(test_settings()).retrieve("成都美食")[0]
        source = result.to_source()
        self.assertEqual(source.source_type, result.source_type)

    def test_retriever_result_json_safe_dict(self):
        data = NaiveTravelRetriever(test_settings()).retrieve("成都美食")[0].to_dict()
        self.assertIn("metadata", data)

    def test_multimodal_vector_raw_score_is_recorded(self):
        retriever = MultimodalVectorMarkdownRetriever(
            test_settings(embedding_api_key="fake"),
            vector_loader=lambda query: [(fake_doc("台北101", source_path="assets/result_markdown/taiwan/taiwan.md"), 1.5)],
        )
        result = retriever.retrieve("台北101有什么看点？")[0]
        self.assertEqual(result.metadata["raw_score"], 1.5)

    def test_naive_auto_fake_empty_fallbacks(self):
        retriever = NaiveAutoTravelRetriever(test_settings(embedding_api_key="fake"), faiss_loader=lambda query: [])
        retriever.retrieve("大理")
        self.assertEqual(retriever.last_reason, "faiss_no_relevant_evidence")
