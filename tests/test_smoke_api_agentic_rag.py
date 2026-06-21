import unittest
import os
from unittest.mock import patch

from scripts.smoke_api_agentic_rag import classify_case, extract_status
from scripts.serve_http_smoke import prepare_safe_environment


class ApiSmokeHelperTests(unittest.TestCase):
    def payload(self, route="naive_rag", retrieval_mode="faiss", trace=None, answer="中文答案"):
        agent = {
            "naive_rag": "NaiveTravelAgent",
            "graphrag": "GraphRAGAgent",
            "multimodal_rag": "MultimodalTravelAgent",
            "hybrid_rag": "HybridAggregator",
        }.get(route)
        return {
            "answer": answer,
            "route": route,
            "runtime_summary": {"llm_enabled": True, "key_present": True},
            "global_search_status": {
                "requested": False,
                "service_enabled": False,
                "effective_allowed": False,
                "executed": False,
                "succeeded": False,
            },
            "retrieved": [
                {
                    "metadata": {
                        "retrieval_mode": retrieval_mode,
                        "evidence_valid": True,
                    }
                }
            ],
            "trace": trace
            or [
                "system:route_source:llm",
                "naive:retriever_mode:faiss",
                "grade:llm:pass",
                "generate:llm",
            ],
            "execution_status": {
                "agent": agent,
                "retrieval_mode": retrieval_mode,
                "evidence_status": "sufficient",
                "generation_mode": "llm",
                "llm_stages": {
                    "router": "executed",
                    "grade": "executed",
                    "rewrite": "not_needed",
                    "generate": "executed",
                },
            },
            "hybrid_branch_status": (
                {
                    "graphrag": {
                        "execution": "completed",
                        "evidence_valid": False,
                        "retrieval_modes": ["graphrag_local_evidence"],
                        "fallback_reason": "official_local_failed",
                    },
                    "multimodal": {
                        "execution": "completed",
                        "evidence_valid": True,
                        "retrieval_modes": [retrieval_mode],
                        "fallback_reason": None,
                    },
                }
                if route == "hybrid_rag"
                else None
            ),
        }

    def test_extract_status_reads_trace_and_retrieval_mode(self):
        status = extract_status(self.payload())
        self.assertTrue(status["route_source_llm"])
        self.assertTrue(status["grade_llm"])
        self.assertTrue(status["generate_llm"])
        self.assertTrue(status["naive_faiss"])

    def test_safe_http_server_disables_official_graphrag_config(self):
        with patch.dict(
            os.environ,
            {
                "TRAVELMIND_LLM_API_KEY": "must-not-be-used",
                "TRAVELMIND_EMBEDDING_API_KEY": "must-not-be-used",
            },
            clear=False,
        ):
            prepare_safe_environment()
            self.assertEqual(
                os.environ["TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED"],
                "false",
            )
            self.assertEqual(os.environ["TRAVELMIND_LLM_API_KEY"], "")
            self.assertEqual(os.environ["TRAVELMIND_EMBEDDING_API_KEY"], "")
            self.assertIn(
                "smoke_disabled_graphrag_config",
                os.environ["TRAVELMIND_GRAPHRAG_CONFIG_DIR"],
            )

    def test_classify_passes_real_naive_chain(self):
        status, reason = classify_case(self.payload(), "naive_rag", "naive_valid")
        self.assertEqual(status, "PASS")
        self.assertEqual(reason, "")

    def test_classify_accepts_honestly_disabled_llm_runtime(self):
        payload = self.payload()
        payload["runtime_summary"]["llm_enabled"] = False
        status, reason = classify_case(payload, "naive_rag", "naive_valid")
        self.assertEqual((status, reason), ("PASS", ""))

    def test_classify_fails_when_runtime_summary_has_extra_key(self):
        payload = self.payload()
        payload["runtime_summary"]["api_key_name"] = True
        status, reason = classify_case(payload, "naive_rag", "naive_valid")
        self.assertEqual(status, "FAIL")
        self.assertEqual(reason, "runtime_summary_not_safe")

    def test_classify_fails_when_generate_not_llm(self):
        payload = self.payload(trace=["system:route_source:llm", "grade:llm:pass", "generate:template"])
        status, reason = classify_case(payload, "naive_rag", "naive_valid")
        self.assertEqual(status, "FAIL")
        self.assertEqual(reason, "generate_not_llm")

    def test_ordinary_http_smoke_rejects_paid_graphrag_mode(self):
        payload = self.payload(
            route="graphrag",
            retrieval_mode="graphrag_global_search",
            trace=["system:route_source:llm", "graphrag:retriever_mode:global_search", "grade:llm:weak", "generate:llm"],
            answer="Daqikong Scenic Area and Tian Sheng Qiao This community comprises two natural entities "
            "with mostly English raw retrieved text and little Chinese.",
        )
        status, reason = classify_case(payload, "graphrag", "graphrag_global_search")
        self.assertEqual(status, "FAIL")
        self.assertEqual(reason, "unexpected_paid_global_search")

    def test_classify_accepts_relevant_local_graphrag_evidence(self):
        payload = self.payload(
            route="graphrag",
            retrieval_mode="graphrag_local_evidence",
            trace=[
                "system:route_source:llm",
                "grade:skipped:evidence_preview_only",
                "generate:skipped:evidence_preview_only",
            ],
            answer="本地证据模式仅供预览，本地证据不足以生成正式结论。",
        )
        payload["fallback_reason"] = "global_search_disabled"
        payload["retrieved"][0]["metadata"]["graphrag_relevance"] = True
        status, reason = classify_case(payload, "graphrag", "graphrag_relevant_or_safe")
        self.assertEqual((status, reason), ("PASS", ""))

    def test_classify_accepts_low_relevance_local_preview(self):
        payload = self.payload(
            route="graphrag",
            retrieval_mode="graphrag_local_evidence",
            trace=[
                "system:route_source:llm",
                "grade:skipped:evidence_preview_only",
                "generate:skipped:evidence_preview_only",
            ],
            answer="本地证据模式仅供预览，本地证据不足以生成正式结论。",
        )
        payload["fallback_reason"] = "graphrag_low_relevance"
        payload["retrieved"][0]["metadata"]["graphrag_relevance"] = False
        status, reason = classify_case(payload, "graphrag", "graphrag_relevant_or_safe")
        self.assertEqual((status, reason), ("PASS", ""))

    def test_classify_rejects_local_evidence_formal_generation(self):
        payload = self.payload(
            route="graphrag",
            retrieval_mode="graphrag_local_evidence",
            trace=[
                "system:route_source:llm",
                "grade:llm:pass",
                "generate:llm",
            ],
            answer="这是被包装成正式结论的本地证据答案。",
        )
        payload["retrieved"][0]["metadata"]["graphrag_relevance"] = True
        status, reason = classify_case(
            payload,
            "graphrag",
            "graphrag_relevant_or_safe",
        )
        self.assertEqual((status, reason), ("FAIL", "local_evidence_formal_answer"))

    def test_classify_accepts_explicit_safe_graphrag_fallback(self):
        payload = self.payload(
            route="graphrag",
            retrieval_mode="graphrag_local_evidence",
            trace=[
                "system:route_source:llm",
                "grade:skipped:evidence_preview_only",
                "generate:skipped:evidence_preview_only",
            ],
            answer="当前未检索到足够证据，因此不生成具体旅游结论。",
        )
        payload["fallback_reason"] = "incomplete_config"
        payload["retrieved"][0]["metadata"]["graphrag_relevance"] = True
        status, reason = classify_case(payload, "graphrag", "graphrag_relevant_or_safe")
        self.assertEqual((status, reason), ("PASS", ""))

    def test_classify_rejects_hybrid_without_aggregator_trace(self):
        payload = self.payload(
            route="hybrid_rag",
            retrieval_mode="markdown_vector",
            trace=[
                "system:route_source:llm",
                "agent:graphrag_agent:start",
                "agent:multimodal_travel_agent:start",
                "grade:llm:pass",
                "generate:llm",
            ],
            answer="当前仅完成多源候选聚合。",
        )

        status, reason = classify_case(payload, "hybrid_rag", None)

        self.assertEqual(
            (status, reason),
            ("FAIL", "hybrid_aggregator_trace_missing"),
        )


if __name__ == "__main__":
    unittest.main()
