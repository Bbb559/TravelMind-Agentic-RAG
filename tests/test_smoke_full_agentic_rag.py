import unittest

from scripts.smoke_full_agentic_rag import classify, extract, readiness, run


READY_ALL_FALSE = {
    "llm_ready": False,
    "embedding_ready": False,
    "naive_faiss_ready": False,
    "graphrag_ready": False,
    "graphrag_local_evidence_ready": False,
    "multimodal_vector_ready": False,
}


class FullSmokeHelperTests(unittest.TestCase):
    def sample_answer(self, route="naive_rag", trace=None, retrieval_mode="faiss", answer="ok"):
        agent = {
            "naive_rag": "NaiveTravelAgent",
            "graphrag": "GraphRAGAgent",
            "multimodal_rag": "MultimodalTravelAgent",
            "hybrid_rag": "HybridAggregator",
        }.get(route)
        return {
            "answer": answer,
            "route": route,
            "confidence": "high",
            "sources": [],
            "retrieved": [
                {
                    "metadata": {
                        "retrieval_mode": retrieval_mode,
                        "evidence_valid": True,
                    }
                }
            ],
            "fallback_reason": None,
            "trace": trace or ["system:route_source:llm", "agent:naive_travel_agent:start", "grade:llm:pass", "generate:llm"],
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
                        "retrieval_modes": ["markdown_keyword"],
                        "fallback_reason": None,
                    },
                }
                if route == "hybrid_rag"
                else None
            ),
        }

    def test_readiness_contains_boolean_values(self):
        self.assertTrue(all(isinstance(value, bool) for value in readiness().values()))

    def test_extract_trace_statuses(self):
        info = extract(self.sample_answer())
        self.assertEqual(info["route_source"], "system:route_source:llm")
        self.assertEqual(info["agent_name"], "naive_travel_agent")
        self.assertEqual(info["retrieval_mode"], "faiss")

    def test_strict_requires_llm_router_when_ready(self):
        ready = dict(READY_ALL_FALSE, llm_ready=True)
        answer = self.sample_answer(trace=["system:route_source:rule", "grade:llm:pass", "generate:llm"])
        status, reason = classify("大理", "naive_rag", answer, ready, strict=True)
        self.assertEqual(status, "FAIL")
        self.assertEqual(reason, "router_not_llm")

    def test_strict_requires_llm_generate_when_ready(self):
        ready = dict(READY_ALL_FALSE, llm_ready=True)
        answer = self.sample_answer(trace=["system:route_source:llm", "grade:llm:pass", "generate:template"])
        status, reason = classify("大理", "naive_rag", answer, ready, strict=True)
        self.assertEqual(status, "FAIL")
        self.assertEqual(reason, "generate_not_llm")

    def test_non_strict_allows_template_when_llm_not_ready(self):
        answer = self.sample_answer(trace=["system:route_source:rule", "grade:deterministic:pass", "generate:template"])
        status, _ = classify("大理", "naive_rag", answer, READY_ALL_FALSE, strict=False)
        self.assertEqual(status, "PASS")

    def test_strict_accepts_valid_csv_when_faiss_index_is_ready_but_irrelevant(self):
        ready = dict(READY_ALL_FALSE, llm_ready=True, naive_faiss_ready=True)
        answer = self.sample_answer(retrieval_mode="csv")
        status, reason = classify("大理", "naive_rag", answer, ready, strict=True)
        self.assertEqual((status, reason), ("PASS", ""))

    def test_ordinary_strict_rejects_paid_global_search(self):
        ready = dict(READY_ALL_FALSE, llm_ready=True, graphrag_ready=True)
        answer = self.sample_answer(
            route="graphrag",
            retrieval_mode="graphrag_global_search",
        )
        answer["retrieved"][0]["metadata"]["graphrag_relevance"] = True
        status, reason = classify("上海", "graphrag", answer, ready, strict=True)
        self.assertEqual(status, "FAIL")
        self.assertEqual(reason, "unexpected_paid_global_search")

    def test_strict_accepts_relevant_local_evidence_when_runtime_is_ready(self):
        ready = dict(READY_ALL_FALSE, llm_ready=True, graphrag_ready=True)
        answer = self.sample_answer(
            route="graphrag",
            retrieval_mode="graphrag_local_evidence",
            trace=[
                "system:route_source:llm",
                "grade:skipped:evidence_preview_only",
                "generate:skipped:evidence_preview_only",
            ],
            answer=(
                "GraphRAG Global Search 未开启。当前仅完成本地低成本证据检查，"
                "本地证据不足以生成正式结论。"
            ),
        )
        answer["fallback_reason"] = "global_search_disabled"
        answer["retrieved"][0]["metadata"]["graphrag_relevance"] = True
        status, reason = classify(
            "对比西安和南京",
            "graphrag",
            answer,
            ready,
            strict=True,
        )
        self.assertEqual((status, reason), ("PASS", ""))

    def test_strict_allows_relevant_local_graphrag_evidence_when_full_runtime_is_not_ready(self):
        answer = self.sample_answer(
            route="graphrag",
            retrieval_mode="graphrag_local_evidence",
            trace=[
                "system:route_source:llm",
                "grade:skipped:evidence_preview_only",
                "generate:skipped:evidence_preview_only",
            ],
            answer="本地证据模式仅供预览，不生成正式结论。",
        )
        answer["fallback_reason"] = "request_not_allowed"
        answer["retrieved"][0]["metadata"]["graphrag_relevance"] = True
        status, reason = classify("对比西安和南京", "graphrag", answer, READY_ALL_FALSE, strict=True)
        self.assertEqual((status, reason), ("PASS", ""))

    def test_strict_allows_low_relevance_local_preview_when_boundary_is_honest(self):
        answer = self.sample_answer(
            route="graphrag",
            retrieval_mode="graphrag_local_evidence",
            trace=[
                "system:route_source:llm",
                "grade:skipped:evidence_preview_only",
                "generate:skipped:evidence_preview_only",
            ],
            answer="本地证据模式仅供预览，本地证据不足以生成正式结论。",
        )
        answer["fallback_reason"] = "graphrag_low_relevance"
        answer["retrieved"][0]["metadata"]["graphrag_relevance"] = False
        status, reason = classify("对比西安和南京", "graphrag", answer, READY_ALL_FALSE, strict=True)
        self.assertEqual((status, reason), ("PASS", ""))

    def test_strict_rejects_local_evidence_formal_generation(self):
        answer = self.sample_answer(
            route="graphrag",
            retrieval_mode="graphrag_local_evidence",
            trace=[
                "system:route_source:llm",
                "grade:llm:pass",
                "generate:llm",
            ],
            answer="这是被包装成正式结论的本地证据答案。",
        )
        answer["fallback_reason"] = None
        status, reason = classify(
            "对比西安和南京",
            "graphrag",
            answer,
            READY_ALL_FALSE,
            strict=True,
        )
        self.assertEqual((status, reason), ("FAIL", "local_evidence_formal_answer"))

    def test_strict_allows_explicit_safe_graphrag_fallback(self):
        answer = self.sample_answer(
            route="graphrag",
            retrieval_mode="graphrag_wrapper",
            trace=[
                "system:route_source:llm",
                "grade:skipped:evidence_preview_only",
                "generate:skipped:evidence_preview_only",
            ],
            answer="当前 GraphRAG 索引未检索到相关证据，因此不生成具体旅游结论。",
        )
        answer["fallback_reason"] = "graphrag_low_relevance"
        status, reason = classify("对比西安和南京", "graphrag", answer, READY_ALL_FALSE, strict=True)
        self.assertEqual((status, reason), ("PASS", ""))

    def test_strict_accepts_valid_keyword_fallback_when_vector_is_irrelevant(self):
        ready = dict(READY_ALL_FALSE, llm_ready=True, multimodal_vector_ready=True)
        answer = self.sample_answer(route="multimodal_rag", retrieval_mode="markdown_keyword")
        status, reason = classify("香港", "multimodal_rag", answer, ready, strict=True)
        self.assertEqual((status, reason), ("PASS", ""))

    def test_hybrid_deep_fusion_claim_fails(self):
        answer = self.sample_answer(route="hybrid_rag", retrieval_mode="graphrag_global_search", answer="深度融合完成")
        status, reason = classify("台湾和西安", "hybrid_rag", answer, READY_ALL_FALSE, strict=False)
        self.assertEqual(status, "FAIL")
        self.assertEqual(reason, "hybrid_claims_deep_fusion")

    def test_hybrid_requires_aggregator_identity_trace(self):
        answer = self.sample_answer(
            route="hybrid_rag",
            retrieval_mode="markdown_keyword",
            trace=[
                "system:route_source:rule",
                "agent:graphrag_agent:start",
                "agent:multimodal_travel_agent:start",
                "generate:template",
            ],
            answer="当前仅完成多源候选聚合。",
        )

        status, reason = classify(
            "香港和成都哪个更适合亲子游？",
            "hybrid_rag",
            answer,
            READY_ALL_FALSE,
            strict=False,
        )

        self.assertEqual((status, reason), ("FAIL", "hybrid_aggregator_trace_missing"))

    def test_unsupported_claim_fails_even_with_route_fallback(self):
        answer = {
            "route": "fallback",
            "answer": "代码如下",
            "fallback_reason": "unsupported_query",
            "trace": [],
            "retrieved": [],
            "execution_status": {
                "agent": None,
                "retrieval_mode": "none",
                "evidence_status": "insufficient",
                "generation_mode": "none",
                "llm_stages": {
                    "router": "disabled",
                    "grade": "disabled",
                    "rewrite": "disabled",
                    "generate": "disabled",
                },
            },
        }
        status, reason = classify("帮我写一段 Python 快排", "fallback", answer, READY_ALL_FALSE, strict=True)
        self.assertEqual(status, "FAIL")
        self.assertEqual(reason, "unsupported_capability_claim")

    def test_route_mismatch_fails(self):
        status, reason = classify("大理", "naive_rag", self.sample_answer(route="graphrag"), READY_ALL_FALSE, strict=False)
        self.assertEqual(status, "FAIL")
        self.assertTrue(reason.startswith("route_mismatch"))

    def test_limited_run_returns_cases(self):
        payload = run(strict=False, limit=1)
        self.assertIn("readiness", payload)
        self.assertEqual(len(payload["cases"]), 1)

    def test_ordinary_strict_run_never_loads_remote_credentials(self):
        payload = run(strict=True, limit=0)

        self.assertFalse(payload["readiness"]["llm_ready"])
        self.assertFalse(payload["readiness"]["embedding_ready"])
        self.assertFalse(payload["readiness"]["graphrag_ready"])
        self.assertFalse(payload["readiness"]["graphrag_official_local_ready"])

    def test_smoke_output_does_not_expose_key_names_as_values(self):
        payload = run(strict=False, limit=1)
        self.assertNotIn("api_key", str(payload).lower())
