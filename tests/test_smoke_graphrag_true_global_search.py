import unittest

from _helpers import test_settings
from scripts.smoke_graphrag_true_global_search import (
    classify_case,
    classify_three_layers,
    run,
    summarize_cases,
)


class GraphRAGTrueGlobalSearchSmokeTests(unittest.TestCase):
    def test_three_layer_classification_accepts_safe_high_quality_answer(self):
        statuses = classify_three_layers(
            {
                "config_ready": True,
                "key_present": True,
                "index_ready": True,
            },
            {
                "global_search_called": True,
                "global_search_succeeded": True,
            },
            retrieval_mode="graphrag_global_search",
            global_search_available=True,
            graphrag_relevance=True,
            answer={
                "answer": "阳朔与张家界的山水风格各有特点。",
                "confidence": "medium",
                "sources": [{"source_type": "graphrag_index"}],
                "fallback_reason": None,
            },
        )

        self.assertEqual(statuses, ("PASS", "PASS", "PASS"))

    def test_three_layer_classification_accepts_safe_low_relevance_fallback(self):
        statuses = classify_three_layers(
            {
                "config_ready": True,
                "key_present": True,
                "index_ready": True,
            },
            {
                "global_search_called": True,
                "global_search_succeeded": True,
                "fallback_reason": "graphrag_low_relevance",
            },
            retrieval_mode="graphrag_local_evidence",
            global_search_available=False,
            graphrag_relevance=True,
            answer={
                "answer": "当前 GraphRAG 索引未检索到足够证据，因此不生成具体旅游结论。",
                "confidence": "low",
                "sources": [{"source_type": "graphrag_index"}],
                "fallback_reason": "graphrag_low_relevance",
            },
        )

        self.assertEqual(statuses, ("PASS", "FAIL", "PASS"))

    def test_successful_invocation_with_low_relevance_is_pass_fail(self):
        readiness = {
            "config_ready": True,
            "key_present": True,
            "index_ready": True,
        }
        diagnostics = {
            "global_search_called": True,
            "global_search_succeeded": True,
            "fallback_reason": "graphrag_low_relevance",
        }

        invocation_status, quality_status = classify_case(
            readiness,
            diagnostics,
            retrieval_mode="graphrag_local_evidence",
            global_search_available=False,
            graphrag_relevance=True,
        )

        self.assertEqual(invocation_status, "PASS")
        self.assertEqual(quality_status, "FAIL")

    def test_missing_key_is_warn_and_quality_skip(self):
        readiness = {
            "config_ready": True,
            "key_present": False,
            "index_ready": True,
        }

        invocation_status, quality_status = classify_case(
            readiness,
            {},
            retrieval_mode="graphrag_local_evidence",
            global_search_available=False,
            graphrag_relevance=True,
        )

        self.assertEqual((invocation_status, quality_status), ("WARN", "SKIP"))

    def test_successful_relevant_global_result_is_pass_pass(self):
        readiness = {
            "config_ready": True,
            "key_present": True,
            "index_ready": True,
        }
        diagnostics = {
            "global_search_called": True,
            "global_search_succeeded": True,
        }

        statuses = classify_case(
            readiness,
            diagnostics,
            retrieval_mode="graphrag_global_search",
            global_search_available=True,
            graphrag_relevance=True,
        )

        self.assertEqual(statuses, ("PASS", "PASS"))

    def test_ready_environment_with_failed_call_is_fail_skip(self):
        readiness = {
            "config_ready": True,
            "key_present": True,
            "index_ready": True,
        }
        diagnostics = {
            "global_search_called": True,
            "global_search_succeeded": False,
            "fallback_reason": "global_search_timeout",
        }

        statuses = classify_case(
            readiness,
            diagnostics,
            retrieval_mode="graphrag_local_evidence",
            global_search_available=False,
            graphrag_relevance=True,
        )

        self.assertEqual(statuses, ("FAIL", "SKIP"))

    def test_smoke_case_exposes_safe_invocation_and_quality_fields(self):
        payload = run(
            ["阳朔和张家界哪个更适合看山水风景？"],
            settings=test_settings(
                graphrag_llm_api_key="",
                graphrag_global_search_enabled=True,
            ),
            allow_paid_global_search=True,
        )
        case = payload["cases"][0]

        for key in [
            "config_ready",
            "key_present",
            "global_search_called",
            "global_search_succeeded",
            "invocation_status",
            "quality_status",
            "elapsed_ms",
            "raw_result_type",
            "raw_result_length",
        ]:
            self.assertIn(key, case)
        self.assertEqual(case["invocation_status"], "WARN")
        self.assertEqual(case["quality_status"], "SKIP")
        self.assertNotIn("api_key", str(payload).lower())

    def test_smoke_run_exposes_three_overall_statuses_and_audit_gate(self):
        payload = run(
            ["阳朔和张家界哪个更适合看山水风景？"],
            settings=test_settings(
                graphrag_llm_api_key="",
                graphrag_global_search_enabled=True,
            ),
            allow_paid_global_search=True,
        )
        case = payload["cases"][0]

        self.assertEqual(case["invocation_status"], "WARN")
        self.assertEqual(case["retrieval_quality_status"], "SKIP")
        self.assertEqual(case["answer_status"], "PASS")
        self.assertEqual(payload["overall_invocation_status"], "WARN")
        self.assertEqual(payload["overall_retrieval_quality_status"], "SKIP")
        self.assertEqual(payload["overall_answer_status"], "PASS")
        self.assertEqual(payload["overall_status"], "WARN")
        self.assertEqual(payload["recommended_action"], "ENVIRONMENT_REQUIRED")

    def test_smoke_refuses_to_run_without_explicit_paid_authorization(self):
        with self.assertRaisesRegex(
            PermissionError,
            "allow_paid_global_search_required",
        ):
            run(
                ["阳朔和张家界哪个更适合看山水风景？"],
                settings=test_settings(
                    graphrag_llm_api_key="",
                    graphrag_global_search_enabled=True,
                ),
            )

    def test_quality_gap_recommends_quality_review(self):
        summary = summarize_cases(
            [
                {
                    "invocation_status": "PASS",
                    "retrieval_quality_status": "FAIL",
                    "answer_status": "PASS",
                }
            ]
        )

        self.assertEqual(summary["overall_status"], "PASS_WITH_QUALITY_GAP")
        self.assertEqual(summary["recommended_action"], "QUALITY_REVIEW")

    def test_timeout_is_distinct_and_blocks_closure_even_with_safe_answer(self):
        diagnostics = {
            "global_search_called": True,
            "global_search_succeeded": False,
            "fallback_reason": "global_search_timeout",
        }
        statuses = classify_three_layers(
            {
                "config_ready": True,
                "key_present": True,
                "index_ready": True,
            },
            diagnostics,
            retrieval_mode="graphrag_local_evidence",
            global_search_available=False,
            graphrag_relevance=True,
            answer={
                "answer": "当前资料不足，无法可靠回答。",
                "confidence": "low",
                "sources": [{"source_type": "graphrag_index"}],
                "fallback_reason": "global_search_timeout",
            },
        )
        summary = summarize_cases(
            [
                {
                    "invocation_status": statuses[0],
                    "retrieval_quality_status": statuses[1],
                    "answer_status": statuses[2],
                }
            ]
        )

        self.assertEqual(diagnostics["fallback_reason"], "global_search_timeout")
        self.assertEqual(statuses, ("FAIL", "SKIP", "PASS"))
        self.assertEqual(summary["overall_status"], "FAIL")
        self.assertEqual(summary["recommended_action"], "FIX_REQUIRED")

    def test_answer_with_raw_error_or_secret_marker_fails_audit(self):
        statuses = classify_three_layers(
            {
                "config_ready": True,
                "key_present": True,
                "index_ready": True,
            },
            {
                "global_search_called": True,
                "global_search_succeeded": False,
                "fallback_reason": "global_search_error:RuntimeError",
            },
            retrieval_mode="graphrag_local_evidence",
            global_search_available=False,
            graphrag_relevance=True,
            answer={
                "answer": "global_search_error:RuntimeError api_key leaked",
                "confidence": "low",
                "sources": [],
                "fallback_reason": "global_search_error:RuntimeError",
            },
        )

        self.assertEqual(statuses, ("FAIL", "SKIP", "FAIL"))


if __name__ == "__main__":
    unittest.main()
