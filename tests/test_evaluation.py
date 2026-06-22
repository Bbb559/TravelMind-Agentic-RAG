import unittest

from _helpers import ROOT

from travelmind.evaluation import (
    manual_faithfulness_metrics,
    load_evaluation_bundle,
    percentile,
    route_classification_metrics,
    validate_evaluation_bundle,
    workflow_effect_metrics,
)
from scripts.evaluate_agentic_rag import (
    run_offline_evaluation,
    run_paid_local_evaluation,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_route_metrics_report_accuracy_and_macro_f1_across_all_routes(self):
        expected = [
            "naive_rag",
            "graphrag",
            "multimodal_rag",
            "hybrid_rag",
            "invalid_input",
            "fallback",
        ]
        predicted = [
            "naive_rag",
            "naive_rag",
            "multimodal_rag",
            "hybrid_rag",
            "invalid_input",
            "fallback",
        ]

        metrics = route_classification_metrics(expected, predicted)

        self.assertAlmostEqual(metrics["accuracy"], 5 / 6)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0 + 1 + 1 + 1 + 1) / 6)
        self.assertEqual(metrics["sample_count"], 6)

    def test_public_evaluation_bundle_has_frozen_counts_and_unique_queries(self):
        bundle = load_evaluation_bundle(ROOT / "evals" / "v1")

        summary = validate_evaluation_bundle(bundle)

        self.assertEqual(
            summary["route_counts"],
            {
                "naive_rag": 15,
                "graphrag": 10,
                "multimodal_rag": 10,
                "hybrid_rag": 10,
                "invalid_input": 5,
                "fallback": 10,
            },
        )
        self.assertEqual(summary["workflow_count"], 40)
        self.assertEqual(summary["paid_local_count"], 6)
        self.assertEqual(summary["manual_annotation_count"], 30)
        self.assertEqual(summary["duplicate_ids"], [])
        self.assertEqual(summary["duplicate_queries"], [])

    def test_workflow_metrics_separate_hit_safe_refusal_and_unsafe_generation(self):
        cases = [
            {
                "id": "a1",
                "answerable": True,
                "expected_evidence_entities": ["大理"],
                "expected_evidence_intents": ["itinerary"],
            },
            {
                "id": "a2",
                "answerable": True,
                "expected_evidence_entities": ["香港"],
                "expected_evidence_intents": ["multimodal_topic"],
            },
            {"id": "u1", "answerable": False},
            {"id": "u2", "answerable": False},
        ]
        payloads = [
            {
                "retrieved": [
                    {
                        "metadata": {
                            "evidence_valid": True,
                            "matched_entities": ["大理"],
                            "matched_intents": ["itinerary"],
                        }
                    }
                ],
                "fallback_reason": None,
                "execution_status": {"generation_mode": "template"},
            },
            {
                "retrieved": [],
                "fallback_reason": "no_relevant_evidence",
                "answer": "当前资料不足，无法可靠回答这个问题。",
                "execution_status": {"generation_mode": "template"},
            },
            {
                "retrieved": [],
                "fallback_reason": "no_relevant_evidence",
                "execution_status": {"generation_mode": "none"},
            },
            {
                "retrieved": [],
                "fallback_reason": None,
                "execution_status": {"generation_mode": "template"},
            },
        ]

        metrics = workflow_effect_metrics(cases, payloads)

        self.assertEqual(metrics["answerable_count"], 2)
        self.assertEqual(metrics["unanswerable_count"], 2)
        self.assertEqual(metrics["evidence_hit_at_3"], 0.5)
        self.assertEqual(metrics["safe_refusal_rate"], 0.5)
        self.assertEqual(metrics["unsafe_generation_rate"], 0.5)

    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)

    def test_manual_metrics_reject_pending_annotations_and_score_completed_claims(self):
        with self.assertRaisesRegex(
            ValueError,
            "manual_annotations_incomplete",
        ):
            manual_faithfulness_metrics(
                [{"case_id": "case-1", "review_status": "pending", "claims": []}]
            )

        metrics = manual_faithfulness_metrics(
            [
                {
                    "case_id": "case-1",
                    "review_status": "completed",
                    "claims": [
                        {"label": "supported"},
                        {"label": "not_verifiable"},
                    ],
                },
                {
                    "case_id": "case-2",
                    "review_status": "completed",
                    "claims": [
                        {"label": "supported"},
                        {"label": "unsupported"},
                    ],
                },
            ]
        )

        self.assertEqual(metrics["claim_support_rate"], 2 / 3)
        self.assertEqual(metrics["answer_hallucination_rate"], 0.5)

    def test_offline_runner_uses_public_api_and_never_executes_paid_search(self):
        result = run_offline_evaluation(
            route_cases=[
                {
                    "id": "route-1",
                    "query": "成都青城山适合怎么玩？",
                    "expected_route": "naive_rag",
                },
                {
                    "id": "route-2",
                    "query": "澳门有哪些代表性景点？",
                    "expected_route": "multimodal_rag",
                },
            ],
            workflow_cases=[
                {
                    "id": "workflow-1",
                    "query": "荔波小七孔有哪些必看景点？",
                    "expected_route": "naive_rag",
                    "answerable": True,
                    "expected_evidence_entities": ["荔波小七孔"],
                    "expected_evidence_intents": ["attractions"],
                },
                {
                    "id": "workflow-2",
                    "query": "？？！！",
                    "expected_route": "invalid_input",
                    "answerable": False,
                },
            ],
            repeats=1,
        )

        self.assertEqual(result["route_metrics"]["sample_count"], 2)
        self.assertEqual(result["workflow_metrics"]["sample_count"], 2)
        self.assertEqual(result["paid_search_calls"], 0)
        self.assertIn("overall", result["latency_ms"])
        self.assertIn("NaiveTravelAgent", result["latency_ms"]["by_agent"])
        self.assertEqual(
            result["workflow_cases"][1]["payload"]["route"],
            "invalid_input",
        )

    def test_paid_local_runner_checks_authorization_before_adapter_initialization(self):
        initialized = []

        with self.assertRaisesRegex(
            PermissionError,
            "allow_paid_local_search_required",
        ):
            run_paid_local_evaluation(
                [{"id": "paid-1", "query": "北京和西安有什么差异？"}],
                allow_paid_local_search=False,
                adapter_factory=lambda settings: initialized.append(settings),
            )

        self.assertEqual(initialized, [])

    def test_paid_local_runner_scores_official_local_without_global_search(self):
        class FakeLocalAdapter:
            def __init__(self, settings):
                self.last_diagnostics = {}

            def readiness(self):
                return True, ""

            def search(self, query):
                self.last_diagnostics = {
                    "official_local_called": True,
                    "official_local_succeeded": True,
                }
                return {
                    "content": (
                        "北京适合历史建筑与博物馆旅行，"
                        "西安适合古都遗址与城墙主题旅行。"
                    ),
                    "source_summary": [
                        {
                            "section": "reports",
                            "row_count": 2,
                            "titles_or_ids": ["beijing", "xian"],
                        }
                    ],
                    "diagnostics": dict(self.last_diagnostics),
                }

        result = run_paid_local_evaluation(
            [
                {
                    "id": "paid-1",
                    "query": "北京和西安的历史旅行有什么差异？",
                    "expected_route": "graphrag",
                }
            ],
            allow_paid_local_search=True,
            adapter_factory=FakeLocalAdapter,
        )

        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["global_search_calls"], 0)
        self.assertEqual(
            result["cases"][0]["retrieval_mode"],
            "graphrag_local_search",
        )


if __name__ == "__main__":
    unittest.main()
