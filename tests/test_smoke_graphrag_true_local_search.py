import unittest

from _helpers import test_settings
from scripts.smoke_graphrag_true_local_search import classify_case, run


class GraphRAGTrueLocalSearchSmokeTests(unittest.TestCase):
    def test_refuses_without_explicit_paid_authorization(self):
        with self.assertRaisesRegex(
            PermissionError,
            "allow_paid_local_search_required",
        ):
            run(
                ["阳朔和张家界哪个更适合看山水风景？"],
                settings=test_settings(),
            )

    def test_classification_requires_local_mode_coverage_and_sources(self):
        status = classify_case(
            {
                "route": "graphrag",
                "answer": "阳朔与张家界各有特点。",
                "fallback_reason": None,
                "trace": [
                    "agent:graphrag_agent:start",
                    "graphrag:official_local_called",
                    "graphrag:official_local_succeeded",
                    "generate:official_local_response",
                ],
                "retrieved": [
                    {
                        "metadata": {
                            "retrieval_mode": "graphrag_local_search",
                            "graphrag_relevance": True,
                            "source_summary": [
                                {
                                    "section": "sources",
                                    "row_count": 1,
                                    "titles_or_ids": ["unit-1"],
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "official_local_called": True,
                "official_local_succeeded": True,
            },
        )

        self.assertEqual(status, ("PASS", ""))


if __name__ == "__main__":
    unittest.main()
