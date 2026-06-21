import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.smoke_full_agentic_rag import classify, readiness
from scripts.debug_graphrag_retrieval import diagnose_query


class SmokeHelperTests(unittest.TestCase):
    def test_readiness_has_no_secret_values(self):
        payload = readiness()
        self.assertTrue(all(isinstance(value, bool) for value in payload.values()))

    def test_unsupported_claim_fails(self):
        answer = {
            "route": "fallback",
            "answer": "代码如下",
            "fallback_reason": None,
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
        status, reason = classify("帮我写一段 Python 快排", "fallback", answer, {"llm_ready": False}, strict=True)
        self.assertEqual(status, "FAIL")
        self.assertEqual(reason, "unsupported_capability_claim")

    def test_debug_defaults_to_local_only_mode(self):
        payload = diagnose_query(
            "对比西安和南京的人文景点",
            allow_paid_global_search=False,
        )

        self.assertFalse(payload["global_search_status"]["requested"])
        self.assertFalse(payload["global_search_status"]["effective_allowed"])
        self.assertEqual(
            set(payload["global_search_status"]),
            {
                "requested",
                "service_enabled",
                "effective_allowed",
                "executed",
                "succeeded",
            },
        )
        self.assertFalse(payload["global_search_status"]["executed"])
        self.assertFalse(payload["global_search_status"]["succeeded"])
        self.assertFalse(payload["global_search_called"])
        self.assertIn(
            payload["normalized_result"]["metadata"]["retrieval_mode"],
            {"graphrag_local_evidence", "graphrag_wrapper"},
        )


if __name__ == "__main__":
    unittest.main()
