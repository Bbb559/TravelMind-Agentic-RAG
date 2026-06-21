import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travelmind.graphs import AgenticRAGWorkflow


class CoreWorkflowTests(unittest.TestCase):
    def test_naive_route_returns_answer_shape(self):
        answer = AgenticRAGWorkflow().run("大理到双廊怎么去？")
        self.assertEqual(answer.route, "naive_rag")
        self.assertTrue(answer.retrieved)
        self.assertIn("system:route:naive_rag", answer.trace)
        self.assertIn("agent:naive_travel_agent:generate_response", answer.trace)

    def test_multimodal_route_uses_markdown_agent(self):
        answer = AgenticRAGWorkflow().run("香港迪士尼怎么玩？")
        self.assertEqual(answer.route, "multimodal_rag")
        self.assertTrue(any("multimodal_travel_agent" in step for step in answer.trace))

    def test_unsupported_query_fallbacks(self):
        answer = AgenticRAGWorkflow().run("帮我写一段 Python 快排")
        self.assertEqual(answer.route, "fallback")
        self.assertEqual(answer.fallback_reason, "unsupported_query")


if __name__ == "__main__":
    unittest.main()
