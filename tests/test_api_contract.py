import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from travelmind.api import app


class ApiContractTests(unittest.TestCase):
    def test_route_and_workflow(self):
        client = TestClient(app)
        route = client.post("/api/route", json={"query": "香港迪士尼怎么玩？"})
        self.assertEqual(route.status_code, 200)
        self.assertEqual(route.json()["route"], "multimodal_rag")
        workflow = client.post("/api/workflow", json={"query": "香港迪士尼怎么玩？"})
        self.assertEqual(workflow.status_code, 200)
        payload = workflow.json()
        self.assertIn("answer", payload)
        self.assertIn("trace", payload)
        self.assertIn("retrieved", payload)


if __name__ == "__main__":
    unittest.main()
