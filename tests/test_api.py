import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from _helpers import ROOT
import travelmind.api as api_module
from travelmind.api import app, global_search_status
from travelmind.config import get_settings


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.graph_env = patch.dict(
            "os.environ",
            {
                "TRAVELMIND_GRAPHRAG_LLM_API_KEY": "",
                "TRAVELMIND_GRAPHRAG_LLM_BASE_URL": "",
                "TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL": "",
                "TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL": "",
                "TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED": "false",
            },
        )
        self.graph_env.start()
        get_settings.cache_clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.graph_env.stop()
        get_settings.cache_clear()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_inventory(self):
        response = self.client.get("/api/inventory")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["csv_rows"], 0)

    def test_route(self):
        response = self.client.post("/api/route", json={"query": "大理到双廊怎么去？"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "naive_rag")
        self.assertIn("run_id", payload)

    def test_workflow(self):
        response = self.client.post("/api/workflow", json={"query": "香港迪士尼怎么玩？"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "multimodal_rag")
        self.assertIn("answer", payload)
        self.assertIn("run_id", payload)
        self.assertIn("runtime_summary", payload)

    def test_workflow_defaults_global_search_request_to_disabled(self):
        payload = self.client.post(
            "/api/workflow",
            json={"query": "对比西安和南京的人文景点"},
        ).json()

        self.assertEqual(
            payload["global_search_status"],
            {
                "requested": False,
                "service_enabled": False,
                "effective_allowed": False,
                "executed": False,
                "succeeded": False,
            },
        )
        self.assertIn("graphrag:global_search_disabled", payload["trace"])

    def test_workflow_reports_request_and_service_gate_separately(self):
        with patch.dict(
            "os.environ",
            {"TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED": "true"},
        ):
            get_settings.cache_clear()
            try:
                payload = self.client.post(
                    "/api/workflow",
                    json={
                        "query": "香港迪士尼怎么玩？",
                        "allow_global_search": True,
                    },
                ).json()
            finally:
                get_settings.cache_clear()

        self.assertEqual(
            payload["global_search_status"],
            {
                "requested": True,
                "service_enabled": True,
                "effective_allowed": False,
                "executed": False,
                "succeeded": False,
            },
        )

    def test_graphrag_request_can_be_policy_allowed_without_claiming_execution(self):
        with patch.dict(
            "os.environ",
            {
                "TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED": "true",
                "TRAVELMIND_GRAPHRAG_LLM_API_KEY": "",
            },
        ):
            get_settings.cache_clear()
            try:
                payload = self.client.post(
                    "/api/workflow",
                    json={
                        "query": "对比西安和南京的人文景点",
                        "allow_global_search": True,
                    },
                ).json()
            finally:
                get_settings.cache_clear()

        modes = {
            item["metadata"].get("retrieval_mode")
            for item in payload["retrieved"]
        }
        self.assertTrue(payload["global_search_status"]["effective_allowed"])
        self.assertFalse(payload["global_search_status"]["executed"])
        self.assertFalse(payload["global_search_status"]["succeeded"])
        self.assertNotIn("graphrag_global_search", modes)

    def test_global_search_status_reports_actual_execution_from_trace(self):
        with patch.dict(
            "os.environ",
            {"TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED": "true"},
        ):
            get_settings.cache_clear()
            try:
                status = global_search_status(
                    requested=True,
                    route="graphrag",
                    trace=[
                        "graphrag:global_search_called",
                        "graphrag:global_search_succeeded",
                        "graphrag:retriever_mode:local_evidence:graphrag_low_relevance",
                    ],
                )
            finally:
                get_settings.cache_clear()

        self.assertEqual(
            status,
            {
                "requested": True,
                "service_enabled": True,
                "effective_allowed": True,
                "executed": True,
                "succeeded": True,
            },
        )

    def test_workflow_structure_for_fallback(self):
        payload = self.client.post("/api/workflow", json={"query": "qwxjkp"}).json()
        for key in ["answer", "route", "confidence", "sources", "retrieved", "fallback_reason", "trace"]:
            self.assertIn(key, payload)

    def test_invalid_input_returns_domain_result_with_structured_status(self):
        payload = self.client.post(
            "/api/workflow",
            json={"query": " ！！！ "},
        ).json()

        self.assertEqual(payload["route"], "invalid_input")
        self.assertEqual(payload["answer"], "请先输入具体旅游问题。")
        self.assertEqual(payload["retrieved"], [])
        self.assertEqual(
            payload["execution_status"],
            {
                "agent": None,
                "retrieval_mode": "none",
                "evidence_status": "not_run",
                "generation_mode": "none",
                "llm_stages": {
                    "router": "disabled",
                    "grade": "disabled",
                    "rewrite": "disabled",
                    "generate": "disabled",
                },
            },
        )
        self.assertIsNone(payload["hybrid_branch_status"])

    def test_naive_response_exposes_template_generation_status(self):
        payload = self.client.post(
            "/api/workflow",
            json={"query": "成都都江堰适合怎么玩？"},
        ).json()

        self.assertEqual(payload["execution_status"]["agent"], "NaiveTravelAgent")
        self.assertIn(payload["execution_status"]["retrieval_mode"], {"faiss", "csv"})
        self.assertEqual(payload["execution_status"]["evidence_status"], "sufficient")
        self.assertEqual(payload["execution_status"]["generation_mode"], "template")

    def test_route_and_workflow_agree(self):
        query = "台湾和西安哪个更适合亲子游？"
        route = self.client.post("/api/route", json={"query": query}).json()["route"]
        workflow = self.client.post("/api/workflow", json={"query": query}).json()["route"]
        self.assertEqual(route, workflow)

    def test_global_search_request_does_not_change_naive_route(self):
        query = "荔波小七孔怎么玩比较合适？"
        disabled = self.client.post(
            "/api/workflow",
            json={"query": query, "allow_global_search": False},
        ).json()
        requested = self.client.post(
            "/api/workflow",
            json={"query": query, "allow_global_search": True},
        ).json()

        self.assertEqual(disabled["route"], "naive_rag")
        self.assertEqual(requested["route"], "naive_rag")

    def test_hybrid_response_exposes_aggregator_identity(self):
        payload = self.client.post(
            "/api/workflow",
            json={"query": "香港和成都哪个更适合亲子游？"},
        ).json()

        self.assertEqual(payload["route"], "hybrid_rag")
        self.assertIn("agent:hybrid_aggregator:start", payload["trace"])
        self.assertIn("agent:hybrid_aggregator:end", payload["trace"])

    def test_invalid_shape_returns_422(self):
        response = self.client.post("/api/workflow", json={"bad": "query"})
        self.assertEqual(response.status_code, 422)

    def test_cors_preflight_allows_local_vite_origin(self):
        for path in ["/api/route", "/api/workflow"]:
            response = self.client.options(
                path,
                headers={
                    "Origin": "http://127.0.0.1:5173",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type,x-travelmind-run-id",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:5173")

    def test_workflow_run_log_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            api_module,
            "RUN_LOG_DIR",
            Path(tmp),
        ), patch.dict(
            "os.environ",
            {"TRAVELMIND_RUN_LOG_ENABLED": "false"},
        ):
            get_settings.cache_clear()
            response = self.client.post(
                "/api/workflow",
                json={"query": "qwxjkp"},
                headers={"X-TravelMind-Run-Id": "disabled-run-log"},
            )
            get_settings.cache_clear()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(list(Path(tmp).glob("*.json")), [])

    def test_workflow_writes_run_log_only_when_explicitly_enabled(self):
        run_id = "enabled-run-log"
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            api_module,
            "RUN_LOG_DIR",
            Path(tmp),
        ), patch.dict(
            "os.environ",
            {"TRAVELMIND_RUN_LOG_ENABLED": "true"},
        ):
            get_settings.cache_clear()
            response = self.client.post(
                "/api/workflow",
                json={"query": "qwxjkp"},
                headers={"X-TravelMind-Run-Id": run_id},
            )
            get_settings.cache_clear()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["run_id"], run_id)
            logs = list(Path(tmp).glob(f"*{run_id}_workflow.json"))
            self.assertTrue(logs)
            text = logs[-1].read_text(encoding="utf-8")
            self.assertIn(run_id, text)
            self.assertNotIn("TRAVELMIND_LLM_API_KEY", text)
            self.assertNotIn("TRAVELMIND_GRAPHRAG_LLM_API_KEY", text)

    def test_runtime_summary_only_exposes_booleans(self):
        payload = self.client.post("/api/workflow", json={"query": "qwxjkp"}).json()
        summary = payload["runtime_summary"]
        self.assertEqual(set(summary), {"llm_enabled", "key_present"})
        self.assertTrue(all(isinstance(value, bool) for value in summary.values()))
        text = str(summary).lower()
        for forbidden in ["api_key", "deepseek", "qwen", "sk-", "length", "prefix", "suffix"]:
            self.assertNotIn(forbidden, text)
