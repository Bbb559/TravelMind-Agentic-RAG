import json
import subprocess
import sys
import unittest

from _helpers import ROOT


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        proc = subprocess.run(
            [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "src" / "travelmind" / "cli.py"), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def test_inventory_outputs_assets(self):
        data = self.run_cli("--inventory")
        self.assertIn("assets", data["assets_dir"])
        self.assertGreater(data["csv_rows"], 0)

    def test_query_route_outputs_route_decision(self):
        data = self.run_cli("--query", "香港迪士尼怎么玩？")
        self.assertEqual(data["route"], "multimodal_rag")
        self.assertIn("confidence", data)

    def test_workflow_outputs_rag_answer(self):
        data = self.run_cli("--workflow", "--query", "大理到双廊怎么去？")
        self.assertEqual(data["route"], "naive_rag")
        self.assertIn("retrieved", data)

    def test_retrieve_alias_outputs_rag_answer(self):
        data = self.run_cli("--retrieve", "--query", "香港迪士尼怎么玩？")
        self.assertEqual(data["route"], "multimodal_rag")
        self.assertIn("sources", data)

    def test_cli_fallback_query(self):
        data = self.run_cli("--workflow", "--query", "qwxjkp")
        self.assertEqual(data["route"], "invalid_input")
        self.assertEqual(data["fallback_reason"], "invalid_input")

    def test_package_entrypoint(self):
        proc = subprocess.run(
            [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "travelmind", "--query", "澳门大三巴牌坊在哪里？"],
            cwd=ROOT,
            env={**dict(), "PYTHONPATH": str(ROOT / "src")},
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["route"], "multimodal_rag")

    def test_cli_json_parseable_for_graphrag(self):
        data = self.run_cli("--workflow", "--query", "对比西安和南京的人文景点")
        self.assertEqual(data["route"], "graphrag")

    def test_cli_json_parseable_for_hybrid(self):
        data = self.run_cli("--workflow", "--query", "台湾和西安哪个更适合亲子游？")
        self.assertEqual(data["route"], "hybrid_rag")
