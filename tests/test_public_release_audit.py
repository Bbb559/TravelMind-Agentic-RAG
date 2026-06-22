import tempfile
import unittest
from pathlib import Path

from _helpers import ROOT
from scripts.public_release_audit import (
    audit_public_documentation,
    audit_public_tree,
    publishable_paths,
)
from scripts.export_public_release import export_public_tree


class PublicReleaseAuditTests(unittest.TestCase):
    def test_public_documentation_contract(self):
        report = audit_public_documentation(ROOT)
        self.assertTrue(report.ok, report.to_json())

    def test_sensitive_scan_rejects_values_and_private_paths_without_echoing_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_secret = "s" + "k-example-secret-value"
            (root / "README.md").write_text(
                "TRAVELMIND_LLM_API_KEY=\n"
                "TRAVELMIND_LLM_BASE_URL=\n"
                f"private={''.join(['D:', chr(92), '3', chr(92), 'private'])}\n"
                f"token={fake_secret}\n",
                encoding="utf-8",
            )

            report = audit_public_tree(root)

        self.assertFalse(report.ok)
        self.assertIn("absolute_private_path", report.error_codes)
        self.assertIn("secret_pattern", report.error_codes)
        serialized = report.to_json()
        self.assertNotIn(fake_secret, serialized)
        self.assertNotIn("private=", serialized)

    def test_sensitive_scan_rejects_forbidden_artifact_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            forbidden = [
                root / ".env",
                root / ".runtime" / "request.json",
                root / "run_logs" / "workflow.json",
                root / "backend.log",
                root / "docs" / "history" / "note.md",
            ]
            for path in forbidden:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("safe placeholder", encoding="utf-8")

            report = audit_public_tree(root)

        self.assertFalse(report.ok)
        self.assertEqual(
            sum(issue.code == "forbidden_artifact" for issue in report.issues),
            len(forbidden),
        )

    def test_sensitive_scan_compares_private_env_values_without_echoing_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "public"
            root.mkdir()
            private_env = base / "private.env"
            private_env.write_text(
                "TRAVELMIND_LLM_API_KEY=private-value-123456\n"
                "TRAVELMIND_LLM_BASE_URL=https://private.example/v1\n",
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "accidentally copied private-value-123456",
                encoding="utf-8",
            )

            report = audit_public_tree(root, secret_env_path=private_env)

        self.assertFalse(report.ok)
        self.assertIn("private_env_value", report.error_codes)
        self.assertNotIn("private-value-123456", report.to_json())

    def test_sensitive_scan_rejects_internal_process_terms(self):
        internal_terms = [
            "P" + "13.8",
            "Co" + "dex",
            "clean" + "up manifest",
            "Travel" + "_111",
            "pure" + "_ocr",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text(
                "\n".join(internal_terms),
                encoding="utf-8",
            )

            report = audit_public_tree(root)

        self.assertFalse(report.ok)
        self.assertEqual(
            sum(issue.code == "internal_process_term" for issue in report.issues),
            len(internal_terms),
        )

    def test_sensitive_scan_classifies_safe_security_identifiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "safe.py").write_text(
                'api_key = ""\n'
                'base_url = "https://example.invalid/v1"\n'
                'blocked = "traceback"\n'
                'runtime_path = ".runtime/run_logs"\n',
                encoding="utf-8",
            )

            report = audit_public_tree(root)

        self.assertTrue(report.ok, report.to_json())
        self.assertGreaterEqual(report.classifications["api_key_identifier"], 1)
        self.assertGreaterEqual(report.classifications["base_url_identifier"], 1)
        self.assertGreaterEqual(report.classifications["traceback_guard"], 1)
        self.assertGreaterEqual(report.classifications["runtime_guard"], 1)

    def test_publishable_paths_use_an_explicit_allowlist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            included = [
                root / ".env.example",
                root / "README.md",
                root / "evals" / "v1" / "route_cases.jsonl",
                root / "src" / "travelmind" / "api.py",
                root / "tests" / "test_api.py",
                root / "scripts" / "smoke.py",
            ]
            excluded = [
                root / ".env",
                root / ".runtime" / "request.json",
                root / "frontend" / "dist" / "index.html",
                root / "frontend" / "node_modules" / "pkg.js",
            ]
            for path in included + excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            selected = publishable_paths(root)

        for path in included:
            self.assertIn(path.relative_to(root).as_posix(), selected)
        for path in excluded:
            self.assertNotIn(path.relative_to(root).as_posix(), selected)

    def test_export_public_tree_copies_only_allowlisted_files_and_verifies_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            target = base / "target"
            source.mkdir()
            (source / "README.md").write_text("public", encoding="utf-8")
            (source / ".env").write_text("private", encoding="utf-8")
            code = source / "src" / "travelmind" / "api.py"
            code.parent.mkdir(parents=True)
            code.write_text("app = object()", encoding="utf-8")
            evaluation = source / "evals" / "v1" / "route_cases.jsonl"
            evaluation.parent.mkdir(parents=True)
            evaluation.write_text("{}\n", encoding="utf-8")

            result = export_public_tree(source, target)

        self.assertEqual(result.file_count, 3)
        self.assertTrue(result.verified)
        self.assertIn("README.md", result.paths)
        self.assertIn("src/travelmind/api.py", result.paths)
        self.assertIn("evals/v1/route_cases.jsonl", result.paths)
        self.assertNotIn(".env", result.paths)

    def test_export_public_tree_refuses_a_nonempty_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            target = base / "target"
            source.mkdir()
            target.mkdir()
            (source / "README.md").write_text("public", encoding="utf-8")
            (target / "keep.txt").write_text("do not overwrite", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_public_tree(source, target)

            self.assertEqual(
                (target / "keep.txt").read_text(encoding="utf-8"),
                "do not overwrite",
            )


if __name__ == "__main__":
    unittest.main()
