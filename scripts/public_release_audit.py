"""Validate that a candidate public tree is safe to publish."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
}
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".mjs",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
ABSOLUTE_PRIVATE_PATH = re.compile(r"(?i)\bD:[\\/]3[\\/]")
PUBLIC_CONFIG_VALUES = {
    "https://api.deepseek.com",
    "deepseek-chat",
    "text-embedding-v3",
}
INTERNAL_PROCESS_PATTERNS = (
    re.compile(r"\bP1[34](?:\.[A-Za-z0-9]+)*\b", re.IGNORECASE),
    re.compile("Co" + "dex", re.IGNORECASE),
    re.compile("clean" + "up", re.IGNORECASE),
    re.compile("Travel" + "_111", re.IGNORECASE),
    re.compile("pure" + "_ocr", re.IGNORECASE),
    re.compile("内部" + "审计"),
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
REQUIRED_PUBLIC_DOCUMENTS = (
    "README.md",
    "LICENSE",
    "docs/architecture.md",
    "docs/configuration.md",
    "docs/evaluation.md",
    "docs/assets.md",
)
PUBLIC_TOP_LEVEL_FILES = {
    ".env.example",
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "requirements-assets.txt",
    "requirements-dev.txt",
    "requirements.txt",
}
PUBLIC_TOP_LEVEL_DIRS = {
    "assets",
    "docs",
    "evals",
    "frontend",
    "scripts",
    "src",
    "tests",
}


@dataclass(frozen=True)
class AuditIssue:
    code: str
    path: str
    line: int | None = None
    severity: str = "error"


@dataclass
class AuditReport:
    root: str
    issues: list[AuditIssue] = field(default_factory=list)
    scanned_files: int = 0
    classifications: dict[str, int] = field(
        default_factory=lambda: {
            "api_key_identifier": 0,
            "base_url_identifier": 0,
            "traceback_guard": 0,
            "runtime_guard": 0,
        }
    )

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_codes(self) -> set[str]:
        return {
            issue.code
            for issue in self.issues
            if issue.severity == "error"
        }

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "root": self.root,
            "scanned_files": self.scanned_files,
            "classifications": dict(self.classifications),
            "issues": [asdict(issue) for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def _iter_candidate_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        yield path


def _iter_publishable_files(root: Path):
    for name in sorted(PUBLIC_TOP_LEVEL_FILES):
        path = root / name
        if path.is_file():
            yield path
    for name in sorted(PUBLIC_TOP_LEVEL_DIRS):
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in IGNORED_DIRS for part in relative.parts):
                continue
            yield path


def publishable_paths(root: Path) -> set[str]:
    root = root.resolve()
    return {
        path.relative_to(root).as_posix()
        for path in _iter_publishable_files(root)
    }


def _iter_tracked_files(root: Path):
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        path = root / relative
        if path.is_file():
            yield path


def _is_text_file(path: Path) -> bool:
    return (
        path.name in {".env.example", ".gitignore", ".gitattributes", "LICENSE"}
        or path.suffix.lower() in TEXT_SUFFIXES
    )


def _is_forbidden_artifact(relative: Path) -> bool:
    parts = relative.parts
    name = relative.name
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    if ".runtime" in parts or "run_logs" in parts:
        return True
    if len(parts) >= 2 and parts[0] == "docs" and parts[1] == "history":
        return True
    return name.lower().endswith(".log")


def _private_env_values(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    values: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip().strip("\"'")
        if not value or value in PUBLIC_CONFIG_VALUES:
            continue
        if not any(term in key for term in ("API_KEY", "BASE_URL", "MODEL")):
            continue
        if len(value) >= 6:
            values.add(value)
    return values


def audit_public_tree(
    root: Path,
    *,
    secret_env_path: Path | None = None,
    file_mode: str = "tree",
) -> AuditReport:
    root = root.resolve()
    report = AuditReport(root=str(root))
    private_values = _private_env_values(secret_env_path)
    if file_mode == "tree":
        files = _iter_candidate_files(root)
    elif file_mode == "publishable":
        files = _iter_publishable_files(root)
    elif file_mode == "tracked":
        files = _iter_tracked_files(root)
    else:
        raise ValueError(f"unsupported_file_mode:{file_mode}")
    for path in files:
        report.scanned_files += 1
        relative_path = path.relative_to(root)
        if _is_forbidden_artifact(relative_path):
            report.issues.append(
                AuditIssue("forbidden_artifact", relative_path.as_posix())
            )
        if not _is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = relative_path.as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            lower_line = line.lower()
            if "api_key" in lower_line:
                report.classifications["api_key_identifier"] += 1
            if "base_url" in lower_line:
                report.classifications["base_url_identifier"] += 1
            if "traceback" in lower_line:
                report.classifications["traceback_guard"] += 1
            if ".runtime" in lower_line or "run_logs" in lower_line:
                report.classifications["runtime_guard"] += 1
            if ABSOLUTE_PRIVATE_PATH.search(line):
                report.issues.append(
                    AuditIssue("absolute_private_path", relative, line_number)
                )
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                report.issues.append(
                    AuditIssue("secret_pattern", relative, line_number)
                )
            if any(value in line for value in private_values):
                report.issues.append(
                    AuditIssue("private_env_value", relative, line_number)
                )
            if any(pattern.search(line) for pattern in INTERNAL_PROCESS_PATTERNS):
                report.issues.append(
                    AuditIssue("internal_process_term", relative, line_number)
                )
    return report


def _env_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        names.add(line.split("=", 1)[0].strip())
    return names


def audit_public_documentation(root: Path) -> AuditReport:
    root = root.resolve()
    report = AuditReport(root=str(root))
    for relative in REQUIRED_PUBLIC_DOCUMENTS:
        if not (root / relative).is_file():
            report.issues.append(
                AuditIssue("documentation_missing", relative)
            )

    markdown_files = [
        root / "README.md",
        root / "docs" / "architecture.md",
        root / "docs" / "configuration.md",
        root / "docs" / "evaluation.md",
        root / "docs" / "assets.md",
    ]
    for path in markdown_files:
        if not path.exists():
            continue
        report.scanned_files += 1
        text = path.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            clean_target = target.split("#", 1)[0].strip()
            if (
                not clean_target
                or clean_target.startswith(("http://", "https://", "mailto:"))
            ):
                continue
            resolved = (path.parent / clean_target).resolve()
            if not resolved.exists():
                report.issues.append(
                    AuditIssue(
                        "broken_markdown_link",
                        path.relative_to(root).as_posix(),
                    )
                )

    config_path = root / "docs" / "configuration.md"
    if config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")
        expected_env = _env_names(root / ".env.example") | _env_names(
            root / "frontend" / ".env.example"
        )
        for name in sorted(expected_env):
            if name not in config_text:
                report.issues.append(
                    AuditIssue("missing_config_documentation", name)
                )

    readme_path = root / "README.md"
    assets_path = root / "docs" / "assets.md"
    readme = (
        readme_path.read_text(encoding="utf-8")
        if readme_path.exists()
        else ""
    )
    assets = (
        assets_path.read_text(encoding="utf-8")
        if assets_path.exists()
        else ""
    )
    for command in ("git lfs install", "git lfs pull"):
        if command not in readme:
            report.issues.append(
                AuditIssue("missing_lfs_instruction", "README.md")
            )
    boundary_terms = ("MIT License", "Demo 资产", "源代码")
    if not all(term in readme for term in boundary_terms):
        report.issues.append(
            AuditIssue("missing_license_boundary", "README.md")
        )
    if not all(term in assets for term in boundary_terms):
        report.issues.append(
            AuditIssue("missing_license_boundary", "docs/assets.md")
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--secret-env")
    parser.add_argument(
        "--file-mode",
        choices=("tree", "publishable", "tracked"),
        default="tree",
    )
    parser.add_argument("--check-docs", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_public_tree(
        Path(args.root),
        secret_env_path=Path(args.secret_env) if args.secret_env else None,
        file_mode=args.file_mode,
    )
    if args.check_docs:
        docs_report = audit_public_documentation(Path(args.root))
        report.scanned_files += docs_report.scanned_files
        report.issues.extend(docs_report.issues)
    if args.json:
        print(report.to_json())
    else:
        status = "PASS" if report.ok else "FAIL"
        print(f"public release audit: {status} ({report.scanned_files} files)")
        for issue in report.issues:
            location = f"{issue.path}:{issue.line}" if issue.line else issue.path
            print(f"- {issue.severity}: {issue.code} at {location}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
