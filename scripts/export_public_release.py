"""将通过验证的公开文件白名单导出到干净目录。"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.public_release_audit import publishable_paths
except ImportError:  # 支持直接执行脚本。
    from public_release_audit import publishable_paths


@dataclass(frozen=True)
class ExportResult:
    source: str
    target: str
    file_count: int
    paths: tuple[str, ...]
    verified: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_public_tree(source: Path, target: Path) -> ExportResult:
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError("source_and_target_must_differ")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("target_must_be_empty")
    target.mkdir(parents=True, exist_ok=True)

    selected = tuple(sorted(publishable_paths(source)))
    for relative in selected:
        source_path = source / relative
        target_path = target / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        if _sha256(source_path) != _sha256(target_path):
            raise OSError(f"hash_verification_failed:{relative}")

    return ExportResult(
        source=str(source),
        target=str(target),
        file_count=len(selected),
        paths=selected,
        verified=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--source", default=".")
    args = parser.parse_args()
    result = export_public_tree(Path(args.source), Path(args.target))
    print(
        f"public export: PASS files={result.file_count} verified={result.verified}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
