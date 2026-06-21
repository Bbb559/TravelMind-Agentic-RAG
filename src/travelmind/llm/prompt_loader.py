"""安全读取包内 prompt 草稿。"""

from __future__ import annotations

from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("prompt_path_not_allowed")
    target = (PROMPT_ROOT / path).resolve()
    if not str(target).startswith(str(PROMPT_ROOT.resolve())):
        raise ValueError("prompt_path_not_allowed")
    return target.read_text(encoding="utf-8")
