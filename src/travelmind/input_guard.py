"""用户问题进入 Router 前的轻量输入门禁。"""

from __future__ import annotations

import re
import unicodedata

_GREETINGS = {
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "hello",
    "hi",
    "在吗",
}


def is_invalid_input(query: str) -> bool:
    text = query.strip()
    if not text:
        return True
    normalized = re.sub(r"\s+", "", text).lower()
    if normalized in _GREETINGS:
        return True
    if all(
        unicodedata.category(char).startswith(("P", "S"))
        for char in normalized
    ):
        return True
    if re.fullmatch(r"[a-z0-9_]{4,}", normalized):
        return True
    return False
