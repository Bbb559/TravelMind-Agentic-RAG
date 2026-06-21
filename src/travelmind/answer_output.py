"""最终用户答案的安全展示清洗。"""

from __future__ import annotations

import re

_DATA_CITATION = re.compile(r"\s*\[Data:\s*[^\]]+\]", re.IGNORECASE)
_INTERNAL_REFERENCE = re.compile(
    r"\s*\b(?:Reports|Entities|Relationships|Sources)\s*\([^)]*\)",
    re.IGNORECASE,
)
_INTERNAL_TABLE = re.compile(
    r"\b(?:community_reports|text_units|entities|relationships|sources)"
    r"(?:\.parquet)?\b",
    re.IGNORECASE,
)


def sanitize_user_answer(answer: str) -> str:
    text = _DATA_CITATION.sub("", answer)
    text = _INTERNAL_REFERENCE.sub("", text)
    text = _INTERNAL_TABLE.sub("", text)
    text = re.sub(r"[ \t]+([，。；：！？,.!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
