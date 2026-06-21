"""官方 GraphRAG 查询结果的安全归一化与来源摘要。"""

from __future__ import annotations

import json
from typing import Any


def normalize_official_response(payload: Any, *, prefix: str) -> tuple[str, Any]:
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError(f"{prefix}:invalid_response")
    response, context = payload
    if not isinstance(response, (str, dict, list)):
        raise ValueError(f"{prefix}:invalid_response")
    text = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
    if not text.strip():
        raise ValueError(f"{prefix}:empty_response")
    return text, context


def summarize_official_context(context: Any) -> list[dict[str, Any]]:
    """只返回安全的 section、行数和少量标题/ID，不透传原始 context。"""
    if isinstance(context, list):
        sections = [(f"section_{index + 1}", value) for index, value in enumerate(context)]
    elif isinstance(context, dict):
        sections = [(str(key), value) for key, value in context.items()]
    else:
        sections = []

    summary: list[dict[str, Any]] = []
    for section, value in sections:
        item = _summarize_section(section, value)
        if item is not None:
            summary.append(item)
    return summary


def _summarize_section(section: str, value: Any) -> dict[str, Any] | None:
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - GraphRAG runtime requires pandas
        pd = None

    if pd is not None and isinstance(value, pd.DataFrame):
        frame = value
        if frame.empty:
            return None
        if "in_context" in frame.columns:
            included = frame[frame["in_context"].fillna(False).astype(bool)]
            if included.empty:
                return None
            frame = included
        return {
            "section": section,
            "row_count": int(len(frame)),
            "titles_or_ids": _frame_labels(frame),
        }

    if isinstance(value, list):
        rows = [row for row in value if row not in (None, "", {}, [])]
        if not rows:
            return None
        return {
            "section": section,
            "row_count": len(rows),
            "titles_or_ids": [_safe_label(row) for row in rows[:3]],
        }

    if isinstance(value, dict) and value:
        return {
            "section": section,
            "row_count": 1,
            "titles_or_ids": [_safe_label(value)],
        }
    return None


def _frame_labels(frame) -> list[str]:
    preferred = (
        "title",
        "name",
        "id",
        "human_readable_id",
        "source",
        "target",
    )
    column = next((name for name in preferred if name in frame.columns), None)
    if column is None:
        return []
    return [
        str(value)[:120]
        for value in frame[column].dropna().astype(str).head(3).tolist()
        if str(value).strip()
    ]


def _safe_label(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("title", "name", "id", "human_readable_id", "source"):
            candidate = value.get(key)
            if candidate not in (None, ""):
                return str(candidate)[:120]
        return "record"
    return str(value)[:120]
