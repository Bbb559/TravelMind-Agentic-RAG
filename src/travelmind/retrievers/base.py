"""检索器协议。"""

from __future__ import annotations

from typing import Protocol

from travelmind.schemas import RetrieverResult


class BaseRetriever(Protocol):
    def retrieve(self, query: str) -> list[RetrieverResult]:
        """根据 query 返回统一检索结果。"""
