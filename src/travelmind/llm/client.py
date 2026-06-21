"""OpenAI-compatible LLM 客户端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from travelmind.config import ProjectSettings


class LLMClientProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        """根据 prompt 返回文本。"""


class OpenAICompatibleClient:
    """最小 OpenAI-compatible Chat Completions 客户端。"""

    def __init__(self, settings: ProjectSettings, timeout_seconds: int = 60) -> None:
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self.settings.llm_api_key and self.settings.llm_base_url and self.settings.llm_model)

    def generate(self, prompt: str) -> str:
        if not self.available:
            raise RuntimeError("llm_not_configured")
        base_url = self.settings.llm_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llm_request_failed:{exc.__class__.__name__}") from exc
        data = json.loads(body)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return str(content).strip()
