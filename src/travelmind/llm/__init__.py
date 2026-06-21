"""LLM 客户端与 prompt 读取。"""

from .client import LLMClientProtocol, OpenAICompatibleClient
from .prompt_loader import load_prompt

__all__ = ["LLMClientProtocol", "OpenAICompatibleClient", "load_prompt"]
