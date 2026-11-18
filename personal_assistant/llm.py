"""Zero-dependency OpenAI-compatible Chat Completions client."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from urllib.request import Request, urlopen


class LLMNotConfigured(RuntimeError):
    """Raised when an LLM call is requested but LLM_MODE/LLM_API_KEY is not set."""


PostJson = Callable[[str, dict[str, object], dict[str, str]], dict[str, object]]


def default_post_json(
    url: str, payload: dict[str, object], headers: dict[str, str]
) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "my-helper-v2/0.1",
            **headers,
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


class LLMClient(ABC):
    """Minimal chat-completions interface used by the ReAct loop."""

    @abstractmethod
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]: ...


class NoopLLMClient(LLMClient):
    """Fallback client when LLM_MODE is off or API key is missing."""

    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        raise LLMNotConfigured(
            "LLM 未启用。请在环境变量或 .env 中设置 LLM_MODE=on 和 LLM_API_KEY。"
        )


@dataclass(frozen=True)
class OpenAICompatibleLLM(LLMClient):
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    post_json: PostJson = default_post_json

    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = self.post_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return dict(response["choices"][0]["message"])


def from_env() -> LLMClient:
    mode = os.getenv("LLM_MODE", "off").strip().lower() or "off"
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if mode != "on" or not api_key:
        return NoopLLMClient()
    return OpenAICompatibleLLM(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
    )
