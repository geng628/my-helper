"""零第三方依赖的 OpenAI 兼容大模型客户端。"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from urllib.request import Request, urlopen


# 当未配置 LLM_MODE=on 或缺少 API Key 时抛出的异常
class LLMNotConfigured(RuntimeError):
    """未配置大模型参数时抛出。"""


PostJson = Callable[[str, dict[str, object], dict[str, str]], dict[str, object]]


# 使用 Python 标准库 urllib 向指定接口发送 JSON POST 请求并解析返回体
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


# 大模型客户端抽象基类，供 ReAct 循环调用
class LLMClient(ABC):
    # 发送对话历史与工具定义，返回模型本轮的 assistant 消息（含纯文本或 tool_calls）
    @abstractmethod
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]: ...


# 占位客户端：未启用模型时给出友好提示，避免程序崩溃
class NoopLLMClient(LLMClient):
    # 直接抛出未配置异常，提示用户设置环境变量
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        raise LLMNotConfigured(
            "LLM 未启用。请在环境变量或 .env 中设置 LLM_MODE=on 和 LLM_API_KEY。"
        )


# OpenAI 兼容 Chat Completions 客户端（支持 DeepSeek / OpenAI / 通义千问等）
@dataclass(frozen=True)
class OpenAICompatibleLLM(LLMClient):
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    post_json: PostJson = default_post_json

    # 组装 OpenAI 格式请求体调用 /chat/completions，原样返回首条 assistant 消息
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


# 从环境变量读取配置，返回可用的 LLM 客户端（未开启则返回 NoopLLMClient）
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
