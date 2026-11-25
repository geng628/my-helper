"""推理模型协议插件（Slot 4 LLMProviderProtocol）：支持 DeepSeek Reasoner / R1 等 reasoning_content 协议模型。"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from personal_assistant.agent.plugins import Plugin
from personal_assistant.ai import LLMClient, LLMStreamEvent, OpenAIChatClient, consume_openai_sse

PROVIDER_NAME = "openai_reasoning"


# 为推理模型准备请求消息副本：仅跨轮（最后一条 user 之前）剥离历史 reasoning_content，且不改动原消息
def _prepare_wire_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = i

    return [
        {k: v for k, v in msg.items() if k != "reasoning_content"}
        if i < last_user_idx and "reasoning_content" in msg
        else msg
        for i, msg in enumerate(messages)
    ]


# 统一规范 assistant 消息的 reasoning_content：存在思维链或工具续传时保留字符串，否则移除冗余字段
def _apply_reasoning_content(message: dict[str, object], reasoning_text: str) -> dict[str, object]:
    if reasoning_text or message.get("tool_calls"):
        message["reasoning_content"] = reasoning_text
    else:
        message.pop("reasoning_content", None)
    return message


# 原生推理模型客户端：复用微内核 OpenAIChatClient 底座，仅声明 reasoning_content 生命周期与 SSE 钩子差异
@dataclass
class OpenAIReasoningLLM(OpenAIChatClient):
    model: str = "deepseek-reasoner"

    # 发往接口前跨轮剥离历史 reasoning_content，轮内工具调用思维链保持透传
    def _prepare_messages(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        return _prepare_wire_messages(messages)

    # 规范化同步响应中的 reasoning_content（工具续传时保底为字符串，纯文本空值时省去）
    def _normalize_response_message(self, message: dict[str, object]) -> dict[str, object]:
        raw = message.get("reasoning_content")
        reasoning_text = raw if isinstance(raw, str) else ""
        return _apply_reasoning_content(message, reasoning_text)

    # 挂载推理增量与收尾钩子：思维链首包计入 ttft_ms、不吐出主文本、收尾统一写入 reasoning_content
    def _consume_sse(self, lines: Iterable[str], t0: float) -> Iterator[LLMStreamEvent]:
        reasoning_parts: list[str] = []

        # 收集 reasoning_content 片段并告知底层将首包计入 ttft_ms
        def _on_delta(delta: dict[str, object]) -> bool:
            rc = delta.get("reasoning_content")
            if isinstance(rc, str) and rc:
                reasoning_parts.append(rc)
                return True
            return False

        # 在 StreamDone 消息中回写聚合后的 reasoning_content
        def _on_message(msg: dict[str, object]) -> None:
            _apply_reasoning_content(msg, "".join(reasoning_parts))

        return consume_openai_sse(lines, t0, on_delta=_on_delta, on_message=_on_message)


# 实现内层 LLMProviderProtocol 协议的推理模型构造器
class OpenAIReasoningProvider:
    name = PROVIDER_NAME

    # 按统一环境变量/Profile 映射构建 OpenAIReasoningLLM 实例
    def build(self, env: Mapping[str, str]) -> LLMClient:
        return OpenAIReasoningLLM(
            api_key=env.get("LLM_API_KEY", "").strip(),
            base_url=env.get("LLM_BASE_URL", "").strip() or "https://api.deepseek.com/v1",
            model=env.get("LLM_MODEL", "").strip() or "deepseek-reasoner",
        )


# 推理模型主板插件：通过 Slot 4（LLMProviderProtocol）注册 openai_reasoning
class ReasoningProviderPlugin(Plugin):
    plugin_id = PROVIDER_NAME

    # 初始化生命周期：向主板注册 OpenAIReasoningProvider
    def initialize(self) -> None:
        self.context.register_provider(OpenAIReasoningProvider())
