"""推理模型协议插件（Slot 4 LLMProviderProtocol）：支持 DeepSeek Reasoner / R1 等 reasoning_content 协议模型。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from personal_assistant.agent.plugins import Plugin
from personal_assistant.ai import LLMClient, OpenAIChatClient

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


# 原生推理模型客户端：复用微内核 OpenAIChatClient 底座，仅声明 reasoning_content 流式采集与消息边界差异
@dataclass
class OpenAIReasoningLLM(OpenAIChatClient):
    # 声明流式增量中需聚合的思维链字段（计入 ttft_ms，不吐出主文本）
    def _stream_delta_fields(self) -> tuple[str, ...]:
        return ("reasoning_content",)

    # 发往接口前跨轮剥离历史 reasoning_content，轮内工具调用思维链保持透传
    def _prepare_messages(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        return _prepare_wire_messages(messages)

    # 统一规范同步与流式响应中的 reasoning_content：存在思维链或工具续传时保留字符串，否则移除冗余键
    def _normalize_response_message(self, message: dict[str, object]) -> dict[str, object]:
        raw = message.get("reasoning_content")
        reasoning_text = raw if isinstance(raw, str) else ""
        if reasoning_text or message.get("tool_calls"):
            message["reasoning_content"] = reasoning_text
        else:
            message.pop("reasoning_content", None)
        return message


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
