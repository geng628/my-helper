"""Layer 1 推理模型协议翻译器（api = "openai-reasoning"）：双向翻译 DeepSeek Reasoner / R1 等 reasoning_content 方言。"""
from __future__ import annotations

from dataclasses import dataclass

from personal_assistant.ai.protocol import (
    AssistantMessage,
    ModelProfile,
    StreamOptions,
)
from personal_assistant.ai.providers.openai_completions import OpenAICompletionsProvider


# DeepSeek Reasoner / R1 推理协议翻译器：复用 OpenAI SSE 底座，双向翻译 reasoning_content <-> ThinkingBlock
@dataclass
class OpenAIReasoningProvider(OpenAICompletionsProvider):
    api: str = "openai-reasoning"

    # 声明流式 delta 中需实时翻译为 ThinkingStart/Delta/End 与 ThinkingBlock 的思维链字段
    def _thinking_delta_fields(self) -> tuple[str, ...]:
        return ("reasoning_content",)

    # 推理模型不发送 temperature 与 reasoning_effort，仅在显式配置时透传 max_tokens
    def _extra_payload(
        self,
        model: ModelProfile,
        options: StreamOptions | None,
    ) -> dict[str, object]:
        max_tok = (
            options.max_tokens
            if (options is not None and options.max_tokens is not None)
            else model.max_tokens
        )
        return {"max_tokens": max_tok} if max_tok is not None else {}

    # 跨轮剥离旧思维链、轮内续传当前思维链（轮内含工具调用时即使无思考内容也补齐空串以满足协议要求）
    def _serialize_assistant_message(
        self,
        message: AssistantMessage,
        *,
        is_current_turn: bool,
    ) -> dict[str, object]:
        wire = super()._serialize_assistant_message(message, is_current_turn=is_current_turn)
        if is_current_turn and (message.thinking or message.tool_calls):
            wire["reasoning_content"] = message.thinking
        return wire
