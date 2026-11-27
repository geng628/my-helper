"""Layer 1 内建协议翻译器汇总包（Pi ch04 第三层）：统一注册 OpenAI / Anthropic / Gemini 三大原生协议。"""
from __future__ import annotations

from personal_assistant.ai.providers.anthropic_messages import AnthropicMessagesProvider
from personal_assistant.ai.providers.google_genai import GoogleGenAIProvider
from personal_assistant.ai.providers.http import (
    PostJson,
    PostStream,
    default_post_json,
    default_post_stream,
    format_http_error,
    iter_sse_events,
    parse_retry_after_seconds,
    redact_error_tokens,
)
from personal_assistant.ai.providers.openai_completions import (
    OpenAICompletionsProvider,
    extract_openai_usage,
)
from personal_assistant.ai.registry import ApiProviderRegistry


# 向目标注册表登记三大内建原生协议翻译器（openai-completions / anthropic-messages / google-generative-ai）
def register_builtin_providers(registry: ApiProviderRegistry) -> None:
    registry.register(OpenAICompletionsProvider())
    registry.register(AnthropicMessagesProvider())
    registry.register(GoogleGenAIProvider())


__all__ = [
    "AnthropicMessagesProvider",
    "GoogleGenAIProvider",
    "OpenAICompletionsProvider",
    "PostJson",
    "PostStream",
    "default_post_json",
    "default_post_stream",
    "extract_openai_usage",
    "format_http_error",
    "iter_sse_events",
    "parse_retry_after_seconds",
    "redact_error_tokens",
    "register_builtin_providers",
]
