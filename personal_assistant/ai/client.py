"""Layer 1 零依赖大模型客户端与 Provider 注册中心（对标 pi-ai）：不感知任何 Agent 或业务模块。"""
from __future__ import annotations

import http.client
import json
import math
import os
import re
import ssl
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from personal_assistant.ai.types import (
    LLMStreamEvent,
    ModelProfile,
    StreamDone,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
    ToolCall,
)


# 当未配置 LLM_MODE=on 或缺少 API Key 时抛出的异常
class LLMNotConfigured(RuntimeError):
    """未配置大模型参数时抛出。"""


PostJson = Callable[[str, dict[str, object], dict[str, str]], dict[str, object]]
PostStream = Callable[[str, dict[str, object], dict[str, str]], Iterator[str]]
ProviderFactory = Callable[[ModelProfile], "LLMClient"]


# 安全将接口返回的数字字段解析为非负整数
def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


# 从 OpenAI / DeepSeek / 通义千问等兼容响应中提取 Token 与缓存命中统计
def extract_usage(response: dict[str, object]) -> TokenUsage | None:
    raw = response.get("usage")
    if not isinstance(raw, dict):
        return None
    prompt_tokens = _non_negative_int(raw.get("prompt_tokens") or raw.get("input_tokens"))
    completion_tokens = _non_negative_int(raw.get("completion_tokens") or raw.get("output_tokens"))

    # 兼容 DeepSeek 顶级字段、OpenAI 标准 prompt_tokens_details 及 Anthropic 代理字段
    cached_raw: object = raw.get("prompt_cache_hit_tokens")
    if cached_raw is None:
        details = raw.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached_raw = details.get("cached_tokens")
    if cached_raw is None:
        cached_raw = raw.get("cache_read_input_tokens")
    cached_tokens = _non_negative_int(cached_raw)
    if prompt_tokens > 0:
        cached_tokens = min(cached_tokens, prompt_tokens)

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )


_T = TypeVar("_T")
_TRANSIENT_HTTP_ERRORS = (
    URLError,
    TimeoutError,
    ConnectionError,
    http.client.IncompleteRead,
    ssl.SSLError,
)
_BEARER_SECRET_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_SK_SECRET_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9._-]+")
_MAX_HTTP_ERROR_BODY_BYTES = 1024
_STREAM_OPTIONS_MARKERS = ("stream_options", "include_usage")
_CONTEXT_LIMIT_MARKERS = ("context_length", "maximum context", "tokens")


# 脱敏 HTTP 错误文本中的疑似 Bearer 或 sk- 令牌，防止错误透出泄漏密钥
def _redact_error_tokens(text: str) -> str:
    masked = _BEARER_SECRET_RE.sub("Bearer ***", text)
    return _SK_SECRET_RE.sub("sk-***", masked)


# 纯函数读取最多 1024 字节 HTTPError 响应体并脱敏，构造零猴子补丁的新 HTTPError 与组合诊断文本
def _format_http_error(error: HTTPError) -> tuple[HTTPError, str]:
    orig_reason = str(error.reason or "").strip()
    body = ""
    try:
        raw = error.read(_MAX_HTTP_ERROR_BODY_BYTES)
    except (OSError, ValueError, AttributeError):
        raw = b""
    if isinstance(raw, (bytes, bytearray)):
        body = raw.decode("utf-8", errors="replace").strip()
    elif isinstance(raw, str):
        body = raw.strip()
    clean_body = _redact_error_tokens(body)
    if clean_body and clean_body not in orig_reason:
        msg = f"{orig_reason} | {clean_body}" if orig_reason else clean_body
    else:
        msg = orig_reason
    return HTTPError(error.url, error.code, msg, error.headers, None), f"{orig_reason} {clean_body}".strip()


# 从 HTTP 429 响应头解析合法数字 Retry-After，并夹紧在 [0.1, 10.0] 秒内
def _parse_retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    raw_val = headers.get("Retry-After") if hasattr(headers, "get") else None
    if raw_val is None and isinstance(headers, Mapping):
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == "retry-after":
                raw_val = value
                break
    if isinstance(raw_val, bool) or raw_val is None:
        return None
    try:
        parsed = float(str(raw_val).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return max(0.1, min(10.0, parsed))


# 判断 HTTP 400 错误文本是否明确包含 stream_options / include_usage 不兼容标记（排除上下文超长错误）
def _is_stream_options_incompat(error_text: str) -> bool:
    lowered = error_text.lower()
    if any(marker in lowered for marker in _CONTEXT_LIMIT_MARKERS):
        return False
    return any(marker in lowered for marker in _STREAM_OPTIONS_MARKERS)


# 构造 JSON POST 请求，并在连接或指定读取阶段遇到瞬时网络/TLS/5xx/429 异常时按指数退避或 Retry-After 重试
def _post_with_retry(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
    operation: Callable[[Any], _T],
) -> _T:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "my-helper-v2/0.1",
        **headers,
    }
    for attempt in range(max_retries + 1):
        request = Request(url, data=body, headers=req_headers, method="POST")
        delay = 0.5 * (2**attempt)
        try:
            return operation(urlopen(request, timeout=60))
        except HTTPError as error:
            if (error.code < 500 and error.code != 429) or attempt >= max_retries:
                formatted_error, _ = _format_http_error(error)
                raise formatted_error from error
            if error.code == 429:
                retry_after = _parse_retry_after_seconds(error.headers)
                if retry_after is not None:
                    delay = retry_after
        except _TRANSIENT_HTTP_ERRORS:
            if attempt >= max_retries:
                raise
        time.sleep(delay)
    raise RuntimeError("请求重试耗尽。")


# 完整读取 HTTP 响应字节流并按 UTF-8 解码为字符串
def _read_utf8_body(response: Any) -> str:
    with response:
        return response.read().decode("utf-8")


# 使用 Python 标准库 urllib 向指定接口发送 JSON POST 请求，将响应读取与解码纳入瞬时网络/TLS 抖动重试闭环
def default_post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
) -> dict[str, object]:
    raw_text = _post_with_retry(
        url,
        payload,
        headers,
        max_retries=max_retries,
        operation=_read_utf8_body,
    )
    return json.loads(raw_text)


# 使用 Python 标准库 urllib 向指定接口发起 SSE 流式 POST 请求并逐行吐出 UTF-8 文本（仅在连接建立前重试）
def default_post_stream(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
) -> Iterator[str]:
    with _post_with_retry(
        url,
        payload,
        headers,
        max_retries=max_retries,
        operation=lambda response: response,
    ) as response:
        for raw_line in response:
            yield raw_line.decode("utf-8")


# 流式 SSE 增量工具调用分片累加器
@dataclass
class _ToolCallAcc:
    id: str = ""
    name: str = ""
    args: list[str] = field(default_factory=list)


# 逐行解析 OpenAI 兼容 SSE 流：实时吐出 ThinkingDelta/TextDelta、拼接分片 tool_calls 并聚合思维链字段
def _consume_openai_sse(
    lines: Iterable[str],
    t0: float,
    *,
    thinking_delta_fields: Iterable[str] = (),
) -> Iterator[LLMStreamEvent]:
    thinking_fields = tuple(dict.fromkeys(thinking_delta_fields))
    thinking_parts: dict[str, list[str]] = {name: [] for name in thinking_fields}
    role = "assistant"
    parts: list[str] = []
    tool_acc: dict[int, _ToolCallAcc] = {}
    usage: TokenUsage | None = None
    ttft_ms: float | None = None
    finish_reason: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str:
            continue
        if data_str == "[DONE]":
            break
        chunk = json.loads(data_str)
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error") is not None:
            raise RuntimeError(f"流式响应返回错误：{chunk['error']}")
        usage = extract_usage(chunk) or usage
        choices = chunk.get("choices") or []
        first_choice = choices[0] if choices and isinstance(choices[0], dict) else None
        if first_choice is not None:
            raw_finish = first_choice.get("finish_reason")
            if isinstance(raw_finish, str) and raw_finish:
                finish_reason = raw_finish
        delta = first_choice.get("delta") if first_choice is not None else None
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("role"), str) and delta["role"]:
            role = str(delta["role"])
        for field_name in thinking_fields:
            thinking_piece = delta.get(field_name)
            if isinstance(thinking_piece, str) and thinking_piece:
                ttft_ms = ttft_ms if ttft_ms is not None else round((time.perf_counter() - t0) * 1000, 3)
                thinking_parts[field_name].append(thinking_piece)
                yield ThinkingDelta(text=thinking_piece)
        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            ttft_ms = ttft_ms if ttft_ms is not None else round((time.perf_counter() - t0) * 1000, 3)
            parts.append(piece)
            yield TextDelta(text=piece)
        raw_tool_calls = delta.get("tool_calls") or ()
        if isinstance(raw_tool_calls, (list, tuple)):
            valid_tcs = [tc for tc in raw_tool_calls if isinstance(tc, dict)]
            for pos, tc in enumerate(valid_tcs):
                ttft_ms = ttft_ms if ttft_ms is not None else round((time.perf_counter() - t0) * 1000, 3)
                raw_idx = tc.get("index")
                idx = (
                    raw_idx
                    if (isinstance(raw_idx, int) and not isinstance(raw_idx, bool) and raw_idx >= 0)
                    else pos
                )
                tc_id = tc["id"] if isinstance(tc.get("id"), str) and tc["id"] else ""
                fn = tc.get("function") or {}
                fn_name = (
                    fn["name"]
                    if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"]
                    else ""
                )
                fn_args = (
                    fn["arguments"]
                    if isinstance(fn, dict) and isinstance(fn.get("arguments"), str) and fn["arguments"]
                    else ""
                )
                acc = tool_acc.setdefault(idx, _ToolCallAcc())
                if tc_id:
                    acc.id = tc_id
                if fn_name:
                    acc.name = f"{acc.name}{fn_name}"
                if fn_args:
                    acc.args.append(fn_args)

    full_text = "".join(parts)
    calls = [
        ToolCall(
            id=acc.id or f"call_{slot_idx}",
            name=acc.name,
            arguments="".join(acc.args) or "{}",
        ).to_openai_dict()
        for slot_idx, acc in sorted(tool_acc.items())
    ]
    msg: dict[str, object] = {"role": role, "content": full_text if (full_text or not calls) else None}
    if calls:
        msg["tool_calls"] = calls
    for field_name, chunks in thinking_parts.items():
        if chunks:
            msg[field_name] = "".join(chunks)
    yield StreamDone(
        message=msg,
        usage=usage,
        ttft_ms=ttft_ms,
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        finish_reason=finish_reason,
    )


# 打开 OpenAI 兼容 SSE 流；首包前若明确因 stream_options/include_usage 返回 400 则记录不支持并去参数重试一次
def _open_openai_stream(
    post_stream: PostStream,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    on_stream_options_unsupported: Callable[[], None] | None = None,
) -> Iterator[str]:
    yielded_any = False
    try:
        for line in post_stream(url, payload, headers):
            yielded_any = True
            yield line
    except HTTPError as error:
        formatted_error, error_text = _format_http_error(error)
        if error.code != 400 or yielded_any or "stream_options" not in payload or not _is_stream_options_incompat(error_text):
            raise formatted_error from error
        if on_stream_options_unsupported is not None:
            on_stream_options_unsupported()
        fallback_payload = {k: v for k, v in payload.items() if k != "stream_options"}
        try:
            yield from post_stream(url, fallback_payload, headers)
        except HTTPError as retry_error:
            formatted_retry, _ = _format_http_error(retry_error)
            raise formatted_retry from retry_error


# 大模型客户端抽象基类，供上层 Agent 循环或独立脚本调用
class LLMClient(ABC):
    last_usage: TokenUsage | None = None

    # 发送对话历史与工具定义，返回模型本轮的 assistant 消息（含纯文本或 tool_calls）
    @abstractmethod
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]: ...

    # 流式发送对话历史与工具定义并吐出 TextDelta / StreamDone 事件；未重写时自动包装同步 respond()
    def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> Iterator[LLMStreamEvent]:
        t0 = time.perf_counter()
        msg = self.respond(messages, tools)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        usage = self.last_usage if isinstance(self.last_usage, TokenUsage) else None
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        ttft_ms: float | None = None
        if isinstance(content, str) and content:
            ttft_ms = elapsed_ms
            yield TextDelta(text=content)
        elif tool_calls:
            ttft_ms = elapsed_ms
        yield StreamDone(message=msg, usage=usage, ttft_ms=ttft_ms, latency_ms=elapsed_ms)


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


# 通用 OpenAI Chat Completions 传输底座（不含 temperature 等方言字段，供标准版与插件扩展复用）
@dataclass
class OpenAIChatClient(LLMClient):
    api_key: str
    base_url: str
    model: str
    post_json: PostJson = field(default=default_post_json, kw_only=True)
    post_stream: PostStream = field(default=default_post_stream, kw_only=True)
    _usage_local: threading.local = field(default_factory=threading.local, init=False, repr=False)
    _stream_usage_supported: bool = field(default=True, init=False, repr=False)

    # 读取当前线程最近一次调用记录的 Token 统计（避免多线程共享客户端串台）
    @property
    def last_usage(self) -> TokenUsage | None:
        return getattr(self._usage_local, "value", None)

    # 设置当前线程最近一次调用的 Token 统计
    @last_usage.setter
    def last_usage(self, value: TokenUsage | None) -> None:
        self._usage_local.value = value

    # 记录当前端点不支持 stream_options，后续流式调用直接跳过该字段
    def _disable_stream_usage(self) -> None:
        self._stream_usage_supported = False

    # 预处理发往接口的消息列表（默认返回浅拷贝列表，子类可按协议边界重写）
    def _prepare_messages(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        return list(messages)

    # 返回附加到请求体顶层的协议参数（如 temperature，默认为空）
    def _extra_payload(self) -> dict[str, object]:
        return {}

    # 返回流式解析时需实时吐出 ThinkingDelta 并按字符串聚合的思维链字段名（默认为空）
    def _thinking_delta_fields(self) -> tuple[str, ...]:
        return ()

    # 规范化同步与流式响应产出的 assistant 消息字典（子类可重写）
    def _normalize_response_message(self, message: dict[str, object]) -> dict[str, object]:
        return message

    # 组装 /chat/completions 的请求地址、请求体与鉴权头
    def _build_request(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        *,
        stream: bool,
    ) -> tuple[str, dict[str, object], dict[str, str]]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            **self._extra_payload(),
            "stream": stream,
        }
        if stream and self._stream_usage_supported:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return (
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )

    # 组装 OpenAI 格式请求体调用 /chat/completions，校验响应结构、记录 usage 并返回首条 assistant 消息
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        url, payload, headers = self._build_request(messages, tools, stream=False)
        self.last_usage = None
        response = self.post_json(url, payload, headers)
        if not isinstance(response, dict):
            raise RuntimeError(f"模型接口返回格式错误：期望 JSON 对象，实际为 {type(response).__name__}")
        if response.get("error") is not None:
            raise RuntimeError(f"模型接口返回错误：{response['error']}")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("模型接口返回缺少非空 choices 列表。")
        first = choices[0]
        raw_message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(raw_message, dict):
            raise RuntimeError("模型接口返回 choices[0] 缺少有效的 message 对象。")
        msg = self._normalize_response_message(dict(raw_message))
        self.last_usage = extract_usage(response)
        return msg

    # 通过 SSE 流式调用 /chat/completions（兼容不支持 stream_options 返回 400 的端点）
    def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> Iterator[LLMStreamEvent]:
        url, payload, headers = self._build_request(messages, tools, stream=True)
        self.last_usage = None
        t0 = time.perf_counter()
        lines = _open_openai_stream(
            self.post_stream,
            url,
            payload,
            headers,
            on_stream_options_unsupported=self._disable_stream_usage,
        )
        for event in _consume_openai_sse(lines, t0, thinking_delta_fields=self._thinking_delta_fields()):
            if isinstance(event, StreamDone):
                event = StreamDone(
                    message=self._normalize_response_message(event.message),
                    usage=event.usage,
                    ttft_ms=event.ttft_ms,
                    latency_ms=event.latency_ms,
                    finish_reason=event.finish_reason,
                )
                self.last_usage = event.usage
            yield event


# OpenAI 兼容 Chat Completions 客户端（支持 DeepSeek / OpenAI / 通义千问等）
@dataclass
class OpenAICompatibleLLM(OpenAIChatClient):
    temperature: float = 0.2

    # 附带标准对话模型所需的 temperature 采样参数
    def _extra_payload(self) -> dict[str, object]:
        return {"temperature": self.temperature}


# 默认内建 OpenAI 兼容 Provider 构造器：直接读取已实化的 ModelProfile 字段
def _build_openai_compatible(profile: ModelProfile) -> LLMClient:
    return OpenAICompatibleLLM(
        api_key=profile.api_key,
        base_url=profile.base_url,
        model=profile.model,
        temperature=profile.temperature,
    )


_PROVIDERS: dict[str, ProviderFactory] = {
    "openai": _build_openai_compatible,
    "openai_compatible": _build_openai_compatible,
}


# 注册自定义大模型厂商 Provider 工厂（支持外部插件显式扩展非 OpenAI 原生协议）
def register_provider(name: str, factory: ProviderFactory) -> None:
    key = name.strip().lower()
    if not key or not callable(factory):
        raise ValueError(f"非法 Provider 注册参数：{name!r}")
    _PROVIDERS[key] = factory


# 返回当前所有全局已注册的 Provider 名称列表
def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


# 根据单个 ModelProfile 直通构造对应的 LLMClient（优先查局部 providers，再查全局 _PROVIDERS）
def from_model_profile(
    profile: ModelProfile,
    *,
    env: Mapping[str, str] | None = None,
    providers: Mapping[str, ProviderFactory] | None = None,
) -> LLMClient:
    api_key = profile.resolve_api_key(env)
    if not api_key:
        return NoopLLMClient()
    provider_key = profile.provider.strip().lower() or "openai_compatible"
    factory = (providers.get(provider_key) if providers is not None else None) or _PROVIDERS.get(provider_key)
    if factory is None:
        all_names = sorted(set(_PROVIDERS) | (set(providers) if providers is not None else set()))
        raise ValueError(
            f"模型 {profile.name!r} 使用了未知 provider={provider_key!r}，可用：{', '.join(all_names)}"
        )
    resolved_profile = replace(profile, provider=provider_key, api_key=api_key)
    return factory(resolved_profile)


# 从环境变量提取字段构造 ModelProfile 后直接委派给 from_model_profile（未开启则返回 NoopLLMClient）
def from_env(
    env: Mapping[str, str] | None = None,
    *,
    providers: Mapping[str, ProviderFactory] | None = None,
) -> LLMClient:
    source = os.environ if env is None else env
    mode = source.get("LLM_MODE", "off").strip().lower() or "off"
    if mode != "on":
        return NoopLLMClient()
    raw_temp = source.get("LLM_TEMPERATURE", "").strip()
    temp = float(raw_temp) if raw_temp else 0.2
    profile = ModelProfile(
        name="env",
        provider=source.get("LLM_PROVIDER", "openai_compatible").strip().lower() or "openai_compatible",
        base_url=source.get("LLM_BASE_URL", "").strip() or "https://api.deepseek.com/v1",
        model=source.get("LLM_MODEL", "").strip() or "deepseek-chat",
        api_key=source.get("LLM_API_KEY", "").strip(),
        temperature=temp,
    )
    return from_model_profile(profile, env=source, providers=providers)
