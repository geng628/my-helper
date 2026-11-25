"""Layer 1 零依赖大模型客户端与 Provider 注册中心（对标 pi-ai）：不感知任何 Agent 或业务模块。"""
from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from personal_assistant.ai.types import (
    LLMStreamEvent,
    ModelProfile,
    ModelsConfig,
    StreamDone,
    TextDelta,
    TokenUsage,
)


# 当未配置 LLM_MODE=on 或缺少 API Key 时抛出的异常
class LLMNotConfigured(RuntimeError):
    """未配置大模型参数时抛出。"""


PostJson = Callable[[str, dict[str, object], dict[str, str]], dict[str, object]]
PostStream = Callable[[str, dict[str, object], dict[str, str]], Iterator[str]]
ProviderFactory = Callable[[Mapping[str, str]], "LLMClient"]


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


# 构造带退避重试的 POST HTTP 响应打开器（仅在连接建立前重试，避免流式中途重复发送）
def _open_with_retry(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "my-helper-v2/0.1",
        **headers,
    }
    for attempt in range(max_retries + 1):
        request = Request(url, data=body, headers=req_headers, method="POST")
        try:
            return urlopen(request, timeout=60)
        except HTTPError as error:
            # 4xx 客户端错误（429 限流除外）不重试，直接抛出
            if error.code < 500 and error.code != 429:
                raise
            if attempt >= max_retries:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt >= max_retries:
                raise
        time.sleep(0.5 * (2**attempt))
    raise RuntimeError("请求重试耗尽。")


# 使用 Python 标准库 urllib 向指定接口发送 JSON POST 请求，并对瞬时网络/TLS 抖动自动退避重试
def default_post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
) -> dict[str, object]:
    with _open_with_retry(url, payload, headers, max_retries=max_retries) as response:
        return json.loads(response.read().decode("utf-8"))


# 使用 Python 标准库 urllib 向指定接口发起 SSE 流式 POST 请求并逐行吐出 UTF-8 文本
def default_post_stream(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
) -> Iterator[str]:
    with _open_with_retry(url, payload, headers, max_retries=max_retries) as response:
        for raw_line in response:
            yield raw_line.decode("utf-8")


# 逐行解析 OpenAI 兼容 SSE 流：实时吐出 TextDelta、拼接分片 tool_calls 并按需聚合扩展文本增量字段
def _consume_openai_sse(
    lines: Iterable[str],
    t0: float,
    *,
    extra_delta_fields: Iterable[str] = (),
) -> Iterator[LLMStreamEvent]:
    extra_fields = tuple(extra_delta_fields)
    extra_parts: dict[str, list[str]] = {name: [] for name in extra_fields}
    role, parts, tool_acc = "assistant", [], {}
    usage: TokenUsage | None = None
    ttft_ms: float | None = None

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
        delta = choices[0].get("delta") if choices and isinstance(choices[0], dict) else None
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("role"), str) and delta["role"]:
            role = str(delta["role"])
        for field_name in extra_fields:
            extra_piece = delta.get(field_name)
            if isinstance(extra_piece, str) and extra_piece:
                ttft_ms = ttft_ms if ttft_ms is not None else round((time.perf_counter() - t0) * 1000, 3)
                extra_parts[field_name].append(extra_piece)
        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            ttft_ms = ttft_ms if ttft_ms is not None else round((time.perf_counter() - t0) * 1000, 3)
            parts.append(piece)
            yield TextDelta(text=piece)
        for tc in delta.get("tool_calls") or ():
            if not isinstance(tc, dict):
                continue
            ttft_ms = ttft_ms if ttft_ms is not None else round((time.perf_counter() - t0) * 1000, 3)
            raw_idx = tc.get("index")
            idx = raw_idx if isinstance(raw_idx, int) and not isinstance(raw_idx, bool) else 0
            acc = tool_acc.setdefault(idx, {"id": "", "type": "function", "name": "", "args": []})
            if isinstance(tc.get("id"), str) and tc["id"]:
                acc["id"] = tc["id"]
            if isinstance(tc.get("type"), str) and tc["type"]:
                acc["type"] = tc["type"]
            fn = tc.get("function") or {}
            if isinstance(fn, dict):
                if isinstance(fn.get("name"), str) and fn["name"]:
                    acc["name"] = f"{acc['name']}{fn['name']}"
                if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                    acc["args"].append(fn["arguments"])  # type: ignore[attr-defined]

    full_text = "".join(parts)
    calls = [
        {"id": a["id"], "type": a["type"], "function": {"name": a["name"], "arguments": "".join(a["args"])}}
        for _, a in sorted(tool_acc.items())
    ]
    msg: dict[str, object] = {"role": role, "content": full_text if (full_text or not calls) else None}
    if calls:
        msg["tool_calls"] = calls
    for field_name, chunks in extra_parts.items():
        if chunks:
            msg[field_name] = "".join(chunks)
    yield StreamDone(message=msg, usage=usage, ttft_ms=ttft_ms, latency_ms=round((time.perf_counter() - t0) * 1000, 3))


# 打开 OpenAI 兼容 SSE 流；若首包前因 stream_options 返回 HTTP 400 则自动去掉该参数重试一次
def _open_openai_stream(
    post_stream: PostStream,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
) -> Iterator[str]:
    yielded_any = False
    try:
        for line in post_stream(url, payload, headers):
            yielded_any = True
            yield line
    except HTTPError as error:
        if error.code != 400 or yielded_any or "stream_options" not in payload:
            raise
        fallback_payload = {k: v for k, v in payload.items() if k != "stream_options"}
        yield from post_stream(url, fallback_payload, headers)


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
        raw_usage = getattr(self, "last_usage", None)
        usage = raw_usage if isinstance(raw_usage, TokenUsage) else None
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        ttft_ms: float | None = None
        if isinstance(content, str) and content and not tool_calls:
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

    # 读取当前线程最近一次调用记录的 Token 统计（避免多线程共享客户端串台）
    @property
    def last_usage(self) -> TokenUsage | None:
        return getattr(self._usage_local, "value", None)

    # 设置当前线程最近一次调用的 Token 统计
    @last_usage.setter
    def last_usage(self, value: TokenUsage | None) -> None:
        self._usage_local.value = value

    # 预处理发往接口的消息列表（子类可按协议边界重写，默认原样透传）
    def _prepare_messages(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        return messages

    # 返回附加到请求体顶层的协议参数（如 temperature，默认为空）
    def _extra_payload(self) -> dict[str, object]:
        return {}

    # 返回流式解析时需按字符串聚合的扩展增量字段名（默认为空）
    def _stream_delta_fields(self) -> tuple[str, ...]:
        return ()

    # 规范化同步与流式响应产出的 assistant 消息字典（子类可重写）
    def _normalize_response_message(self, message: dict[str, object]) -> dict[str, object]:
        return message

    # 组装 /chat/completions 的请求地址、请求体与鉴权头（同步与流式调用共用）
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
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return (
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )

    # 组装 OpenAI 格式请求体调用 /chat/completions，记录 usage 并返回首条 assistant 消息
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        url, payload, headers = self._build_request(messages, tools, stream=False)
        self.last_usage = None
        response = self.post_json(url, payload, headers)
        msg = self._normalize_response_message(dict(response["choices"][0]["message"]))
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
        lines = _open_openai_stream(self.post_stream, url, payload, headers)
        for event in _consume_openai_sse(lines, t0, extra_delta_fields=self._stream_delta_fields()):
            if isinstance(event, StreamDone):
                event = StreamDone(
                    message=self._normalize_response_message(event.message),
                    usage=event.usage,
                    ttft_ms=event.ttft_ms,
                    latency_ms=event.latency_ms,
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


# 默认内建 OpenAI 兼容 Provider 构造器
def _build_openai_compatible(env: Mapping[str, str]) -> LLMClient:
    raw_temp = env.get("LLM_TEMPERATURE", "").strip()
    temp = float(raw_temp) if raw_temp else 0.2
    return OpenAICompatibleLLM(
        api_key=env.get("LLM_API_KEY", "").strip(),
        base_url=env.get("LLM_BASE_URL", "").strip() or "https://api.deepseek.com/v1",
        model=env.get("LLM_MODEL", "").strip() or "deepseek-chat",
        temperature=temp,
    )


_PROVIDERS: dict[str, ProviderFactory] = {
    "openai": _build_openai_compatible,
    "openai_compatible": _build_openai_compatible,
}


# 注册自定义大模型厂商 Provider 工厂（支持外部插件扩展非 OpenAI 原生协议，不改动核心循环）
def register_provider(name: str, factory: ProviderFactory) -> None:
    key = name.strip().lower()
    if not key or not callable(factory):
        raise ValueError(f"非法 Provider 注册参数：{name!r}")
    _PROVIDERS[key] = factory


# 返回当前所有已注册的 Provider 名称列表
def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


# 从环境变量读取配置并按 LLM_PROVIDER 分发构造 LLMClient（未开启则返回 NoopLLMClient）
def from_env(env: Mapping[str, str] | None = None) -> LLMClient:
    source = os.environ if env is None else env
    mode = source.get("LLM_MODE", "off").strip().lower() or "off"
    api_key = source.get("LLM_API_KEY", "").strip()
    if mode != "on" or not api_key:
        return NoopLLMClient()
    provider = source.get("LLM_PROVIDER", "openai_compatible").strip().lower() or "openai_compatible"
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(
            f"未知 LLM_PROVIDER={provider!r}，可用 Provider：{', '.join(available_providers())}"
        )
    return factory(source)


# 根据单个 ModelProfile 构造对应的 LLMClient（统一复用已注册 Provider，支持直写密钥免配环境变量）
def from_model_profile(
    profile: ModelProfile,
    *,
    env: Mapping[str, str] | None = None,
) -> LLMClient:
    api_key = profile.resolve_api_key(env)
    if not api_key:
        return NoopLLMClient()
    factory = _PROVIDERS.get(profile.provider)
    if factory is None:
        raise ValueError(
            f"模型 {profile.name!r} 使用了未知 provider={profile.provider!r}，可用：{', '.join(available_providers())}"
        )
    merged: dict[str, str] = dict(os.environ if env is None else env)
    merged.update(
        {
            "LLM_MODE": "on",
            "LLM_PROVIDER": profile.provider,
            "LLM_API_KEY": api_key,
            "LLM_BASE_URL": profile.base_url,
            "LLM_MODEL": profile.model,
            "LLM_TEMPERATURE": str(profile.temperature),
        }
    )
    return factory(merged)


# 从标准 models.json 字典与可选模型名直接构造 LLMClient
def from_models_dict(
    raw: dict[str, object],
    *,
    profile_name: str | None = None,
    env: Mapping[str, str] | None = None,
) -> LLMClient:
    cfg = ModelsConfig.from_dict(raw)
    return from_model_profile(cfg.get_profile(profile_name), env=env)
