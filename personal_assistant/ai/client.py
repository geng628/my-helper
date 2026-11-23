"""Layer 1 零依赖大模型客户端与 Provider 注册中心（对标 pi-ai）：不感知任何 Agent 或业务模块。"""
from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from personal_assistant.ai.types import ModelProfile, ModelsConfig, TokenUsage


# 当未配置 LLM_MODE=on 或缺少 API Key 时抛出的异常
class LLMNotConfigured(RuntimeError):
    """未配置大模型参数时抛出。"""


PostJson = Callable[[str, dict[str, object], dict[str, str]], dict[str, object]]
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


# 使用 Python 标准库 urllib 向指定接口发送 JSON POST 请求，并对瞬时网络/TLS 抖动自动退避重试
def default_post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
) -> dict[str, object]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "my-helper-v2/0.1",
        **headers,
    }
    for attempt in range(max_retries + 1):
        request = Request(url, data=body, headers=req_headers, method="POST")
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
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
@dataclass
class OpenAICompatibleLLM(LLMClient):
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    post_json: PostJson = default_post_json
    _usage_local: threading.local = field(default_factory=threading.local, init=False, repr=False)

    # 读取当前线程最近一次调用记录的 Token 统计（避免多线程共享客户端串台）
    @property
    def last_usage(self) -> TokenUsage | None:
        return getattr(self._usage_local, "value", None)

    # 设置当前线程最近一次调用的 Token 统计
    @last_usage.setter
    def last_usage(self, value: TokenUsage | None) -> None:
        self._usage_local.value = value

    # 组装 OpenAI 格式请求体调用 /chat/completions，记录 usage 并返回首条 assistant 消息
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

        self.last_usage = None
        response = self.post_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        msg = dict(response["choices"][0]["message"])
        self.last_usage = extract_usage(response)
        return msg


# 默认内建 OpenAI 兼容 Provider 构造器
def _build_openai_compatible(env: Mapping[str, str]) -> LLMClient:
    raw_temp = env.get("LLM_TEMPERATURE", "").strip()
    temp = float(raw_temp) if raw_temp else 0.2
    return OpenAICompatibleLLM(
        api_key=env.get("LLM_API_KEY", "").strip(),
        base_url=env.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        model=env.get("LLM_MODEL", "deepseek-chat"),
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
