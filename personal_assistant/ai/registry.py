"""Layer 1 统一前台入口与翻译器通讯录注册表（Pi ch04 第一层）：仅依赖 protocol.py，零感知具体翻译器实现。"""
from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace

from personal_assistant.ai.protocol import (
    THINKING_BUDGETS,
    ApiProvider,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    LLMNotConfigured,
    ModelInvoker,
    ModelProfile,
    StreamDone,
    StreamError,
    StreamOptions,
    TokenUsage,
)


# 确定性关闭支持 close() 的流式迭代器，防止异常或早退泄漏底层 HTTP 连接
def _close_iterator(iterator: Iterator[AssistantMessageEvent]) -> None:
    close_fn = getattr(iterator, "close", None)
    if callable(close_fn):
        close_fn()


# 线程安全的协议翻译器通讯录注册表（按小写归一化 api 路由分发）
class ApiProviderRegistry:
    # 初始化可重入互斥锁与翻译器映射表
    def __init__(self, initial: Mapping[str, ApiProvider] | None = None) -> None:
        self._lock = threading.RLock()
        self._providers: dict[str, ApiProvider] = {}
        if initial is not None:
            for provider in initial.values():
                self.register(provider)

    # 按 provider.api 小写归一化登记翻译器实例
    def register(self, provider: ApiProvider) -> None:
        if not isinstance(provider, ApiProvider):
            raise TypeError(f"翻译器必须实现 ApiProvider 协议：{provider!r}")
        key = str(provider.api or "").strip().lower()
        if not key:
            raise ValueError("翻译器的 api 标识不能为空。")
        with self._lock:
            self._providers[key] = provider

    # 按 api 标识查找对应翻译器；未注册时抛出附带可用协议列表的 ValueError
    def resolve(self, api: str) -> ApiProvider:
        key = str(api or "").strip().lower()
        with self._lock:
            provider = self._providers.get(key)
            if provider is not None:
                return provider
            available = ", ".join(sorted(self._providers)) or "（无）"
        raise ValueError(f"未知的 API 协议 {api!r}，当前已注册：{available}")

    # 返回当前已注册的全部 api 协议名称排序列表
    def available(self) -> list[str]:
        with self._lock:
            return sorted(self._providers)

    # 返回当前通讯录的独立浅拷贝，用于按应用或插件隔离扩展翻译器
    def copy(self) -> "ApiProviderRegistry":
        with self._lock:
            copied = ApiProviderRegistry()
            copied._providers = dict(self._providers)
            return copied

    # 导出当前通讯录字典快照（供测试隔离保存现场）
    def snapshot(self) -> dict[str, ApiProvider]:
        with self._lock:
            return dict(self._providers)

    # 用指定快照原子恢复当前通讯录内容（供测试隔离还原现场）
    def restore(self, items: Mapping[str, ApiProvider]) -> None:
        with self._lock:
            self._providers.clear()
            self._providers.update(items)


_DEFAULT_REGISTRY = ApiProviderRegistry()


# 返回进程级默认翻译器通讯录注册表
def default_registry() -> ApiProviderRegistry:
    return _DEFAULT_REGISTRY


# 向默认全局注册表登记一个 ApiProvider 翻译器
def register_api_provider(provider: ApiProvider) -> None:
    _DEFAULT_REGISTRY.register(provider)


# 从默认全局注册表解析指定 api 协议对应的翻译器
def resolve_api_provider(api: str) -> ApiProvider:
    return _DEFAULT_REGISTRY.resolve(api)


# 列出默认全局注册表中所有可用 api 协议名称
def available_api_providers() -> list[str]:
    return _DEFAULT_REGISTRY.available()


# 统一流式前台入口：同步校验密钥并按 model.api 分派给注册表中的对应翻译器
def stream(
    model: ModelProfile,
    context: Context,
    options: StreamOptions | None = None,
    *,
    registry: ApiProviderRegistry | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[AssistantMessageEvent]:
    api_key = model.resolve_api_key(env)
    if not api_key:
        raise LLMNotConfigured(
            f"模型 {model.name!r} 未配置 API Key。请配置 models.dsl / models.json 或设置 LLM_MODE=on 与 LLM_API_KEY。"
        )
    active_registry = registry if registry is not None else _DEFAULT_REGISTRY
    normalized_api = str(model.api or "").strip().lower()
    translator = active_registry.resolve(normalized_api)
    resolved_model = replace(model, api=normalized_api, api_key=api_key)
    return translator.stream(resolved_model, context, options)


# 合并 ModelProfile 默认配置与显式 StreamOptions（开启思考且未指定 max_tokens 时自动补足上限）
def _merge_stream_options(
    model: ModelProfile,
    options: StreamOptions | None,
) -> StreamOptions:
    if options is None:
        temperature: float | None = model.temperature
        max_tokens = model.max_tokens
        reasoning = model.reasoning
        cache_retention = model.cache_retention
        session_id: str | None = None
    else:
        temperature = options.temperature if options.temperature is not None else model.temperature
        max_tokens = options.max_tokens if options.max_tokens is not None else model.max_tokens
        reasoning = options.reasoning if options.reasoning != "off" else model.reasoning
        cache_retention = options.cache_retention if options.cache_retention != "short" else model.cache_retention
        session_id = options.session_id

    if reasoning != "off" and max_tokens is None:
        max_tokens = THINKING_BUDGETS[reasoning] + 4096

    return StreamOptions(
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
        cache_retention=cache_retention,
        session_id=session_id,
    )


# 简化流式前台入口：自动合并模型默认参数与显式调用选项后委派给 stream()
def stream_simple(
    model: ModelProfile,
    context: Context,
    options: StreamOptions | None = None,
    *,
    registry: ApiProviderRegistry | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[AssistantMessageEvent]:
    merged = _merge_stream_options(model, options)
    return stream(model, context, merged, registry=registry, env=env)


# 同步完成前台入口：确定性消费 stream_simple() 事件流，成功返回 AssistantMessage，遇 StreamError 抛 RuntimeError
def complete(
    model: ModelProfile,
    context: Context,
    options: StreamOptions | None = None,
    *,
    registry: ApiProviderRegistry | None = None,
    env: Mapping[str, str] | None = None,
) -> AssistantMessage:
    event_iter = stream_simple(model, context, options, registry=registry, env=env)
    try:
        for event in event_iter:
            if isinstance(event, StreamError):
                raise RuntimeError(event.error_message)
            if isinstance(event, StreamDone):
                return event.message
    finally:
        _close_iterator(event_iter)
    raise RuntimeError("模型流式响应异常终止，未收到 StreamDone 事件。")


# 轻量模型绑定句柄：持有 ModelProfile 与可选局部注册表，线程安全记录 last_usage 并实现 ModelInvoker
@dataclass
class ModelClient:
    profile: ModelProfile
    registry: ApiProviderRegistry | None = None
    env: Mapping[str, str] | None = None
    _usage_local: threading.local = field(default_factory=threading.local, init=False, repr=False)

    # 返回底层模型 ID，便于评测报告与日志观测
    @property
    def model(self) -> str:
        return self.profile.model

    # 返回底层路由协议标识 api
    @property
    def api(self) -> str:
        return self.profile.api

    # 读取当前线程最近一次成功调用的 Token 用量统计（多线程共享实例零串台）
    @property
    def last_usage(self) -> TokenUsage | None:
        value = getattr(self._usage_local, "value", None)
        return value if isinstance(value, TokenUsage) else None

    # 设置当前线程最近一次调用的 Token 用量统计
    @last_usage.setter
    def last_usage(self, value: TokenUsage | None) -> None:
        self._usage_local.value = value

    # 流式调用模型：同步完成密钥与协议校验，流结束时自动更新当前线程 last_usage
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        self.last_usage = None
        raw_iter = stream_simple(
            self.profile,
            context,
            options,
            registry=self.registry,
            env=self.env,
        )

        # 包装底层事件流，在 StreamDone/StreamError 时更新线程局部 last_usage 并确定性释放连接
        def _tracked() -> Iterator[AssistantMessageEvent]:
            try:
                for event in raw_iter:
                    if isinstance(event, StreamDone):
                        self.last_usage = event.usage
                    elif isinstance(event, StreamError):
                        self.last_usage = None
                    yield event
            finally:
                _close_iterator(raw_iter)

        return _tracked()

    # 同步调用模型：确定性消费 self.stream() 并返回完整 AssistantMessage
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        event_iter = self.stream(context, options)
        try:
            for event in event_iter:
                if isinstance(event, StreamError):
                    raise RuntimeError(event.error_message)
                if isinstance(event, StreamDone):
                    return event.message
        finally:
            _close_iterator(event_iter)
        raise RuntimeError("模型流式响应异常终止，未收到 StreamDone 事件。")


# 占位未配置句柄：实现 ModelInvoker，调用即同步抛出 LLMNotConfigured
class NoopModelClient:
    last_usage: TokenUsage | None = None

    # 立即抛出未配置大模型异常
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        raise LLMNotConfigured(
            "LLM 未启用。请在 models.dsl / models.json 中配置模型，或在环境变量/.env 中设置 LLM_MODE=on 和 LLM_API_KEY。"
        )

    # 立即抛出未配置大模型异常
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        raise LLMNotConfigured(
            "LLM 未启用。请在 models.dsl / models.json 中配置模型，或在环境变量/.env 中设置 LLM_MODE=on 和 LLM_API_KEY。"
        )


# 从单个 ModelProfile 构造绑定后的 ModelInvoker（缺密钥返回 NoopModelClient，有密钥则校验已注册 api）
def from_model_profile(
    profile: ModelProfile,
    *,
    env: Mapping[str, str] | None = None,
    registry: ApiProviderRegistry | None = None,
) -> ModelInvoker:
    api_key = profile.resolve_api_key(env)
    if not api_key:
        return NoopModelClient()
    active_registry = registry if registry is not None else _DEFAULT_REGISTRY
    normalized_api = str(profile.api or "").strip().lower()
    active_registry.resolve(normalized_api)
    resolved_profile = replace(profile, api=normalized_api, api_key=api_key)
    return ModelClient(profile=resolved_profile, registry=active_registry, env=env)


# 从环境变量提取参数并通过 ModelProfile.from_dict 单点校验构造 ModelInvoker（默认 api="openai-completions"）
def from_env(
    env: Mapping[str, str] | None = None,
    *,
    registry: ApiProviderRegistry | None = None,
) -> ModelInvoker:
    source = os.environ if env is None else env
    mode = str(source.get("LLM_MODE", "off") or "off").strip().lower() or "off"
    if mode != "on":
        return NoopModelClient()
    raw: dict[str, object] = {
        "api": str(source.get("LLM_API", "") or "").strip() or "openai-completions",
        "base_url": str(source.get("LLM_BASE_URL", "") or "").strip() or "https://api.deepseek.com/v1",
        "model": str(source.get("LLM_MODEL", "") or "").strip() or "deepseek-chat",
        "api_key": str(source.get("LLM_API_KEY", "") or "").strip(),
    }
    raw_temp = str(source.get("LLM_TEMPERATURE", "") or "").strip()
    if raw_temp:
        raw["temperature"] = raw_temp
    raw_max_tokens = str(source.get("LLM_MAX_TOKENS", "") or "").strip()
    if raw_max_tokens:
        raw["max_tokens"] = raw_max_tokens
    raw_reasoning = str(source.get("LLM_REASONING", "") or "").strip()
    if raw_reasoning:
        raw["reasoning"] = raw_reasoning
    raw_cache = str(source.get("LLM_CACHE_RETENTION", "") or "").strip()
    if raw_cache:
        raw["cache_retention"] = raw_cache

    profile = ModelProfile.from_dict("env", raw)
    return from_model_profile(profile, env=source, registry=registry)
