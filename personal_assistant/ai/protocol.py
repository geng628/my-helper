"""Layer 1 事件与契约协议层（Pi ch04 第二层）：纯数据契约、12 种统一流式事件、模型档案与翻译器协议，零内部导入。"""
from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
THINKING_LEVELS: tuple[ThinkingLevel, ...] = ("off", "minimal", "low", "medium", "high", "xhigh")

CacheRetention = Literal["none", "short", "long"]
CACHE_RETENTIONS: tuple[CacheRetention, ...] = ("none", "short", "long")

StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]
DoneReason = Literal["stop", "length", "toolUse"]
ErrorReason = Literal["error", "aborted"]

THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    "off": 0,
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
}


# 按 Pi ch04 规则将目标思考级别夹紧到模型支持的级别集合（off 恒返回 off；其余先向上找、找不到再向下找）
def clamp_thinking_level(
    level: ThinkingLevel,
    supported: Sequence[ThinkingLevel],
) -> ThinkingLevel:
    if level not in THINKING_LEVELS:
        raise ValueError(f"非法的思考级别：{level!r}")
    if level == "off":
        return "off"
    supported_set = {item for item in supported if item in THINKING_LEVELS}
    if not supported_set:
        return "off"
    if level in supported_set:
        return level
    idx = THINKING_LEVELS.index(level)
    for candidate in THINKING_LEVELS[idx + 1 :]:
        if candidate in supported_set:
            return candidate
    for candidate in reversed(THINKING_LEVELS[:idx]):
        if candidate in supported_set:
            return candidate
    return "off"


# 单次大模型调用的 Token 消耗与服务端 KV 缓存命中记录
@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    # 计算本次请求的服务端缓存命中率（无输入 token 时返回 None）
    @property
    def cache_hit_rate(self) -> float | None:
        if self.prompt_tokens <= 0:
            return None
        return round(self.cached_tokens / self.prompt_tokens, 4)


# 中立文本内容块
@dataclass(frozen=True)
class TextBlock:
    text: str


# 中立思维链内容块（可选携带 Anthropic / Gemini 思考签名）
@dataclass(frozen=True)
class ThinkingBlock:
    text: str
    signature: str = ""


# 中立工具调用请求块（纯数据结构，不含任何厂商线协议序列化逻辑）
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


AssistantBlock = TextBlock | ThinkingBlock | ToolCall


# 中立用户消息
@dataclass(frozen=True)
class UserMessage:
    content: str


# 中立助手消息（统一以内容块元组表达正文、思维链与工具调用）
@dataclass(frozen=True)
class AssistantMessage:
    blocks: tuple[AssistantBlock, ...] = ()
    stop_reason: StopReason | None = None
    usage: TokenUsage | None = None
    error_message: str = ""

    # 汇总消息内全部 TextBlock 拼接得到的纯文本回复
    @property
    def text(self) -> str:
        return "".join(block.text for block in self.blocks if isinstance(block, TextBlock))

    # 汇总消息内全部 ThinkingBlock 拼接得到的思维链文本
    @property
    def thinking(self) -> str:
        return "".join(block.text for block in self.blocks if isinstance(block, ThinkingBlock))

    # 提取消息内全部 ToolCall 工具调用元组
    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(block for block in self.blocks if isinstance(block, ToolCall))

    # 从纯文本便捷构造 AssistantMessage（空文本生成空 blocks 元组）
    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        stop_reason: StopReason | None = None,
        usage: TokenUsage | None = None,
    ) -> "AssistantMessage":
        blocks: tuple[AssistantBlock, ...] = (TextBlock(text=text),) if text else ()
        return cls(blocks=blocks, stop_reason=stop_reason, usage=usage)


# 中立工具执行结果消息（保留 tool_call_id 与 tool_name，满足 OpenAI/Anthropic/Gemini 全协议翻译需求）
@dataclass(frozen=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False


Message = UserMessage | AssistantMessage | ToolResultMessage


# 底层中立工具签名契约（仅描述工具名、说明、参数与必填约束，不含厂商格式导出与本地执行逻辑）
@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, str]
    param_enums: dict[str, tuple[str, ...]] | None = field(default=None, kw_only=True)
    required: tuple[str, ...] | None = field(default=None, kw_only=True)

    # 返回按参数声明顺序对齐的有效必填参数名元组（供各翻译器与参数校验单点复用）
    @property
    def required_names(self) -> tuple[str, ...]:
        if self.required is None:
            return tuple(self.parameters)
        allowed = set(self.required)
        return tuple(k for k in self.parameters if k in allowed)


# 统一中立推理上下文（独立承载系统提示词、消息轨迹与中立工具签名）
@dataclass(frozen=True)
class Context:
    system_prompt: str = ""
    messages: Sequence[Message] = ()
    tools: Sequence[ToolSchema] = ()


# 统一模型流式调用选项（各字段默认 None 表示沿用 ModelProfile 配置，显式传值即可单次覆写）
@dataclass(frozen=True)
class StreamOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning: ThinkingLevel | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None


# 流式事件 1：流开始
@dataclass(frozen=True)
class StreamStart:
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 2：文本内容块开始
@dataclass(frozen=True)
class TextStart:
    content_index: int
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 3：文本增量片段
@dataclass(frozen=True)
class TextDelta:
    content_index: int
    text: str
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 4：文本内容块结束
@dataclass(frozen=True)
class TextEnd:
    content_index: int
    text: str
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 5：思维链内容块开始
@dataclass(frozen=True)
class ThinkingStart:
    content_index: int
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 6：思维链增量片段
@dataclass(frozen=True)
class ThinkingDelta:
    content_index: int
    text: str
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 7：思维链内容块结束
@dataclass(frozen=True)
class ThinkingEnd:
    content_index: int
    text: str
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 8：工具调用内容块开始
@dataclass(frozen=True)
class ToolCallStart:
    content_index: int
    call_id: str
    name: str
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 9：工具调用参数 JSON 增量片段
@dataclass(frozen=True)
class ToolCallDelta:
    content_index: int
    delta: str
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 10：工具调用内容块结束
@dataclass(frozen=True)
class ToolCallEnd:
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage = field(default_factory=AssistantMessage)


# 流式事件 11：流正常完成（携带终止原因、完整 AssistantMessage、Token 用量与首字/总耗时）
@dataclass(frozen=True)
class StreamDone:
    reason: DoneReason
    message: AssistantMessage
    usage: TokenUsage | None = None
    ttft_ms: float | None = None
    latency_ms: float = 0.0


# 流式事件 12：流异常终止（将传输/HTTP/解析错误收敛为统一事件，不破坏流式协议）
@dataclass(frozen=True)
class StreamError:
    reason: ErrorReason
    message: AssistantMessage
    error_message: str
    latency_ms: float = 0.0


AssistantMessageEvent = (
    StreamStart
    | TextStart
    | TextDelta
    | TextEnd
    | ThinkingStart
    | ThinkingDelta
    | ThinkingEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | StreamDone
    | StreamError
)


# 当未配置有效模型或缺少 API Key 时抛出的异常
class LLMNotConfigured(RuntimeError):
    """未配置大模型参数时抛出。"""


# 解析并校验可选正整数 max_tokens 字段（严格拒绝布尔值、浮点数与非正数）
def _parse_optional_max_tokens(raw_value: object, name: str) -> int | None:
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"模型 {name!r} 的 max_tokens 必须是正整数。")
    if isinstance(raw_value, int):
        parsed = raw_value
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
        except ValueError as error:
            raise ValueError(f"模型 {name!r} 的 max_tokens 必须是正整数。") from error
    else:
        raise ValueError(f"模型 {name!r} 的 max_tokens 必须是正整数。")
    if parsed < 1:
        raise ValueError(f"模型 {name!r} 的 max_tokens 必须大于等于 1。")
    return parsed


# 解析并校验思考级别枚举值
def _parse_thinking_level(raw_value: object, name: str) -> ThinkingLevel:
    if raw_value is None or raw_value == "":
        return "off"
    if not isinstance(raw_value, str):
        raise ValueError(f"模型 {name!r} 的 reasoning 必须是字符串。")
    normalized = raw_value.strip().lower()
    if not normalized:
        return "off"
    if normalized == "off":
        return "off"
    if normalized == "minimal":
        return "minimal"
    if normalized == "low":
        return "low"
    if normalized == "medium":
        return "medium"
    if normalized == "high":
        return "high"
    if normalized == "xhigh":
        return "xhigh"
    allowed = "、".join(THINKING_LEVELS)
    raise ValueError(f"模型 {name!r} 的 reasoning 取值非法：{raw_value!r}（允许：{allowed}）。")


# 解析并校验缓存保留策略枚举值
def _parse_cache_retention(raw_value: object, name: str) -> CacheRetention:
    if raw_value is None or raw_value == "":
        return "short"
    if not isinstance(raw_value, str):
        raise ValueError(f"模型 {name!r} 的 cache_retention 必须是字符串。")
    normalized = raw_value.strip().lower()
    if not normalized:
        return "short"
    if normalized == "none":
        return "none"
    if normalized == "short":
        return "short"
    if normalized == "long":
        return "long"
    allowed = "、".join(CACHE_RETENTIONS)
    raise ValueError(f"模型 {name!r} 的 cache_retention 取值非法：{raw_value!r}（允许：{allowed}）。")


# 单个大模型的标准配置档案（单一路由键 api，零旧 provider 兼容壳）
@dataclass(frozen=True)
class ModelProfile:
    name: str
    api: str = "openai-completions"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    key_env: str = ""
    temperature: float = 0.2
    max_tokens: int | None = None
    reasoning: ThinkingLevel = "off"
    cache_retention: CacheRetention = "short"

    # 从标准 JSON 字典构造并严格校验 ModelProfile（拒绝旧 provider 字段、非法数值与越界参数）
    @classmethod
    def from_dict(cls, name: str, raw: dict[str, object]) -> "ModelProfile":
        if not isinstance(raw, dict):
            raise ValueError(f"模型 {name!r} 的配置必须是 JSON 对象。")
        if "provider" in raw:
            raise ValueError(f"模型 {name!r} 使用了已废弃字段 'provider'，请改用 'api'。")
        raw_api = raw.get("api")
        if raw_api is None:
            api = "openai-completions"
        elif isinstance(raw_api, str) and raw_api.strip():
            api = raw_api.strip().lower()
        else:
            raise ValueError(f"模型 {name!r} 的 api 必须是非空字符串。")

        base_url = str(raw.get("base_url") or "").strip()
        model = str(raw.get("model") or "").strip()
        api_key = str(raw.get("api_key") or "").strip()
        key_env = str(raw.get("key_env") or "").strip()
        raw_temp = raw.get("temperature", 0.2)
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError(f"模型 {name!r} 的 base_url 必须以 http:// 或 https:// 开头。")
        if not model:
            raise ValueError(f"模型 {name!r} 缺少非空 model 名称。")
        if isinstance(raw_temp, bool) or not isinstance(raw_temp, (int, float, str)):
            raise ValueError(f"模型 {name!r} 的 temperature 必须是有限数字。")
        try:
            temp = float(raw_temp)
        except ValueError as error:
            raise ValueError(f"模型 {name!r} 的 temperature 必须是有限数字。") from error
        if not math.isfinite(temp) or not (0.0 <= temp <= 2.0):
            raise ValueError(f"模型 {name!r} 的 temperature 必须在 [0, 2] 之间且为有限数。")

        max_tokens = _parse_optional_max_tokens(raw.get("max_tokens"), name)
        reasoning = _parse_thinking_level(raw.get("reasoning"), name)
        cache_retention = _parse_cache_retention(raw.get("cache_retention"), name)
        return cls(
            name=name,
            api=api,
            base_url=base_url,
            model=model,
            api_key=api_key,
            key_env=key_env,
            temperature=temp,
            max_tokens=max_tokens,
            reasoning=reasoning,
            cache_retention=cache_retention,
        )

    # 解析实际可用的 API Key（优先使用直写的非空 api_key，其次读取 key_env 对应的环境变量）
    def resolve_api_key(self, env: Mapping[str, str] | None = None) -> str:
        inline_key = self.api_key.strip()
        if inline_key:
            return inline_key
        if self.key_env:
            source = os.environ if env is None else env
            return str(source.get(self.key_env, "")).strip()
        return ""

    # 输出完全脱敏的展示字典，严禁暴露任何明文 api_key
    def redacted_dict(self, env: Mapping[str, str] | None = None) -> dict[str, object]:
        resolved = self.resolve_api_key(env)
        if self.api_key.strip():
            key_source = "inline"
        elif self.key_env:
            key_source = f"env:{self.key_env}"
        else:
            key_source = "none"
        return {
            "api": self.api,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning,
            "cache_retention": self.cache_retention,
            "api_key_configured": bool(resolved),
            "key_source": key_source,
        }


# 多模型注册表配置对象（对应标准 models.json 顶层结构，支持通用命名角色映射 roles）
@dataclass(frozen=True)
class ModelsConfig:
    default: str
    models: dict[str, ModelProfile]
    roles: dict[str, str] = field(default_factory=dict)

    # 从标准 JSON 字典构造 ModelsConfig 并校验默认模型与全部命名角色目标模型存在
    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "ModelsConfig":
        if not isinstance(raw, dict):
            raise ValueError("模型配置顶层必须是 JSON 对象。")
        raw_models = raw.get("models")
        if not isinstance(raw_models, dict) or not raw_models:
            raise ValueError("模型配置必须包含非空的 models 映射。")
        parsed: dict[str, ModelProfile] = {}
        for raw_name, entry in raw_models.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("模型名称不能为空。")
            if not isinstance(entry, dict):
                raise ValueError(f"模型 {name!r} 的配置必须是 JSON 对象。")
            parsed[name] = ModelProfile.from_dict(name, entry)
        default_name = str(raw.get("default") or "").strip()
        if not default_name:
            if len(parsed) == 1:
                default_name = next(iter(parsed))
            else:
                raise ValueError("配置多个模型时必须指定 default 默认模型。")
        if default_name not in parsed:
            raise ValueError(f"default 指向不存在的模型：{default_name!r}")
        raw_roles = raw.get("roles")
        roles: dict[str, str] = {}
        if raw_roles is not None:
            if not isinstance(raw_roles, dict):
                raise ValueError("roles 必须是角色名到模型名的映射对象。")
            for raw_role, raw_target in raw_roles.items():
                role_name = str(raw_role).strip()
                if not role_name:
                    raise ValueError("roles 中的角色名称不能为空。")
                if not isinstance(raw_target, str) or not raw_target.strip():
                    raise ValueError(f"角色 {role_name!r} 必须是模型名称字符串。")
                target_name = raw_target.strip()
                if target_name not in parsed:
                    raise ValueError(f"角色 {role_name!r} 指向不存在的模型：{target_name!r}")
                roles[role_name] = target_name
        return cls(default=default_name, models=parsed, roles=roles)

    # 按名称读取目标模型档案（缺省返回 default 模型）
    def get_profile(self, name: str | None = None) -> ModelProfile:
        target = (name or self.default).strip()
        if target not in self.models:
            available = "、".join(sorted(self.models))
            raise ValueError(f"未找到模型配置 {target!r}，可用模型：{available}")
        return self.models[target]

    # 导出全部模型配置的脱敏摘要，供终端打印与状态面板展示
    def redacted_dict(self, env: Mapping[str, str] | None = None) -> dict[str, object]:
        return {
            "default": self.default,
            "models": {name: profile.redacted_dict(env) for name, profile in self.models.items()},
            "roles": dict(self.roles),
        }


# 翻译器入职协议（Pi ch04 第三层宪法）：任意厂商协议只需实现 api 标识与 stream() 翻译方法
@runtime_checkable
class ApiProvider(Protocol):
    api: str

    # 将中立 Context 与 StreamOptions 翻译为厂商请求并吐出 12 种统一流式事件
    def stream(
        self,
        model: ModelProfile,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]: ...


# 绑定后的前台调用句柄协议（供 Agent Loop、缓存观测与业务编排统一消费）
@runtime_checkable
class ModelInvoker(Protocol):
    last_usage: TokenUsage | None

    # 流式发送中立上下文并产出统一 AssistantMessageEvent 事件流
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]: ...

    # 同步消费流式调用并返回最终 AssistantMessage（遇到 StreamError 抛出 RuntimeError）
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage: ...
