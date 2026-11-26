"""Layer 1 原子契约类型（对标 pi-ai）：工具签名 Schema、工具调用、模型配置档案与 Token/缓存统计。"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


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

    # 转换为便于 JSON 序列化的统计字典
    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_rate": self.cache_hit_rate,
        }


# 单次大模型流式响应产生的文本增量片段
@dataclass(frozen=True)
class TextDelta:
    text: str


# 单次大模型流式响应产生的思维链增量片段
@dataclass(frozen=True)
class ThinkingDelta:
    text: str


# 单次大模型流式响应结束事件（携带完整 assistant 消息、Token 统计、首字/总耗时与结束原因）
@dataclass(frozen=True)
class StreamDone:
    message: dict[str, object]
    usage: TokenUsage | None = None
    ttft_ms: float | None = None
    latency_ms: float = 0.0
    finish_reason: str | None = None


LLMStreamEvent = TextDelta | ThinkingDelta | StreamDone


# 底层最小工具契约：仅描述暴露给模型的工具签名，不包含任何本地执行逻辑（供中层 Tool 递进继承）
@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, str]
    param_enums: dict[str, tuple[str, ...]] | None = field(default=None, kw_only=True)
    required: tuple[str, ...] | None = field(default=None, kw_only=True)

    # 导出标准 OpenAI Chat Completions Function Calling JSON Schema
    def to_openai_schema(self) -> dict[str, object]:
        properties: dict[str, dict[str, object]] = {}
        for param, desc in self.parameters.items():
            prop: dict[str, object] = {"type": "string", "description": desc}
            if self.param_enums and param in self.param_enums:
                prop["enum"] = list(self.param_enums[param])
            properties[param] = prop
        required = (
            list(self.parameters)
            if self.required is None
            else [k for k in self.parameters if k in set(self.required)]
        )
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }


# 单条模型工具调用请求的轻量结构化视图
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    # 从 ToolCall 实例、OpenAI wire 字典或扁平工具调用字典解析标准 ToolCall
    @classmethod
    def from_raw(cls, raw: object) -> "ToolCall":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(f"非法的 tool_call 对象：{raw!r}")
        fn = raw.get("function")
        if isinstance(fn, Mapping):
            name = str(fn.get("name") or "")
            raw_args = fn.get("arguments")
        else:
            name = str(raw.get("name") or "")
            raw_args = raw.get("arguments")
        if isinstance(raw_args, Mapping):
            arguments = json.dumps(raw_args, ensure_ascii=False)
        elif raw_args is None or raw_args == "":
            arguments = "{}"
        else:
            arguments = str(raw_args)
        return cls(id=str(raw.get("id") or ""), name=name, arguments=arguments)

    # 导出标准 OpenAI Chat Completions tool_calls 条目字典
    def to_openai_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


# 从 assistant 消息字典中提取规范化的 ToolCall 列表（兼容 ToolCall 实例与 OpenAI wire 字典）
def extract_tool_calls(message: Mapping[str, object]) -> list[ToolCall]:
    raw_calls = message.get("tool_calls")
    if not raw_calls or not isinstance(raw_calls, Iterable) or isinstance(raw_calls, (str, bytes, Mapping)):
        return []
    return [ToolCall.from_raw(item) for item in raw_calls]


# 单个大模型的标准配置档案（仅认识结构化字典/JSON，支持直写 api_key 或引用环境变量 key_env）
@dataclass(frozen=True)
class ModelProfile:
    name: str
    base_url: str
    model: str
    provider: str = "openai_compatible"
    api_key: str = ""
    key_env: str = ""
    temperature: float = 0.2

    # 从标准 JSON 字典构造并校验 ModelProfile
    @classmethod
    def from_dict(cls, name: str, raw: dict[str, object]) -> "ModelProfile":
        if not isinstance(raw, dict):
            raise ValueError(f"模型 {name!r} 的配置必须是 JSON 对象。")
        base_url = str(raw.get("base_url") or "").strip()
        model = str(raw.get("model") or "").strip()
        provider = str(raw.get("provider") or "openai_compatible").strip().lower() or "openai_compatible"
        api_key = str(raw.get("api_key") or "").strip()
        key_env = str(raw.get("key_env") or "").strip()
        raw_temp = raw.get("temperature", raw.get("temp", 0.2))
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError(f"模型 {name!r} 的 base_url 必须以 http:// 或 https:// 开头。")
        if not model:
            raise ValueError(f"模型 {name!r} 缺少非空 model 名称。")
        try:
            temp = float(raw_temp)
        except (TypeError, ValueError) as error:
            raise ValueError(f"模型 {name!r} 的 temperature 必须是数字。") from error
        if isinstance(raw_temp, bool) or not (0.0 <= temp <= 2.0):
            raise ValueError(f"模型 {name!r} 的 temperature 必须在 [0, 2] 之间。")
        return cls(
            name=name,
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            key_env=key_env,
            temperature=temp,
        )

    # 解析实际可用的 API Key（优先使用直写的 api_key，其次读取 key_env 对应的环境变量）
    def resolve_api_key(self, env: Mapping[str, str] | None = None) -> str:
        if self.api_key:
            return self.api_key
        if self.key_env:
            source = os.environ if env is None else env
            return str(source.get(self.key_env, "")).strip()
        return ""

    # 输出完全脱敏的展示字典，严禁暴露任何明文 api_key
    def redacted_dict(self, env: Mapping[str, str] | None = None) -> dict[str, object]:
        resolved = self.resolve_api_key(env)
        if self.api_key:
            key_source = "inline"
        elif self.key_env:
            key_source = f"env:{self.key_env}"
        else:
            key_source = "none"
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "api_key_configured": bool(resolved),
            "key_source": key_source,
        }


# 多模型注册表配置对象（对应标准 models.json 的顶层结构）
@dataclass(frozen=True)
class ModelsConfig:
    default: str
    models: dict[str, ModelProfile]
    consolidator: str | None = None

    # 从标准 JSON 字典构造 ModelsConfig 并校验默认模型与可选后台沉淀模型存在
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
            parsed[name] = ModelProfile.from_dict(name, entry)
        default_name = str(raw.get("default") or "").strip()
        if not default_name:
            if len(parsed) == 1:
                default_name = next(iter(parsed))
            else:
                raise ValueError("配置多个模型时必须指定 default 默认模型。")
        if default_name not in parsed:
            raise ValueError(f"default 指向不存在的模型：{default_name!r}")
        raw_consolidator = raw.get("consolidator")
        if raw_consolidator is not None and not isinstance(raw_consolidator, str):
            raise ValueError("consolidator 必须是模型名称字符串。")
        consolidator_name = (raw_consolidator or "").strip() or None
        if consolidator_name is not None and consolidator_name not in parsed:
            raise ValueError(f"consolidator 指向不存在的模型：{consolidator_name!r}")
        return cls(default=default_name, models=parsed, consolidator=consolidator_name)

    # 按名称读取目标模型档案（缺省返回 default 模型）
    def get_profile(self, name: str | None = None) -> ModelProfile:
        target = (name or self.default).strip()
        if target not in self.models:
            available = "、".join(sorted(self.models))
            raise ValueError(f"未找到模型配置 {target!r}，可用模型：{available}")
        return self.models[target]

    # 导出全部模型配置的脱敏摘要，供终端打印与 /status 展示
    def redacted_dict(self, env: Mapping[str, str] | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "default": self.default,
            "models": {name: profile.redacted_dict(env) for name, profile in self.models.items()},
        }
        if self.consolidator is not None:
            result["consolidator"] = self.consolidator
        return result
