"""Layer 1 原子契约类型（对标 pi-ai）：工具签名 Schema、工具调用、模型配置档案与 Token/缓存统计。"""
from __future__ import annotations

import os
from collections.abc import Mapping
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


# 底层最小工具契约：仅描述暴露给模型的工具签名，不包含任何本地执行逻辑（供中层 Tool 递进继承）
@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, str]
    param_enums: dict[str, tuple[str, ...]] | None = field(default=None, kw_only=True)

    # 导出标准 OpenAI Chat Completions Function Calling JSON Schema
    def to_openai_schema(self) -> dict[str, object]:
        properties: dict[str, dict[str, object]] = {}
        for param, desc in self.parameters.items():
            prop: dict[str, object] = {"type": "string", "description": desc}
            if self.param_enums and param in self.param_enums:
                prop["enum"] = list(self.param_enums[param])
            properties[param] = prop
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(self.parameters),
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

    # 从标准 JSON 字典构造 ModelsConfig 并校验默认模型存在
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
        return cls(default=default_name, models=parsed)

    # 按名称读取目标模型档案（缺省返回 default 模型）
    def get_profile(self, name: str | None = None) -> ModelProfile:
        target = (name or self.default).strip()
        if target not in self.models:
            available = "、".join(sorted(self.models))
            raise ValueError(f"未找到模型配置 {target!r}，可用模型：{available}")
        return self.models[target]

    # 导出全部模型配置的脱敏摘要，供终端打印与 /status 展示
    def redacted_dict(self, env: Mapping[str, str] | None = None) -> dict[str, object]:
        return {
            "default": self.default,
            "models": {name: profile.redacted_dict(env) for name, profile in self.models.items()},
        }
