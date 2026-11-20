"""Layer 1 原子契约类型（对标 pi-ai）：工具签名 Schema、工具调用与 Token/缓存统计，零执行与业务依赖。"""
from __future__ import annotations

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
