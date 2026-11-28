"""Layer 1 标准 /v1/embeddings 向量客户端与 float32 编解码底座（零第三方依赖，兼容 OpenAI/硅基流动/通义/Ollama）。"""
from __future__ import annotations

import math
import os
import struct
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError
from urllib.parse import urlparse

from personal_assistant.ai.protocol import (
    AbortSignal,
    LLMNotConfigured,
    ModelProfile,
    StreamAbortedError,
)
from personal_assistant.ai.providers.http import (
    PostJson,
    default_post_json,
    format_http_error,
    raise_if_aborted,
    redact_error_tokens,
)

DEFAULT_EMBEDDING_BATCH_SIZE = 32
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


# 判断 base_url 是否指向本机回环地址（如本地 Ollama，允许免 API Key 调用）
def _is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").strip().lower()
    return host in _LOCAL_HOSTS


# 校验并 L2 单位归一化原始浮点向量（拒绝空向量、零向量、布尔值与 NaN/Inf）
def normalize_vector(vec: Sequence[float]) -> tuple[float, ...]:
    if not vec:
        raise ValueError("向量不能为空。")
    floats: list[float] = []
    for item in vec:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("向量元素必须是有限实数，不能包含布尔值或非数值类型。")
        val = float(item)
        if not math.isfinite(val):
            raise ValueError("向量元素必须是有限实数（不能为 NaN 或 Inf）。")
        floats.append(val)
    norm_sq = math.sumprod(floats, floats)
    if not math.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ValueError("向量 L2 范数必须为正有限数，不能为零向量。")
    inv_norm = 1.0 / math.sqrt(norm_sq)
    return tuple(v * inv_norm for v in floats)


# 将有限浮点向量编码为小端 IEEE 754 float32 二进制 BLOB（每维 4 字节，跨平台稳定）
def pack_vector(vec: Sequence[float]) -> bytes:
    if not vec:
        raise ValueError("向量不能为空。")
    floats: list[float] = []
    for item in vec:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("向量元素必须是有限实数。")
        val = float(item)
        if not math.isfinite(val):
            raise ValueError("向量元素必须是有限实数。")
        floats.append(val)
    return struct.pack(f"<{len(floats)}f", *floats)


# 将小端 float32 BLOB 解码为浮点元组并严格校验字节长度与有限性
def unpack_vector(blob: bytes | bytearray | memoryview, dims: int) -> tuple[float, ...]:
    if isinstance(dims, bool) or not isinstance(dims, int) or dims < 1:
        raise ValueError("dims 必须是正整数。")
    raw = bytes(blob)
    expected_bytes = dims * 4
    if len(raw) != expected_bytes:
        raise ValueError(f"向量 BLOB 长度不匹配：期望 {expected_bytes} 字节（{dims} 维），实际 {len(raw)} 字节。")
    values = struct.unpack(f"<{dims}f", raw)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("向量 BLOB 包含非法非有限浮点数。")
    return values


# 基于 Python 3.12 内置 C 级 math.sumprod 计算两个单位向量的精确余弦相似度（夹紧至 [-1, 1]）
def cosine_similarity(unit_a: Sequence[float], unit_b: Sequence[float]) -> float:
    if not unit_a or len(unit_a) != len(unit_b):
        raise ValueError("向量为空或维度不一致，无法计算余弦相似度。")
    dot = float(math.sumprod(unit_a, unit_b))
    if not math.isfinite(dot):
        raise ValueError("向量内积结果非有限数。")
    return max(-1.0, min(1.0, dot))


# 统一文本向量化调用协议（供 Layer 3 检索插件与离线批量向量化消费）
@runtime_checkable
class EmbeddingInvoker(Protocol):
    model_id: str

    # 将输入文本序列批量编码为 L2 单位归一化浮点向量列表，支持协同取消
    def embed(
        self,
        texts: Sequence[str],
        *,
        abort_signal: AbortSignal | None = None,
    ) -> list[tuple[float, ...]]: ...


# 兼容 OpenAI / 硅基流动 / 通义千问 / 本地 Ollama 的标准 /v1/embeddings 纯标准库客户端
class OpenAIEmbeddingClient:
    # 绑定模型档案、可选维度、分批大小与可替换 POST 传输函数
    def __init__(
        self,
        profile: ModelProfile,
        *,
        dimensions: int | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        post_json: PostJson | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size 必须是正整数。")
        if dimensions is not None and (isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1):
            raise ValueError("dimensions 必须是正整数或省略。")
        self.profile = profile
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._post_json: PostJson = post_json or default_post_json
        self._env = env

    # 返回规范模型标识（作为持久化向量缓存的主键维度之一）
    @property
    def model_id(self) -> str:
        return self.profile.model

    # 解析目标 /embeddings 端点 URL（避免对已含 /embeddings 后缀的地址重复拼接）
    def _endpoint_url(self) -> str:
        base = self.profile.base_url.rstrip("/")
        return base if base.endswith("/embeddings") else f"{base}/embeddings"

    # 解析单批次响应 JSON，按 index 对齐输入顺序、校验维度一致性并执行 L2 单位归一化
    def _parse_batch_response(
        self,
        response: object,
        expected_count: int,
    ) -> list[tuple[float, ...]]:
        if not isinstance(response, dict):
            raise ValueError("Embedding 响应顶层必须是 JSON 对象。")
        raw_data = response.get("data")
        if not isinstance(raw_data, list) or len(raw_data) != expected_count:
            got = len(raw_data) if isinstance(raw_data, list) else type(raw_data).__name__
            raise ValueError(f"Embedding 响应数量不匹配：期望 {expected_count} 条，实际 {got}。")

        indexed: list[tuple[int, tuple[float, ...]]] = []
        seen_indices: set[int] = set()
        for pos, item in enumerate(raw_data):
            if not isinstance(item, dict):
                raise ValueError("Embedding data 项必须是 JSON 对象。")
            raw_idx = item.get("index", pos)
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int) or not (0 <= raw_idx < expected_count):
                raise ValueError(f"Embedding 响应 index 越界或非法：{raw_idx!r}。")
            if raw_idx in seen_indices:
                raise ValueError(f"Embedding 响应包含重复 index：{raw_idx}。")
            seen_indices.add(raw_idx)

            raw_vec = item.get("embedding")
            if not isinstance(raw_vec, list):
                raise ValueError("Embedding 响应缺少有效的 embedding 数值数组。")
            unit_vec = normalize_vector(raw_vec)
            if self.dimensions is not None and len(unit_vec) != self.dimensions:
                raise ValueError(
                    f"Embedding 返回维度与配置不一致：期望 {self.dimensions}，实际 {len(unit_vec)}。"
                )
            indexed.append((raw_idx, unit_vec))

        indexed.sort(key=lambda pair: pair[0])
        return [vec for _, vec in indexed]

    # 按 batch_size 分批调用 /v1/embeddings，支持 AbortSignal 批次前即时取消与密钥脱敏错误收口
    def embed(
        self,
        texts: Sequence[str],
        *,
        abort_signal: AbortSignal | None = None,
    ) -> list[tuple[float, ...]]:
        if not texts:
            return []
        for idx, txt in enumerate(texts):
            if not isinstance(txt, str):
                raise TypeError(f"第 {idx} 条待向量化输入必须是字符串。")

        api_key = self.profile.resolve_api_key(self._env)
        if not api_key and not _is_local_base_url(self.profile.base_url):
            raise LLMNotConfigured(
                f"向量模型 {self.profile.name!r} 未配置 API Key，请检查 models.dsl 或环境变量。"
            )

        url = self._endpoint_url()
        headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        results: list[tuple[float, ...]] = []
        expected_dims: int | None = self.dimensions

        for start in range(0, len(texts), self.batch_size):
            raise_if_aborted(abort_signal)
            batch = list(texts[start : start + self.batch_size])
            payload: dict[str, object] = {"model": self.profile.model, "input": batch}
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions

            try:
                response = self._post_json(url, payload, headers, abort_signal=abort_signal)
            except StreamAbortedError:
                raise
            except HTTPError as error:
                if abort_signal is not None and abort_signal.aborted:
                    raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
                _, diag = format_http_error(error)
                msg = f"Embedding 请求失败（HTTP {error.code}）：{diag}" if diag else f"Embedding 请求失败（HTTP {error.code}）"
                raise RuntimeError(msg) from None
            except Exception as error:  # noqa: BLE001
                if abort_signal is not None and abort_signal.aborted:
                    raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
                raise RuntimeError(redact_error_tokens(f"Embedding 请求失败：{type(error).__name__}: {error}")) from None

            batch_vecs = self._parse_batch_response(response, len(batch))
            for vec in batch_vecs:
                if expected_dims is None:
                    expected_dims = len(vec)
                elif len(vec) != expected_dims:
                    raise ValueError(
                        f"同一模型返回向量维度不一致：期望 {expected_dims}，实际 {len(vec)}。"
                    )
            results.extend(batch_vecs)

        return results


# 从 ModelProfile 构造标准 OpenAIEmbeddingClient
def from_embedding_profile(
    profile: ModelProfile,
    *,
    dimensions: int | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    env: Mapping[str, str] | None = None,
) -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient(
        profile,
        dimensions=dimensions,
        batch_size=batch_size,
        env=env,
    )


# 从环境变量构造可选 Embedding 客户端（未设置 EMBEDDING_MODEL 时返回 None，优雅退回纯词面检索）
def from_env_embedder(env: Mapping[str, str] | None = None) -> OpenAIEmbeddingClient | None:
    source = os.environ if env is None else env
    model = str(source.get("EMBEDDING_MODEL") or "").strip()
    if not model:
        return None
    base_url = str(source.get("EMBEDDING_BASE_URL") or source.get("LLM_BASE_URL") or "").strip()
    if not base_url:
        raise ValueError("已设置 EMBEDDING_MODEL，但缺少 EMBEDDING_BASE_URL（或 LLM_BASE_URL）。")
    api_key = str(source.get("EMBEDDING_API_KEY") or source.get("LLM_API_KEY") or "").strip()
    raw_dims = str(source.get("EMBEDDING_DIMENSIONS") or "").strip()
    dims: int | None = None
    if raw_dims:
        try:
            dims = int(raw_dims)
        except ValueError as error:
            raise ValueError(f"EMBEDDING_DIMENSIONS 必须是正整数：{raw_dims!r}") from error
        if dims < 1:
            raise ValueError(f"EMBEDDING_DIMENSIONS 必须大于等于 1：{dims}")
    profile = ModelProfile.from_dict(
        "env-embedder",
        {
            "api": "openai-embeddings",
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
        },
    )
    return OpenAIEmbeddingClient(profile, dimensions=dims, env=source)
