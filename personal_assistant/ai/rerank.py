"""Layer 1 标准 /v1/rerank 重排客户端（零第三方依赖，兼容硅基流动 BGE-Reranker / Cohere / Jina / vLLM / TEI）。"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit

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
    http_origin,
    is_local_base_url,
    raise_if_aborted,
    redact_error_tokens,
    same_http_origin,
)


# 单条 Cross-Encoder 重排打分结果（对应输入 documents 下标与相关性得分）
@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


# 统一 Cross-Encoder 重排调用协议（供 Layer 3 检索插件精排阶段消费）
@runtime_checkable
class RerankInvoker(Protocol):
    model_id: str

    # 对 (query, documents) 执行精排，返回按得分降序排列的 RerankResult 列表，支持协同取消
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> list[RerankResult]: ...


# 兼容硅基流动 BGE-Reranker / Cohere v1-v2 / Jina / vLLM / TEI 的标准 /v1/rerank 纯标准库客户端
class OpenAIRerankClient:
    # 绑定模型档案与可替换 JSON POST 传输函数，并校验 HTTP(S) 协议与非空模型名
    def __init__(
        self,
        profile: ModelProfile,
        *,
        post_json: PostJson | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if http_origin(profile.base_url) is None:
            raise ValueError(f"重排模型 {profile.name!r} 的 base_url 必须是合法的 http:// 或 https:// 地址。")
        if not isinstance(profile.model, str) or not profile.model.strip():
            raise ValueError(f"重排模型 {profile.name!r} 缺少非空 model 名称。")
        self.profile = profile
        self._post_json: PostJson = post_json or default_post_json
        self._env = env

    # 返回去空白后的规范重排模型标识
    @property
    def model_id(self) -> str:
        return self.profile.model.strip()

    # 解析目标 /rerank 端点 URL（保留查询参数，若路径已以 /rerank 结尾如 Cohere /v2/rerank 则不重复拼接）
    def _endpoint_url(self) -> str:
        parts = urlsplit(self.profile.base_url.strip())
        path = parts.path.rstrip("/")
        if not path.endswith("/rerank"):
            path = f"{path}/rerank"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))

    # 解析并严格校验重排响应 JSON（兼容 relevance_score 与 score 字段，拒绝越界/重复/非有限值）
    @staticmethod
    def _parse_response(
        response: object,
        doc_count: int,
        top_n: int | None,
    ) -> list[RerankResult]:
        if not isinstance(response, dict):
            raise ValueError("Rerank 响应顶层必须是 JSON 对象。")
        raw_results = response.get("results") if "results" in response else response.get("data")
        if not isinstance(raw_results, list):
            raise ValueError("Rerank 响应缺少有效的 results 数组。")

        parsed: list[RerankResult] = []
        seen_indices: set[int] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                raise ValueError("Rerank results 项必须是 JSON 对象。")
            raw_idx = item.get("index")
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int) or not (0 <= raw_idx < doc_count):
                raise ValueError(f"Rerank 响应 index 越界或非法：{raw_idx!r}。")
            if raw_idx in seen_indices:
                raise ValueError(f"Rerank 响应包含重复 index：{raw_idx}。")
            seen_indices.add(raw_idx)

            raw_score = item["relevance_score"] if "relevance_score" in item else item.get("score")
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError(f"Rerank 响应 score 缺失或非数字：{raw_score!r}。")
            score_val = float(raw_score)
            if not math.isfinite(score_val):
                raise ValueError(f"Rerank 响应 score 必须是有限数字：{raw_score!r}。")
            parsed.append(RerankResult(index=raw_idx, score=score_val))

        parsed.sort(key=lambda r: (-r.score, r.index))
        if top_n is not None:
            parsed = parsed[:top_n]
        return parsed

    # 调用标准 /rerank 接口对候选文档精排，严格校验输入、支持 AbortSignal 与密钥脱敏错误收口
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> list[RerankResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("rerank() 的 query 必须是非空字符串。")
        if isinstance(documents, (str, bytes, bytearray)):
            raise TypeError("rerank() 的 documents 必须是字符串序列（如 list[str]），不能是单个字符串。")
        if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
            raise ValueError("top_n 必须是正整数或省略。")
        if not documents:
            return []
        for idx, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise TypeError(f"第 {idx} 条待重排文档必须是字符串。")

        api_key = self.profile.resolve_api_key(self._env)
        if not api_key and not is_local_base_url(self.profile.base_url):
            raise LLMNotConfigured(
                f"重排模型 {self.profile.name!r} 未配置 API Key，请检查 models.dsl 或环境变量。"
            )

        raise_if_aborted(abort_signal)
        url = self._endpoint_url()
        headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload: dict[str, object] = {
            "model": self.model_id,
            "query": query.strip(),
            "documents": list(documents),
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = min(top_n, len(documents))

        try:
            response = self._post_json(url, payload, headers, abort_signal=abort_signal)
        except StreamAbortedError:
            raise
        except HTTPError as error:
            if abort_signal is not None and abort_signal.aborted:
                try:
                    error.close()
                except (OSError, ValueError, AttributeError):
                    pass
                raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
            _, diag = format_http_error(error)
            msg = f"Rerank 请求失败（HTTP {error.code}）：{diag}" if diag else f"Rerank 请求失败（HTTP {error.code}）"
            raise RuntimeError(msg) from None
        except Exception as error:  # noqa: BLE001
            if abort_signal is not None and abort_signal.aborted:
                raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
            raise RuntimeError(redact_error_tokens(f"Rerank 请求失败：{type(error).__name__}: {error}")) from None

        return self._parse_response(response, len(documents), top_n)


# 从 ModelProfile 构造标准 OpenAIRerankClient
def from_rerank_profile(
    profile: ModelProfile,
    *,
    env: Mapping[str, str] | None = None,
) -> OpenAIRerankClient:
    return OpenAIRerankClient(profile, env=env)


# 从环境变量构造可选 Rerank 客户端（未设 RERANK_MODEL 返回 None；严禁向不同源的 RERANK_BASE_URL 泄漏 LLM_API_KEY）
def from_env_reranker(env: Mapping[str, str] | None = None) -> OpenAIRerankClient | None:
    source = os.environ if env is None else env
    model = str(source.get("RERANK_MODEL") or "").strip()
    if not model:
        return None
    raw_rr_url = str(source.get("RERANK_BASE_URL") or "").strip()
    raw_llm_url = str(source.get("LLM_BASE_URL") or "").strip()
    base_url = raw_rr_url or raw_llm_url
    if not base_url:
        raise ValueError("已设置 RERANK_MODEL，但缺少 RERANK_BASE_URL（或 LLM_BASE_URL）。")

    raw_rr_key = str(source.get("RERANK_API_KEY") or "").strip()
    raw_llm_key = str(source.get("LLM_API_KEY") or "").strip()
    if raw_rr_key:
        api_key = raw_rr_key
    elif not raw_rr_url or (raw_llm_url and same_http_origin(raw_rr_url, raw_llm_url)):
        api_key = raw_llm_key
    else:
        api_key = ""

    profile = ModelProfile.from_dict(
        "env-reranker",
        {
            "api": "openai-rerank",
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
        },
    )
    return OpenAIRerankClient(profile, env=source)
