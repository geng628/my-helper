"""Layer 1 共享标准库 HTTP + SSE 传输底座：重试、429 Retry-After、密钥脱敏、确定性关流与通用 SSE 解析。"""
from __future__ import annotations

import http.client
import json
import math
import re
import ssl
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PostJson = Callable[[str, dict[str, object], dict[str, str]], dict[str, object]]
PostStream = Callable[[str, dict[str, object], dict[str, str]], Iterator[str]]

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
_ANTHROPIC_KEY_RE = re.compile(r"(?i)(x-api-key\s*[:=]\s*)[A-Za-z0-9._~+/=-]+")
_GOOG_HEADER_KEY_RE = re.compile(r"(?i)(x-goog-api-key\s*[:=]\s*)[A-Za-z0-9._~+/=-]+")
_GOOG_AIZA_RE = re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{8,}")
_QUERY_KEY_RE = re.compile(r"(?i)([?&]key=)[A-Za-z0-9._~%-]+")
_MAX_HTTP_ERROR_BODY_BYTES = 1024


# 将任意值安全解析为非负整数（拒绝布尔值与非数字类型）
def non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(value):
        return 0
    return max(0, int(value))


# 脱敏错误文本中的 Bearer / sk- / x-api-key / x-goog-api-key / AIza / key= 等敏感凭据
def redact_error_tokens(text: str) -> str:
    masked = _BEARER_SECRET_RE.sub("Bearer ***", text)
    masked = _SK_SECRET_RE.sub("sk-***", masked)
    masked = _ANTHROPIC_KEY_RE.sub(r"\1***", masked)
    masked = _GOOG_HEADER_KEY_RE.sub(r"\1***", masked)
    masked = _GOOG_AIZA_RE.sub("AIza***", masked)
    return _QUERY_KEY_RE.sub(r"\1***", masked)


# 纯函数读取最多 1024 字节 HTTPError 响应体、确定性关闭底层流并脱敏，构造零猴子补丁的新 HTTPError
def format_http_error(error: HTTPError) -> tuple[HTTPError, str]:
    raw_reason = str(error.reason or "").strip()
    orig_reason = redact_error_tokens(raw_reason)
    body = ""
    try:
        raw = error.read(_MAX_HTTP_ERROR_BODY_BYTES)
    except (OSError, ValueError, AttributeError):
        raw = b""
    finally:
        try:
            error.close()
        except (OSError, ValueError, AttributeError):
            pass
    if isinstance(raw, (bytes, bytearray)):
        body = raw.decode("utf-8", errors="replace").strip()
    elif isinstance(raw, str):
        body = raw.strip()
    clean_body = redact_error_tokens(body)
    if clean_body and clean_body not in orig_reason:
        msg = f"{orig_reason} | {clean_body}" if orig_reason else clean_body
    else:
        msg = orig_reason
    if not clean_body and msg == raw_reason:
        return error, orig_reason
    safe_url = redact_error_tokens(str(error.url or ""))
    return HTTPError(safe_url, error.code, msg, error.headers, None), f"{orig_reason} {clean_body}".strip()


# 从 HTTP 429 响应头解析合法数字 Retry-After，并夹紧在 [0.1, 10.0] 秒内
def parse_retry_after_seconds(headers: Any) -> float | None:
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


# 构造 JSON POST 请求，并在连接或指定读取阶段遇到瞬时网络/TLS/5xx/429 异常时按指数退避或 Retry-After 重试
def _post_with_retry(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
    timeout: int = 60,
    operation: Callable[[Any], _T],
) -> _T:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "my-helper-v2/0.2",
        **headers,
    }
    for attempt in range(max_retries + 1):
        request = Request(url, data=body, headers=req_headers, method="POST")
        delay = 0.5 * (2**attempt)
        try:
            return operation(urlopen(request, timeout=timeout))
        except HTTPError as error:
            if (error.code < 500 and error.code != 429) or attempt >= max_retries:
                formatted_error, _ = format_http_error(error)
                if formatted_error is error:
                    raise
                raise formatted_error from error
            if error.code == 429:
                retry_after = parse_retry_after_seconds(error.headers)
                if retry_after is not None:
                    delay = retry_after
            try:
                error.close()
            except (OSError, ValueError, AttributeError):
                pass
        except _TRANSIENT_HTTP_ERRORS:
            if attempt >= max_retries:
                raise
        time.sleep(delay)
    raise RuntimeError("请求重试耗尽。")


# 完整读取 HTTP 响应字节流并按 UTF-8 解码为字符串
def _read_utf8_body(response: Any) -> str:
    with response:
        return response.read().decode("utf-8")


# 使用标准库 urllib 发送 JSON POST 请求（连接与响应体读取均纳入最多 2 次指数退避重试）
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


# 使用标准库 urllib 发起 SSE 流式 POST 请求并逐行吐出 UTF-8 文本（退出或中断时确定性关闭底层连接）
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


# 判断已累积的 SSE data 缓冲是否已形成完整 JSON 对象或 [DONE] 哨兵（兼容省略空行分隔的紧凑流）
def _is_complete_sse_payload(text: str) -> bool:
    stripped = text.strip()
    if stripped == "[DONE]":
        return True
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        return isinstance(json.loads(stripped), dict)
    except json.JSONDecodeError:
        return False


# 通用 SSE 行流解析器：兼容标准空行分隔事件、多行 data 拼接及无空行紧凑 JSON data 帧，产出 (event_name, data_payload)
def iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    event_name = ""
    data_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines).strip()
                if payload:
                    yield event_name, payload
            event_name = ""
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            if data_lines:
                payload = "\n".join(data_lines).strip()
                if payload:
                    yield event_name, payload
                data_lines = []
            event_name = line[6:].strip()
            continue
        if line.startswith("data:"):
            val = line[5:]
            if val.startswith(" "):
                val = val[1:]
            if not val.strip() and not data_lines:
                continue
            if data_lines:
                prev = "\n".join(data_lines)
                if _is_complete_sse_payload(prev) or val.strip() == "[DONE]":
                    payload = prev.strip()
                    if payload:
                        yield event_name, payload
                    event_name = ""
                    data_lines = []
            data_lines.append(val)

    if data_lines:
        payload = "\n".join(data_lines).strip()
        if payload:
            yield event_name, payload
