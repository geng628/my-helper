"""Layer 1 共享标准库 HTTP + SSE 传输底座：可中断连接、重试/429 Retry-After、脱敏、统一错误收口与通用 SSE 解析。"""
from __future__ import annotations

import http.client
import json
import math
import re
import socket
import ssl
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, HTTPSHandler, Request, build_opener

from personal_assistant.ai.protocol import (
    AbortSignal,
    AssistantBlock,
    AssistantMessage,
    ErrorReason,
    StreamAbortedError,
    StreamError,
    TokenUsage,
)

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
_ANTHROPIC_KEY_RE = re.compile(r"""(?i)(x-api-key["']?\s*[:=]\s*["']?)[A-Za-z0-9._~+/=-]+""")
_GOOG_HEADER_KEY_RE = re.compile(r"""(?i)(x-goog-api-key["']?\s*[:=]\s*["']?)[A-Za-z0-9._~+/=-]+""")
_GOOG_AIZA_RE = re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{8,}")
_QUERY_KEY_RE = re.compile(r"(?i)([?&]key=)[A-Za-z0-9._~%-]+")
_MAX_HTTP_ERROR_BODY_BYTES = 1024


# JSON POST 传输契约（统一单轨接受可选只读取消信号 abort_signal）
class PostJson(Protocol):
    # 发送 JSON POST 请求并返回解析后的 JSON 字典
    def __call__(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        *,
        abort_signal: AbortSignal | None = None,
    ) -> dict[str, object]: ...


# SSE 流式 POST 传输契约（统一单轨接受可选只读取消信号 abort_signal）
class PostStream(Protocol):
    # 发送流式 POST 请求并逐行吐出 UTF-8 文本行
    def __call__(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        *,
        abort_signal: AbortSignal | None = None,
    ) -> Iterator[str]: ...


# 若协同取消信号已触发，立即抛出携带原因的 StreamAbortedError
def raise_if_aborted(abort_signal: AbortSignal | None) -> None:
    if abort_signal is not None and abort_signal.aborted:
        raise StreamAbortedError(abort_signal.reason or "请求已取消")


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
    raw_url = str(error.url or "")
    safe_url = redact_error_tokens(raw_url)
    if not clean_body and msg == raw_reason and safe_url == raw_url:
        return error, orig_reason
    return HTTPError(safe_url, error.code, msg, error.headers, None), f"{orig_reason} {clean_body}".strip()


# 统一将流式翻译器捕获的异常收敛为 StreamError（协同取消统一映射为 reason="aborted" 与 stop_reason="aborted"）
def build_stream_error(
    exc: Exception,
    *,
    blocks: Sequence[AssistantBlock],
    usage: TokenUsage | None,
    started_at: float,
    abort_signal: AbortSignal | None = None,
) -> StreamError:
    is_aborted = isinstance(exc, StreamAbortedError) or (
        abort_signal is not None and abort_signal.aborted
    )
    if is_aborted:
        reason: ErrorReason = "aborted"
        raw_msg = (
            (abort_signal.reason if abort_signal is not None and abort_signal.reason else "")
            or str(exc)
            or "请求已取消"
        )
        err_msg = redact_error_tokens(raw_msg)
    else:
        reason = "error"
        if isinstance(exc, HTTPError):
            _, diag = format_http_error(exc)
            err_msg = f"HTTP {exc.code}: {diag}" if diag else f"HTTP {exc.code}"
        else:
            err_msg = redact_error_tokens(f"{type(exc).__name__}: {exc}")
    err_msg_obj = AssistantMessage(
        blocks=tuple(blocks),
        stop_reason=reason,
        usage=usage,
        error_message=err_msg,
    )
    return StreamError(
        reason=reason,
        message=err_msg_obj,
        error_message=err_msg,
        latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
    )


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


# 安全中断并关闭底层阻塞套接字，用于毫秒级唤醒卡在 TTFB 或流读取上的阻塞线程
def _shutdown_socket(sock: object) -> None:
    if sock is None:
        return
    shutdown_fn = getattr(sock, "shutdown", None)
    if callable(shutdown_fn):
        try:
            shutdown_fn(socket.SHUT_RDWR)
        except OSError:
            pass
    close_fn = getattr(sock, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except OSError:
            pass


# 静默关闭响应或流对象，忽略重复关闭异常
def _close_quietly(obj: object) -> None:
    close_fn = getattr(obj, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except (OSError, ValueError, AttributeError):
            pass


# 从 urllib/http.client 响应对象提取底层 socket 引用（不存在时返回 None）
def _extract_response_socket(response: object) -> object | None:
    direct = getattr(response, "sock", None)
    if direct is not None:
        return direct
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    return getattr(raw, "_sock", None)


# 受控 HTTP 响应包装器：在退出上下文或关闭流时幂等退订全部取消监听器并关闭底层响应，零猴子补丁
class _ManagedHTTPResponse:
    # 绑定原始响应对象与取消监听清理回调
    def __init__(self, inner: Any, on_close: Callable[[], None]) -> None:
        self._inner = inner
        self._on_close = on_close
        self._closed = False

    # 进入 with 上下文返回自身
    def __enter__(self) -> "_ManagedHTTPResponse":
        enter_fn = getattr(self._inner, "__enter__", None)
        if callable(enter_fn):
            enter_fn()
        return self

    # 退出 with 上下文时确定性清理监听器并关闭响应
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # 透传逐行字节迭代能力
    def __iter__(self) -> Iterator[bytes]:
        return iter(self._inner)

    # 透传 read 读取能力
    def read(self, amt: int | None = None) -> bytes:
        return self._inner.read() if amt is None else self._inner.read(amt)

    # 幂等释放所有取消回调并关闭底层响应
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._on_close()
        finally:
            _close_quietly(self._inner)

    # 透传其余响应属性（如 status / headers / fp）
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# 单轨可中断 HTTP 请求发起器：保留 urllib 代理/重定向能力，并在建连后即刻挂接 socket 取消回调
def open_http_request(
    request: Request,
    *,
    timeout: float = 60,
    abort_signal: AbortSignal | None = None,
) -> Any:
    raise_if_aborted(abort_signal)
    unsub_lock = threading.Lock()
    unsubs: list[Callable[[], None]] = []
    released = False

    # 将新建套接字登记到当前请求的取消清理表
    def _track_socket(sock: object) -> None:
        if abort_signal is None or sock is None:
            return
        unsub = abort_signal.subscribe(lambda s=sock: _shutdown_socket(s))
        with unsub_lock:
            if released:
                unsub()
            else:
                unsubs.append(unsub)

    # 幂等退订本次请求登记的全部取消回调
    def _release_unsubs() -> None:
        nonlocal released
        with unsub_lock:
            if released:
                return
            released = True
            pending = list(unsubs)
            unsubs.clear()
        for unsub in pending:
            unsub()

    # 支持协同取消的 HTTP 连接子类：建连后即刻绑定 socket 中断回调
    class _AbortableHTTPConnection(http.client.HTTPConnection):
        # 建连前后检查取消信号并跟踪底层套接字
        def connect(self) -> None:
            raise_if_aborted(abort_signal)
            super().connect()
            _track_socket(self.sock)
            if abort_signal is not None and abort_signal.aborted:
                _shutdown_socket(self.sock)
                raise StreamAbortedError(abort_signal.reason or "请求已取消")

    # 支持协同取消的 HTTPS 连接子类：建连与 TLS 握手完成后即刻绑定 socket 中断回调
    class _AbortableHTTPSConnection(http.client.HTTPSConnection):
        # 建连前后检查取消信号并跟踪底层 TLS 套接字
        def connect(self) -> None:
            raise_if_aborted(abort_signal)
            super().connect()
            _track_socket(self.sock)
            if abort_signal is not None and abort_signal.aborted:
                _shutdown_socket(self.sock)
                raise StreamAbortedError(abort_signal.reason or "请求已取消")

    # 使用可中断 HTTP 连接的 urllib 处理器
    class _AbortableHTTPHandler(HTTPHandler):
        # 替换默认 HTTPConnection 为可中断子类
        def http_open(self, req: Request) -> Any:
            return self.do_open(_AbortableHTTPConnection, req)

    # 使用可中断 HTTPS 连接的 urllib 处理器（保留默认 SSL 上下文）
    class _AbortableHTTPSHandler(HTTPSHandler):
        # 替换默认 HTTPSConnection 为可中断子类
        def https_open(self, req: Request) -> Any:
            return self.do_open(_AbortableHTTPSConnection, req, context=self._context)

    opener = build_opener(_AbortableHTTPHandler(), _AbortableHTTPSHandler())
    try:
        response = opener.open(request, timeout=timeout)
    except BaseException:
        _release_unsubs()
        if abort_signal is not None and abort_signal.aborted:
            raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
        raise
    if abort_signal is not None and abort_signal.aborted:
        _release_unsubs()
        _close_quietly(response)
        raise StreamAbortedError(abort_signal.reason or "请求已取消")
    return _ManagedHTTPResponse(response, _release_unsubs)


# 构造 JSON POST 请求，并在连接或指定读取阶段遇瞬时网络/TLS/5xx/429 时执行可中断指数退避重试
def _post_with_retry(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
    timeout: int = 60,
    abort_signal: AbortSignal | None = None,
    operation: Callable[[Any], _T],
) -> _T:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "my-helper-v2/0.2",
        **headers,
    }
    for attempt in range(max_retries + 1):
        raise_if_aborted(abort_signal)
        request = Request(url, data=body, headers=req_headers, method="POST")
        delay = 0.5 * (2**attempt)
        try:
            response = open_http_request(request, timeout=timeout, abort_signal=abort_signal)
            return operation(response)
        except StreamAbortedError:
            raise
        except HTTPError as error:
            if abort_signal is not None and abort_signal.aborted:
                _close_quietly(error)
                raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
            if (error.code < 500 and error.code != 429) or attempt >= max_retries:
                formatted_error, _ = format_http_error(error)
                if formatted_error is error:
                    raise
                raise formatted_error from error
            if error.code == 429:
                retry_after = parse_retry_after_seconds(error.headers)
                if retry_after is not None:
                    delay = retry_after
            _close_quietly(error)
        except _TRANSIENT_HTTP_ERRORS:
            if abort_signal is not None and abort_signal.aborted:
                raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
            if attempt >= max_retries:
                raise
        except Exception:
            if abort_signal is not None and abort_signal.aborted:
                raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
            raise

        if abort_signal is not None:
            if abort_signal.wait(delay):
                raise StreamAbortedError(abort_signal.reason or "请求已取消")
        else:
            time.sleep(delay)
    raise RuntimeError("请求重试耗尽。")


# 完整读取 HTTP 响应字节流并按 UTF-8 解码，支持读取阶段即时协同取消
def _read_utf8_body(response: Any, abort_signal: AbortSignal | None = None) -> str:
    unsub = (
        abort_signal.subscribe(
            lambda: (_shutdown_socket(_extract_response_socket(response)), _close_quietly(response))
        )
        if abort_signal is not None
        else None
    )
    try:
        with response:
            raise_if_aborted(abort_signal)
            raw = response.read()
            raise_if_aborted(abort_signal)
            return raw.decode("utf-8")
    except StreamAbortedError:
        raise
    except Exception:
        if abort_signal is not None and abort_signal.aborted:
            raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
        raise
    finally:
        if unsub is not None:
            unsub()


# 使用标准库发送可中断 JSON POST 请求（连接与响应体读取均纳入最多 2 次可中断退避重试）
def default_post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
    abort_signal: AbortSignal | None = None,
) -> dict[str, object]:
    raw_text = _post_with_retry(
        url,
        payload,
        headers,
        max_retries=max_retries,
        abort_signal=abort_signal,
        operation=lambda resp: _read_utf8_body(resp, abort_signal),
    )
    return json.loads(raw_text)


# 使用标准库发起可中断 SSE 流式 POST 请求并逐行吐出 UTF-8 文本（退出或取消时确定性关闭底层连接）
def default_post_stream(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int = 2,
    abort_signal: AbortSignal | None = None,
) -> Iterator[str]:
    response = _post_with_retry(
        url,
        payload,
        headers,
        max_retries=max_retries,
        abort_signal=abort_signal,
        operation=lambda resp: resp,
    )
    unsub = (
        abort_signal.subscribe(
            lambda: (_shutdown_socket(_extract_response_socket(response)), _close_quietly(response))
        )
        if abort_signal is not None
        else None
    )
    try:
        with response:
            for raw_line in response:
                raise_if_aborted(abort_signal)
                yield raw_line.decode("utf-8")
            raise_if_aborted(abort_signal)
    except StreamAbortedError:
        raise
    except Exception:
        if abort_signal is not None and abort_signal.aborted:
            raise StreamAbortedError(abort_signal.reason or "请求已取消") from None
        raise
    finally:
        if unsub is not None:
            unsub()


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


# 通用 SSE 行流解析器：兼容标准空行分隔、多行 data 拼接及紧凑 JSON 帧，退出/异常时确定性关闭内层流
def iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    event_name = ""
    data_lines: list[str] = []

    try:
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
    finally:
        close_fn = getattr(lines, "close", None)
        if callable(close_fn):
            close_fn()
