"""联网搜索与网页正文提取插件（零第三方依赖）：支持防 SSRF DNS Pinning 抓网页与多后端联网搜索。"""
from __future__ import annotations

import codecs
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, HTTPHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from personal_assistant.agent.tools import tool
from personal_assistant.assistant.common import (
    DEFAULT_SOFT_WRAP_CHARS,
    truncate_or_offload,
    truncate_utf8,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
LOCAL_HOST_SUFFIXES = (".local", ".localhost", ".internal")
FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")
NAT64_WELL_KNOWN_NETWORK = ipaddress.ip_network("64:ff9b::/96")
NAT64_LOCAL_NETWORK = ipaddress.ip_network("64:ff9b:1::/48")
SENSITIVE_REDIRECT_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-subscription-token", "cookie", "x-api-key"}
)
SKIP_HTML_TAGS = frozenset({"script", "style", "noscript", "svg", "iframe", "nav", "footer", "header"})
BLOCK_HTML_TAGS = frozenset(
    {"p", "div", "section", "article", "main", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br", "pre", "blockquote"}
)
ALLOWED_CONTENT_PREFIXES = ("text/", "application/json", "application/xml", "application/xhtml+xml")

DnsResolver = Callable[..., list[tuple[object, ...]]]
HttpGetFn = Callable[[str, dict[str, str], float, int], tuple[str, str, bytes]]
HttpPostJsonFn = Callable[[str, dict[str, object], dict[str, str], float], dict[str, object]]


# 联网插件的安全与资源限制策略（从 config/tools.json 的 web 段读取）
@dataclass(frozen=True)
class WebPolicy:
    timeout: float = 15.0
    max_download_bytes: int = 2 * 1024 * 1024
    max_output_bytes: int = 8192
    max_results: int = 5
    offload_dir: Path | None = None
    allow_fake_ip: bool = False

    # 从已解析的 web 配置字典直接构建 WebPolicy（支持绑定实例专属 offload_dir 与 allow_fake_ip）
    @classmethod
    def from_dict(
        cls,
        web_cfg: Mapping[str, object],
        *,
        offload_dir: Path | None = None,
    ) -> "WebPolicy":
        return cls(
            timeout=float(web_cfg.get("timeout", 15.0)),
            max_download_bytes=int(web_cfg.get("max_download_bytes", 2 * 1024 * 1024)),
            max_output_bytes=int(web_cfg.get("max_output_bytes", 8192)),
            max_results=max(1, min(10, int(web_cfg.get("max_results", 5)))),
            offload_dir=offload_dir,
            allow_fake_ip=bool(web_cfg.get("allow_fake_ip", False)),
        )

    # 从 config/tools.json 读取 web 配置段，缺省时使用安全默认值
    @classmethod
    def from_config(
        cls,
        config_path: Path | None = None,
        *,
        offload_dir: Path | None = None,
    ) -> "WebPolicy":
        cfg_file = (
            config_path
            if config_path is not None
            else Path(os.getenv("MY_HELPER_CONFIG", str(REPO_ROOT / "config"))) / "tools.json"
        ).resolve()
        web_cfg: dict[str, object] = {}
        if cfg_file.is_file():
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("web"), dict):
                web_cfg = raw["web"]
        return cls.from_dict(web_cfg, offload_dir=offload_dir)


# 判断单个 IPv4/IPv6 对象是否严格属于允许访问的公网地址（阻断 NAT64、ipv4_mapped、6to4、Teredo、IPv4 兼容地址与多播）
def _is_public_ip(
    ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    is_literal_ip: bool = False,
    allow_fake_ip: bool = False,
) -> bool:
    if isinstance(ip_obj, ipaddress.IPv4Address):
        if ip_obj.is_multicast:
            return False
        if allow_fake_ip and not is_literal_ip and ip_obj in FAKE_IP_NETWORK:
            return True
        return ip_obj.is_global

    if ip_obj.is_multicast or not ip_obj.is_global:
        return False
    if ip_obj in NAT64_WELL_KNOWN_NETWORK or ip_obj in NAT64_LOCAL_NETWORK:
        return False
    if ip_obj.packed[:12] == b"\x00" * 12:
        return False
    if ip_obj.ipv4_mapped is not None and not _is_public_ip(
        ip_obj.ipv4_mapped, is_literal_ip=is_literal_ip, allow_fake_ip=allow_fake_ip
    ):
        return False
    if ip_obj.sixtofour is not None and not _is_public_ip(
        ip_obj.sixtofour, is_literal_ip=is_literal_ip, allow_fake_ip=allow_fake_ip
    ):
        return False
    if ip_obj.teredo is not None and not all(
        _is_public_ip(embedded, is_literal_ip=is_literal_ip, allow_fake_ip=allow_fake_ip)
        for embedded in ip_obj.teredo
    ):
        return False
    return True


# 严格校验并解析目标 URL：仅限公开 http/https，返回 (clean_url, hostname, port, pinned_ip) 供连接器钉住 IP
def resolve_and_validate_public_url(
    url: str,
    *,
    resolve_dns: DnsResolver = socket.getaddrinfo,
    allow_fake_ip: bool = False,
) -> tuple[str, str, int, str]:
    clean = (url or "").strip()
    if not clean or any(ch in clean for ch in ("\x00", "\r", "\n")):
        raise ValueError("URL 不能为空且不能包含换行符。")
    parsed = urlparse(clean)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"只允许 http 或 https 协议，已拒绝：{parsed.scheme or '空协议'}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("URL 缺少有效主机名。")
    clean_host = host.rstrip(".")
    if not clean_host or clean_host == "localhost" or clean_host.endswith(LOCAL_HOST_SUFFIXES):
        raise ValueError(f"禁止访问本机或内网域名：{host}")
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as error:
        raise ValueError(f"URL 端口无效：{error}") from error

    ips: list[str] = []
    is_literal_ip = False
    try:
        ip_lit = ipaddress.ip_address(host.split("%", 1)[0])
        ips.append(str(ip_lit))
        is_literal_ip = True
    except ValueError:
        try:
            infos = resolve_dns(host, port)
        except OSError as error:
            raise ValueError(f"域名解析失败（{host}）：{error}") from error
        seen_ips: set[str] = set()
        for info in infos:
            sockaddr = info[4]
            if isinstance(sockaddr, tuple) and sockaddr:
                raw_addr = str(sockaddr[0]).split("%", 1)[0]
                if raw_addr not in seen_ips:
                    seen_ips.add(raw_addr)
                    ips.append(raw_addr)

    if not ips:
        raise ValueError(f"未能解析出有效 IP 地址：{host}")
    normalized_ips: list[str] = []
    for raw_ip in ips:
        ip_obj = ipaddress.ip_address(raw_ip.split("%", 1)[0])
        if not _is_public_ip(ip_obj, is_literal_ip=is_literal_ip, allow_fake_ip=allow_fake_ip):
            raise ValueError(f"禁止访问非公网地址（SSRF 防护）：{host} -> {ip_obj}")
        normalized_ips.append(str(ip_obj))
    return clean, host, port, normalized_ips[0]


# 严格校验目标 URL 仅限公开 http/https 地址，防止通过私网/回环/云元数据 IP 触发 SSRF
def validate_public_url(
    url: str,
    *,
    resolve_dns: DnsResolver = socket.getaddrinfo,
    allow_fake_ip: bool = False,
) -> str:
    clean, _, _, _ = resolve_and_validate_public_url(
        url,
        resolve_dns=resolve_dns,
        allow_fake_ip=allow_fake_ip,
    )
    return clean


# 使用已校验并钉住的 IP 直接建立 TCP 套接字连接，消除二次 DNS 解析引起的 TOCTOU Rebinding 竞态
def _connect_pinned_socket(conn: http.client.HTTPConnection, pinned_ip: str | None) -> None:
    target = pinned_ip or conn.host
    sys.audit("http.client.connect", conn, target, conn.port)
    conn.sock = socket.create_connection((target, conn.port), conn.timeout, conn.source_address)
    if conn._tunnel_host:
        conn._tunnel()


# 直连钉住 IP 的 HTTP 连接器（保留原始 Host 请求头）
class _PinnedHTTPConnection(http.client.HTTPConnection):
    # 保存已校验的目标 IP
    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        pinned_ip: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(host, port=port, **kwargs)  # type: ignore[arg-type]
        self._pinned_ip = pinned_ip

    # 直连钉住的 IP 套接字而不触发系统 DNS 解析
    def connect(self) -> None:
        _connect_pinned_socket(self, self._pinned_ip)


# 直连钉住 IP 的 HTTPS 连接器（保留原始域名用于 TLS SNI 与证书主机名校验）
class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    # 保存已校验的目标 IP
    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        pinned_ip: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(host, port=port, **kwargs)  # type: ignore[arg-type]
        self._pinned_ip = pinned_ip

    # 先直连钉住的 IP TCP 套接字，再用原始域名完成 TLS 握手与证书校验
    def connect(self) -> None:
        _connect_pinned_socket(self, self._pinned_ip)
        server_hostname = self._tunnel_host or self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


# 从请求 URL 提取 (host, port) 并从钉住映射表中查找对应已校验 IP
def _lookup_pinned_ip(
    req: Request,
    pinned_targets: Mapping[tuple[str, int], str],
    default_port: int,
) -> str:
    parsed = urlparse(req.full_url)
    host = (parsed.hostname or "").strip().lower()
    port = parsed.port or default_port
    pinned_ip = pinned_targets.get((host, port))
    if not pinned_ip:
        raise ValueError(f"缺少已校验的钉住 IP：{host}:{port}")
    return pinned_ip


# 绑定钉住 IP 映射表的 HTTP 处理器
class _PinnedHTTPHandler(HTTPHandler):
    # 注入与本次请求生命周期绑定的钉住 IP 映射表
    def __init__(self, pinned_targets: Mapping[tuple[str, int], str]) -> None:
        super().__init__()
        self._pinned_targets = pinned_targets

    # 使用 _PinnedHTTPConnection 发起 HTTP 请求
    def http_open(self, req: Request):  # type: ignore[override]
        pinned_ip = _lookup_pinned_ip(req, self._pinned_targets, 80)

        # 返回固定直连已校验 IP 的 HTTPConnection 实例
        def _conn_factory(conn_host: str, **kwargs: object) -> _PinnedHTTPConnection:
            return _PinnedHTTPConnection(conn_host, pinned_ip=pinned_ip, **kwargs)

        return self.do_open(_conn_factory, req)  # type: ignore[arg-type]


# 绑定钉住 IP 映射表的 HTTPS 处理器
class _PinnedHTTPSHandler(HTTPSHandler):
    # 注入钉住 IP 映射表与默认证书校验 TLS 上下文
    def __init__(
        self,
        pinned_targets: Mapping[tuple[str, int], str],
        context: ssl.SSLContext | None = None,
    ) -> None:
        super().__init__(context=context or ssl.create_default_context())
        self._pinned_targets = pinned_targets

    # 使用 _PinnedHTTPSConnection 发起 HTTPS 请求
    def https_open(self, req: Request):  # type: ignore[override]
        pinned_ip = _lookup_pinned_ip(req, self._pinned_targets, 443)

        # 返回固定直连已校验 IP 的 HTTPSConnection 实例
        def _conn_factory(conn_host: str, **kwargs: object) -> _PinnedHTTPSConnection:
            return _PinnedHTTPSConnection(conn_host, pinned_ip=pinned_ip, **kwargs)

        return self.do_open(_conn_factory, req, context=self._context)  # type: ignore[arg-type]


# 安全的 HTTP 重定向处理器：每跳重新解析校验 pinned_ip，且跨域重定向时剥离敏感鉴权请求头
class _SafeRedirectHandler(HTTPRedirectHandler):
    # 保存 DNS 解析器、Fake-IP 策略及可变钉住 IP 映射表
    def __init__(
        self,
        pinned_targets: dict[tuple[str, int], str],
        *,
        resolve_dns: DnsResolver = socket.getaddrinfo,
        allow_fake_ip: bool = False,
    ) -> None:
        super().__init__()
        self._pinned_targets = pinned_targets
        self._resolve_dns = resolve_dns
        self._allow_fake_ip = allow_fake_ip

    # 重定向前重新校验并更新目标 pinned_ip，跨域跳转时自动清除鉴权与 Cookie 请求头
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        target = urljoin(req.full_url, newurl)
        clean_target, target_host, target_port, target_pinned_ip = resolve_and_validate_public_url(
            target,
            resolve_dns=self._resolve_dns,
            allow_fake_ip=self._allow_fake_ip,
        )
        self._pinned_targets[(target_host, target_port)] = target_pinned_ip
        next_req = super().redirect_request(req, fp, code, msg, headers, clean_target)
        if next_req is None:
            return None

        old_parsed = urlparse(req.full_url)
        old_scheme = old_parsed.scheme.lower()
        old_origin = (
            old_scheme,
            (old_parsed.hostname or "").strip().lower(),
            old_parsed.port or (443 if old_scheme == "https" else 80),
        )
        new_origin = (urlparse(clean_target).scheme.lower(), target_host, target_port)
        if old_origin != new_origin:
            for hdr_map in (next_req.headers, next_req.unredirected_hdrs):
                for key in list(hdr_map):
                    if key.lower() in SENSITIVE_REDIRECT_HEADERS:
                        del hdr_map[key]
        return next_req


# 基于 Python 标准库 HTMLParser 的轻量网页正文与标题提取器
class _HTMLTextExtractor(HTMLParser):
    # 初始化标签跳过栈、标题缓冲与正文片段列表
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    # 遇到起始标签时更新忽略嵌套深度、标题状态或插入段落换行
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in SKIP_HTML_TAGS:
            self._skip_depth += 1
            return
        if lower == "title" and self._skip_depth == 0:
            self._in_title = True
        elif lower in BLOCK_HTML_TAGS and self._skip_depth == 0:
            self.text_parts.append("\n")

    # 遇到闭合标签时恢复嵌套状态或追加换行
    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in SKIP_HTML_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if lower == "title":
            self._in_title = False
        elif lower in BLOCK_HTML_TAGS and self._skip_depth == 0:
            self.text_parts.append("\n")

    # 收集可见文本节点（忽略脚本、样式与页眉页脚区域）
    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.text_parts.append(data)


# 从 HTML 字符串提取干净的页面标题与纯文本正文
def extract_text_from_html(html: str) -> tuple[str, str]:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    title = " ".join("".join(parser.title_parts).split())
    raw_lines = [
        " ".join(line.split())
        for line in "".join(parser.text_parts).splitlines()
    ]
    cleaned_lines = [line for line in raw_lines if line]
    return title, "\n".join(cleaned_lines)


# 从 Content-Type 响应头提取并校验合法文本字符编码，未知或非文本编解码器统一安全回退 utf-8
def _extract_charset(content_type: str) -> str:
    match = re.search(r"charset\s*=\s*\"?([A-Za-z0-9._-]+)\"?", content_type, flags=re.IGNORECASE)
    if not match:
        return "utf-8"
    candidate = match.group(1)
    try:
        info = codecs.lookup(candidate)
    except LookupError:
        return "utf-8"
    if not getattr(info, "_is_text_encoding", True):
        return "utf-8"
    return info.name


# 在墙钟总时长上限与最大字节数约束内分块读取 HTTP 响应体，防慢速滴水攻击挂死
def _read_body_with_deadline(resp: object, max_bytes: int, deadline: float) -> bytes:
    chunks: list[bytes] = []
    total = 0
    target = max_bytes + 1
    read1 = getattr(resp, "read1", None)
    reader = read1 if callable(read1) else getattr(resp, "read")
    while total < target:
        if time.monotonic() > deadline:
            raise TimeoutError("响应读取超过总时长限制。")
        chunk_size = min(16384, target - total)
        chunk = reader(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if time.monotonic() > deadline:
        raise TimeoutError("响应读取超过总时长限制。")
    return b"".join(chunks)


# 构建禁用系统代理、支持 DNS Pinning 与逐跳重定向校验的专用 urllib opener
def _open_pinned_request(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    data: bytes | None,
    timeout: float,
    max_bytes: int,
    resolve_dns: DnsResolver = socket.getaddrinfo,
    allow_fake_ip: bool = False,
) -> tuple[str, str, bytes]:
    clean_url, host, port, pinned_ip = resolve_and_validate_public_url(
        url,
        resolve_dns=resolve_dns,
        allow_fake_ip=allow_fake_ip,
    )
    pinned_targets: dict[tuple[str, int], str] = {(host, port): pinned_ip}
    opener = build_opener(
        ProxyHandler({}),
        _PinnedHTTPHandler(pinned_targets),
        _PinnedHTTPSHandler(pinned_targets),
        _SafeRedirectHandler(
            pinned_targets,
            resolve_dns=resolve_dns,
            allow_fake_ip=allow_fake_ip,
        ),
    )
    req = Request(clean_url, data=data, headers={"User-Agent": DEFAULT_USER_AGENT, **headers}, method=method)
    deadline = time.monotonic() + max(timeout * 2.0, timeout + 2.0)
    with opener.open(req, timeout=timeout) as resp:
        final_url = resp.geturl()
        content_type = resp.headers.get("Content-Type", "text/plain")
        raw_bytes = _read_body_with_deadline(resp, max_bytes, deadline)
        return final_url, content_type, raw_bytes


# 使用钉住 IP 与禁代理 opener 安全发起 HTTP GET（多读 1 字节探测是否触达下载上限）
def _default_http_get(
    url: str,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int,
    *,
    resolve_dns: DnsResolver = socket.getaddrinfo,
    allow_fake_ip: bool = False,
) -> tuple[str, str, bytes]:
    return _open_pinned_request(
        url,
        method="GET",
        headers=headers,
        data=None,
        timeout=timeout,
        max_bytes=max_bytes,
        resolve_dns=resolve_dns,
        allow_fake_ip=allow_fake_ip,
    )


# 尝试将压缩 JSON 响应格式化为多行缩进文本，便于按行阅读与分页检索
def _format_json_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    return json.dumps(parsed, ensure_ascii=False, indent=2)


# 使用钉住 IP 与禁代理 opener 发送 JSON POST 请求（供 Tavily 搜索接口使用）
def _default_http_post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    timeout: float,
    *,
    resolve_dns: DnsResolver = socket.getaddrinfo,
    allow_fake_ip: bool = False,
    max_bytes: int = 2 * 1024 * 1024,
) -> dict[str, object]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    _, _, raw_bytes = _open_pinned_request(
        url,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
        data=body,
        timeout=timeout,
        max_bytes=max_bytes,
        resolve_dns=resolve_dns,
        allow_fake_ip=allow_fake_ip,
    )
    return json.loads(raw_bytes[:max_bytes].decode("utf-8", errors="replace"))


# 抓取公开网页并提取标题与纯文本正文（含 SSRF 校验、DNS Pinning、类型过滤与字节截断）
def execute_fetch_webpage(
    url: str,
    policy: WebPolicy,
    *,
    resolve_dns: DnsResolver = socket.getaddrinfo,
    http_get: HttpGetFn | None = None,
) -> str:
    try:
        safe_url = validate_public_url(
            url,
            resolve_dns=resolve_dns,
            allow_fake_ip=policy.allow_fake_ip,
        )
    except ValueError as error:
        return f"错误：{error}"

    getter = http_get or (
        lambda u, h, t, m: _default_http_get(
            u,
            h,
            t,
            m,
            resolve_dns=resolve_dns,
            allow_fake_ip=policy.allow_fake_ip,
        )
    )
    try:
        final_url, content_type, raw_bytes = getter(
            safe_url,
            {"Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.1"},
            policy.timeout,
            policy.max_download_bytes,
        )
    except HTTPError as error:
        return f"错误：网页请求失败（HTTP {error.code} {error.reason}）。"
    except (URLError, TimeoutError, OSError, ValueError) as error:
        return f"错误：网页抓取失败（{error}）。"

    mime = content_type.split(";", 1)[0].strip().lower()
    if mime and not mime.startswith(ALLOWED_CONTENT_PREFIXES):
        return f"错误：不支持抓取该内容类型（{mime}），仅支持网页与纯文本/JSON。"

    download_truncated = len(raw_bytes) > policy.max_download_bytes
    if download_truncated:
        raw_bytes = raw_bytes[: policy.max_download_bytes]
    trunc_note = (
        f"注意：原始响应超过下载上限（{policy.max_download_bytes} 字节），已仅解析前部内容。\n"
        if download_truncated
        else ""
    )

    text = raw_bytes.decode(_extract_charset(content_type), errors="replace")
    if "html" in mime:
        title, body = extract_text_from_html(text)
        header = f"网页：{final_url}\n" + (f"标题：{title}\n" if title else "") + trunc_note + "\n"
        content = header + (body or "（页面未提取到可见正文）")
    elif "json" in mime:
        content = f"网页：{final_url}\n{trunc_note}\n{_format_json_text(text)}"
    else:
        content = f"网页：{final_url}\n{trunc_note}\n{text.strip()}"

    wrap_chars = min(DEFAULT_SOFT_WRAP_CHARS, max(32, policy.max_output_bytes // 6))
    return truncate_or_offload(
        content,
        policy.max_output_bytes,
        tag="webpage",
        mode="head",
        offload_dir=policy.offload_dir,
        soft_wrap_chars=wrap_chars,
    )


# 解包 DuckDuckGo 跳转链接中的真实目标 URL（提取 uddg 参数）
def _unwrap_ddg_url(raw_href: str) -> str:
    href = unescape(raw_href.strip())
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    uddg = parse_qs(parsed.query).get("uddg")
    if uddg and uddg[0]:
        return unquote(uddg[0])
    return href


# 解析 DuckDuckGo HTML 搜索页，提取真实链接、标题与摘要片段
def parse_duckduckgo_html(html: str, limit: int = 5) -> list[dict[str, str]]:
    anchors = re.findall(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.S | re.I,
    )
    if not anchors:
        # 兼容属性顺序相反的 href / class 写法
        anchors = [
            (href, title)
            for title, href in re.findall(
                r'<a[^>]*href="([^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>',
                html,
                flags=re.S | re.I,
            )
        ]
    snippets = re.findall(
        r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
        html,
        flags=re.S | re.I,
    )
    results: list[dict[str, str]] = []
    for idx, (href, raw_title) in enumerate(anchors[:limit]):
        title = " ".join(unescape(re.sub(r"<[^>]+>", "", raw_title)).split())
        url = _unwrap_ddg_url(href)
        raw_snip = snippets[idx] if idx < len(snippets) else ""
        snippet = " ".join(unescape(re.sub(r"<[^>]+>", "", raw_snip)).split())
        if title or url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


# 解析 Bing 标准 RSS 2.0 搜索结果 XML，返回结构化列表
def parse_bing_rss(xml_text: str, limit: int = 5) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    results: list[dict[str, str]] = []
    for item in root.findall(".//item")[:limit]:
        title = unescape(" ".join((item.findtext("title") or "").split()))
        link = (item.findtext("link") or "").strip()
        snippet = unescape(" ".join((item.findtext("description") or "").split()))
        if title or link:
            results.append({"title": title, "url": link, "snippet": snippet})
    return results


# 将多后端统一结构化的搜索结果格式化为易读编号列表
def _format_search_results(query: str, provider: str, items: list[dict[str, str]], max_bytes: int) -> str:
    if not items:
        return f"未搜索到与「{query}」相关的结果（来源：{provider}）。"
    lines = [f"联网搜索结果（关键词：{query}，引擎：{provider}）："]
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item.get('title') or '（无标题）'}")
        if item.get("url"):
            lines.append(f"   链接：{item['url']}")
        if item.get("snippet"):
            lines.append(f"   摘要：{item['snippet']}")
    return truncate_utf8("\n".join(lines), max_bytes)


# 执行联网搜索：优先 Tavily（TAVILY_API_KEY），次选 Brave（BRAVE_API_KEY），默认免 Key 走 Bing RSS
def execute_web_search(
    query: str,
    policy: WebPolicy,
    *,
    http_get: HttpGetFn | None = None,
    http_post_json: HttpPostJsonFn | None = None,
) -> str:
    clean_query = " ".join((query or "").split())
    if not clean_query:
        return "错误：搜索关键词不能为空。"

    getter = http_get or (
        lambda u, h, t, m: _default_http_get(u, h, t, m, allow_fake_ip=policy.allow_fake_ip)
    )
    poster = http_post_json or (
        lambda u, p, h, t: _default_http_post_json(
            u,
            p,
            h,
            t,
            allow_fake_ip=policy.allow_fake_ip,
            max_bytes=policy.max_download_bytes,
        )
    )
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    brave_key = os.getenv("BRAVE_API_KEY", "").strip()

    try:
        if tavily_key:
            data = poster(
                "https://api.tavily.com/search",
                {"api_key": tavily_key, "query": clean_query, "max_results": policy.max_results},
                {},
                policy.timeout,
            )
            items = [
                {
                    "title": str(r.get("title") or ""),
                    "url": str(r.get("url") or ""),
                    "snippet": " ".join(str(r.get("content") or "").split())[:260],
                }
                for r in (data.get("results") or [])[: policy.max_results]
                if isinstance(r, dict)
            ]
            return _format_search_results(clean_query, "tavily", items, policy.max_output_bytes)

        if brave_key:
            url = f"https://api.search.brave.com/res/v1/web/search?q={quote(clean_query)}&count={policy.max_results}"
            _, _, raw = getter(
                url,
                {"Accept": "application/json", "X-Subscription-Token": brave_key},
                policy.timeout,
                policy.max_download_bytes,
            )
            data = json.loads(raw.decode("utf-8", errors="replace"))
            web_results = (data.get("web") or {}).get("results") or []
            items = [
                {
                    "title": str(r.get("title") or ""),
                    "url": str(r.get("url") or ""),
                    "snippet": " ".join(str(r.get("description") or "").split())[:260],
                }
                for r in web_results[: policy.max_results]
                if isinstance(r, dict)
            ]
            return _format_search_results(clean_query, "brave", items, policy.max_output_bytes)

        # 默认免 Key 模式：优先 DuckDuckGo HTML，失败或无结果时自动降级到 Bing RSS
        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q={quote(clean_query)}"
            _, _, ddg_raw = getter(ddg_url, {}, policy.timeout, policy.max_download_bytes)
            ddg_items = parse_duckduckgo_html(
                ddg_raw.decode("utf-8", errors="replace"),
                limit=policy.max_results,
            )
            if ddg_items:
                return _format_search_results(clean_query, "duckduckgo", ddg_items, policy.max_output_bytes)
        except (HTTPError, URLError, TimeoutError, OSError):
            pass

        rss_url = f"https://www.bing.com/search?format=rss&q={quote(clean_query)}"
        _, _, raw = getter(rss_url, {}, policy.timeout, policy.max_download_bytes)
        items = parse_bing_rss(raw.decode("utf-8", errors="replace"), limit=policy.max_results)
        return _format_search_results(clean_query, "bing_rss", items, policy.max_output_bytes)
    except HTTPError as error:
        return f"错误：联网搜索请求失败（HTTP {error.code} {error.reason}）。"
    except (URLError, TimeoutError, OSError, ET.ParseError, json.JSONDecodeError, ValueError) as error:
        return f"错误：联网搜索失败（{error}）。"


# 插件工具入口：联网搜索实时信息或最新资料，返回标题、链接与摘要
@tool(
    name="web_search",
    description="联网搜索实时信息、最新技术文档或公开网页，返回排名前几条的标题、链接与摘要。",
    parameters={"query": "搜索关键词，例如 'Python 3.14 新特性'"},
    risk="network",
)
def web_search(
    query: str,
    *,
    config_path: Path | None = None,
    web_config: Mapping[str, object] | None = None,
) -> str:
    if isinstance(web_config, Mapping):
        policy = WebPolicy.from_dict(web_config)
    else:
        policy = WebPolicy.from_config(config_path)
    return execute_web_search(query, policy)


# 插件工具入口：抓取指定公开网页（http/https）并提取标题与纯文本正文（超长自动落盘分页）
@tool(
    name="fetch_webpage",
    description=(
        "抓取指定公开 http/https 网页链接并提取标题与纯文本正文，适合阅读文章或核对 web_search 返回的页面细节；"
        "超长正文会自动保存至 data/workspace/tool_outputs/ 供 read_markdown 分页续读。"
    ),
    parameters={"url": "完整的公开 http/https 网址"},
    risk="network",
)
def fetch_webpage(
    url: str,
    *,
    config_path: Path | None = None,
    offload_dir: Path | None = None,
    web_config: Mapping[str, object] | None = None,
) -> str:
    if isinstance(web_config, Mapping):
        policy = WebPolicy.from_dict(web_config, offload_dir=offload_dir)
    else:
        policy = WebPolicy.from_config(config_path, offload_dir=offload_dir)
    return execute_fetch_webpage(url, policy)
