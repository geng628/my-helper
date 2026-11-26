"""联网搜索与网页正文提取插件（零第三方依赖）：支持防 SSRF DNS Pinning 直连抓网页与多后端联网搜索。"""
from __future__ import annotations

import codecs
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse, urlsplit

from personal_assistant.agent.tools import tool
from personal_assistant.assistant.common import DEFAULT_SOFT_WRAP_CHARS, truncate_or_offload, truncate_utf8

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
LOCAL_HOST_SUFFIXES = (".local", ".localhost", ".internal")
FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")
MAX_REDIRECTS = 5
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
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

    # 从主板注入的 web 配置字典直接构建 WebPolicy（缺省使用安全默认值）
    @classmethod
    def from_dict(cls, web_cfg: Mapping[str, object] | None = None, *, offload_dir: Path | None = None) -> "WebPolicy":
        cfg = web_cfg or {}
        raw_timeout = cfg.get("timeout", 15.0)
        raw_dl = cfg.get("max_download_bytes", 2 * 1024 * 1024)
        raw_out = cfg.get("max_output_bytes", 8192)
        raw_res = cfg.get("max_results", 5)
        timeout = float(raw_timeout) if isinstance(raw_timeout, (int, float, str)) and not isinstance(raw_timeout, bool) else 15.0
        max_download = int(raw_dl) if isinstance(raw_dl, (int, float, str)) and not isinstance(raw_dl, bool) else 2 * 1024 * 1024
        max_output = int(raw_out) if isinstance(raw_out, (int, float, str)) and not isinstance(raw_out, bool) else 8192
        max_results = int(raw_res) if isinstance(raw_res, (int, float, str)) and not isinstance(raw_res, bool) else 5
        return cls(
            timeout=timeout,
            max_download_bytes=max_download,
            max_output_bytes=max_output,
            max_results=max(1, min(10, max_results)),
            offload_dir=offload_dir,
            allow_fake_ip=bool(cfg.get("allow_fake_ip", False)),
        )


# 判断单个 IPv4/IPv6 地址是否属于允许访问的公网地址（阻断私网、回环、多播、NAT64 与隧道地址）
def _is_public_ip(
    ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address, *, is_literal_ip: bool = False, allow_fake_ip: bool = False
) -> bool:
    if isinstance(ip_obj, ipaddress.IPv4Address):
        if allow_fake_ip and not is_literal_ip and ip_obj in FAKE_IP_NETWORK:
            return True
        return ip_obj.is_global and not ip_obj.is_multicast
    if ip_obj.ipv4_mapped is not None:
        return _is_public_ip(ip_obj.ipv4_mapped, is_literal_ip=is_literal_ip, allow_fake_ip=allow_fake_ip)
    return (
        ip_obj.is_global
        and not ip_obj.is_multicast
        and not ip_obj.is_reserved
        and ip_obj.sixtofour is None
        and ip_obj.teredo is None
        and not ip_obj.compressed.lower().startswith("64:ff9b:")
    )


# 严格校验并解析公开 http/https URL，返回 (clean_url, hostname, port, pinned_ip)
def resolve_and_validate_public_url(
    url: str, *, resolve_dns: DnsResolver = socket.getaddrinfo, allow_fake_ip: bool = False
) -> tuple[str, str, int, str]:
    clean = (url or "").strip()
    if not clean or any(ch in clean for ch in ("\x00", "\r", "\n")):
        raise ValueError("URL 不能为空且不能包含换行符。")
    parsed = urlparse(clean)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"只允许 http 或 https 协议，已拒绝：{parsed.scheme or '空协议'}")
    host = unicodedata.normalize("NFKC", (parsed.hostname or "").strip()).lower()
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
        ips.append(str(ipaddress.ip_address(host.split("%", 1)[0])))
        is_literal_ip = True
    except ValueError:
        try:
            infos = resolve_dns(host, port)
        except OSError as error:
            raise ValueError(f"域名解析失败（{host}）：{error}") from error
        for info in infos:
            sockaddr = info[4]
            if isinstance(sockaddr, tuple) and sockaddr:
                raw_addr = str(sockaddr[0]).split("%", 1)[0]
                if raw_addr not in ips:
                    ips.append(raw_addr)

    if not ips:
        raise ValueError(f"未能解析出有效 IP 地址：{host}")
    normalized_ips: list[str] = []
    for raw_ip in ips:
        ip_obj = ipaddress.ip_address(raw_ip)
        if not _is_public_ip(ip_obj, is_literal_ip=is_literal_ip, allow_fake_ip=allow_fake_ip):
            raise ValueError(f"禁止访问非公网地址（SSRF 防护）：{host} -> {ip_obj}")
        normalized_ips.append(str(ip_obj))
    return clean, host, port, normalized_ips[0]


# 严格校验目标 URL 仅限公开 http/https 地址，防止私网或云元数据 SSRF
def validate_public_url(
    url: str, *, resolve_dns: DnsResolver = socket.getaddrinfo, allow_fake_ip: bool = False
) -> str:
    return resolve_and_validate_public_url(url, resolve_dns=resolve_dns, allow_fake_ip=allow_fake_ip)[0]


# 基于标准库 HTMLParser 的轻量网页标题与可见正文提取器
class _HTMLTextExtractor(HTMLParser):
    # 初始化标签跳过深度、标题状态与文本缓冲
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    # 遇到起始标签时更新忽略深度、标题状态或插入段落换行
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in SKIP_HTML_TAGS:
            self._skip_depth += 1
        elif lower == "title" and self._skip_depth == 0:
            self._in_title = True
        elif lower in BLOCK_HTML_TAGS and self._skip_depth == 0:
            self.text_parts.append("\n")

    # 遇到闭合标签时恢复嵌套状态或追加段落换行
    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in SKIP_HTML_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif lower == "title":
            self._in_title = False
        elif lower in BLOCK_HTML_TAGS and self._skip_depth == 0:
            self.text_parts.append("\n")

    # 收集可见文本节点（忽略脚本、样式与页眉页脚区域）
    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            (self.title_parts if self._in_title else self.text_parts).append(data)


# 从 HTML 字符串提取干净的页面标题与纯文本正文
def extract_text_from_html(html: str) -> tuple[str, str]:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    title = " ".join("".join(parser.title_parts).split())
    lines = [" ".join(line.split()) for line in "".join(parser.text_parts).splitlines()]
    return title, "\n".join(line for line in lines if line)


# 从 Content-Type 提取并校验合法文本字符编码，未知或伪编码安全回退 utf-8
def _extract_charset(content_type: str) -> str:
    match = re.search(r"charset\s*=\s*\"?([A-Za-z0-9._-]+)\"?", content_type, flags=re.IGNORECASE)
    if not match:
        return "utf-8"
    try:
        info = codecs.lookup(match.group(1))
    except LookupError:
        return "utf-8"
    return info.name if getattr(info, "_is_text_encoding", True) else "utf-8"


# 在总截止时间与最大字节数约束内分块读取响应体，防御慢速滴水挂起
def _read_body_with_deadline(resp: object, max_bytes: int, deadline: float) -> bytes:
    chunks: list[bytes] = []
    total, target = 0, max_bytes + 1
    read1 = getattr(resp, "read1", None)
    reader = read1 if callable(read1) else getattr(resp, "read")
    while total < target:
        if time.monotonic() > deadline:
            raise TimeoutError("响应读取超过总时长限制。")
        chunk = reader(min(16384, target - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if time.monotonic() > deadline:
        raise TimeoutError("响应读取超过总时长限制。")
    return b"".join(chunks)


# 直连已校验的 pinned_ip 套接字并构造 HTTP/HTTPS 连接（忽略系统代理、消除二次 DNS 解析并复用 TLS 上下文）
def _connect_pinned(
    scheme: str, host: str, port: int, pinned_ip: str, timeout: float, ssl_context: Any = None
) -> http.client.HTTPConnection:
    sock = socket.create_connection((pinned_ip, port), timeout=timeout)
    try:
        if scheme == "https":
            ctx = ssl_context if ssl_context is not None else ssl.create_default_context()
            sock = ctx.wrap_socket(sock, server_hostname=host)
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port, timeout=timeout, context=ctx if isinstance(ctx, ssl.SSLContext) else None
            )
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.sock = sock
        return conn
    except BaseException:
        sock.close()
        raise


# 基于 http.client 直连钉住 IP 发起安全 GET 请求（逐跳校验重定向且跨源自动剥离敏感头）
def _default_http_get(
    url: str,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int,
    *,
    resolve_dns: DnsResolver = socket.getaddrinfo,
    allow_fake_ip: bool = False,
    ssl_context: Any = None,
) -> tuple[str, str, bytes]:
    deadline = time.monotonic() + timeout
    req_headers = {"User-Agent": DEFAULT_USER_AGENT, **headers}
    current_url = url
    prev_origin: tuple[str, str, int] | None = None

    for hop in range(MAX_REDIRECTS + 1):
        step_timeout = deadline - time.monotonic()
        if step_timeout <= 0:
            raise TimeoutError("网页请求超过总时长限制。")
        clean_url, host, port, pinned_ip = resolve_and_validate_public_url(
            current_url, resolve_dns=resolve_dns, allow_fake_ip=allow_fake_ip
        )
        parts = urlsplit(clean_url)
        scheme = parts.scheme.lower()
        origin = (scheme, host, port)
        if prev_origin is not None and origin != prev_origin:
            req_headers = {k: v for k, v in req_headers.items() if k.lower() not in SENSITIVE_REDIRECT_HEADERS}
        prev_origin = origin

        target_path = f"{parts.path or '/'}?{parts.query}" if parts.query else (parts.path or "/")
        conn = _connect_pinned(scheme, host, port, pinned_ip, step_timeout, ssl_context=ssl_context)
        resp: http.client.HTTPResponse | None = None
        try:
            conn.request("GET", target_path, headers=req_headers)
            resp = conn.getresponse()
            if resp.status in REDIRECT_STATUSES:
                loc = (resp.getheader("Location") or "").strip()
                if not loc:
                    raise ValueError("重定向响应缺少 Location 头。")
                if hop >= MAX_REDIRECTS:
                    raise ValueError(f"重定向次数超过上限（{MAX_REDIRECTS}）。")
                current_url = urljoin(clean_url, loc)
                continue
            if resp.status >= 400:
                raise HTTPError(clean_url, resp.status, resp.reason, resp.headers, None)
            return clean_url, resp.getheader("Content-Type") or "text/plain", _read_body_with_deadline(resp, max_bytes, deadline)
        finally:
            if resp is not None:
                resp.close()
            conn.close()
    raise ValueError(f"重定向次数超过上限（{MAX_REDIRECTS}）。")


# 尝试将压缩 JSON 响应格式化为多行缩进文本，便于分页阅读
def _format_json_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        return json.dumps(json.loads(stripped), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return stripped


# 基于 http.client 直连钉住 IP 发送 JSON POST 请求（供 Tavily 搜索接口使用）
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
    deadline = time.monotonic() + timeout
    clean_url, host, port, pinned_ip = resolve_and_validate_public_url(
        url, resolve_dns=resolve_dns, allow_fake_ip=allow_fake_ip
    )
    parts = urlsplit(clean_url)
    target_path = f"{parts.path or '/'}?{parts.query}" if parts.query else (parts.path or "/")
    conn = _connect_pinned(parts.scheme.lower(), host, port, pinned_ip, max(0.001, deadline - time.monotonic()))
    resp: http.client.HTTPResponse | None = None
    try:
        conn.request(
            "POST",
            target_path,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"User-Agent": DEFAULT_USER_AGENT, "Content-Type": "application/json", **headers},
        )
        resp = conn.getresponse()
        if resp.status >= 400:
            raise HTTPError(clean_url, resp.status, resp.reason, resp.headers, None)
        raw_bytes = _read_body_with_deadline(resp, max_bytes, deadline)
    finally:
        if resp is not None:
            resp.close()
        conn.close()
    return json.loads(raw_bytes[:max_bytes].decode("utf-8", errors="replace"))


# 抓取公开网页并提取标题与纯文本正文（含 SSRF 校验、DNS Pinning、类型过滤与 TOAST 卸载）
def execute_fetch_webpage(
    url: str, policy: WebPolicy, *, resolve_dns: DnsResolver = socket.getaddrinfo, http_get: HttpGetFn | None = None
) -> str:
    if http_get is not None:
        try:
            safe_url = validate_public_url(url, resolve_dns=resolve_dns, allow_fake_ip=policy.allow_fake_ip)
        except ValueError as error:
            return f"错误：{error}"
        getter = http_get
    else:
        safe_url = (url or "").strip()
        getter = lambda u, h, t, m: _default_http_get(
            u, h, t, m, resolve_dns=resolve_dns, allow_fake_ip=policy.allow_fake_ip
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
    except (URLError, http.client.HTTPException, TimeoutError, OSError, ValueError) as error:
        return f"错误：网页抓取失败（{error}）。"

    mime = content_type.split(";", 1)[0].strip().lower()
    if mime and not mime.startswith(ALLOWED_CONTENT_PREFIXES):
        return f"错误：不支持抓取该内容类型（{mime}），仅支持网页与纯文本/JSON。"

    download_truncated = len(raw_bytes) > policy.max_download_bytes
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
        content, policy.max_output_bytes, tag="webpage", mode="head", offload_dir=policy.offload_dir, soft_wrap_chars=wrap_chars
    )


# 解包 DuckDuckGo 跳转链接中的真实目标 URL（提取 uddg 参数）
def _unwrap_ddg_url(raw_href: str) -> str:
    href = unescape(raw_href.strip())
    if href.startswith("//"):
        href = "https:" + href
    uddg = parse_qs(urlparse(href).query).get("uddg")
    return unquote(uddg[0]) if uddg and uddg[0] else href


# 解析 DuckDuckGo HTML 搜索页，提取真实链接、标题与摘要片段
def parse_duckduckgo_html(html: str, limit: int = 5) -> list[dict[str, str]]:
    anchors = re.findall(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.S | re.I) or [
        (href, title)
        for title, href in re.findall(
            r'<a[^>]*href="([^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', html, flags=re.S | re.I
        )
    ]
    snippets = re.findall(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>', html, flags=re.S | re.I)
    results: list[dict[str, str]] = []
    for idx, (href, raw_title) in enumerate(anchors[:limit]):
        title = " ".join(unescape(re.sub(r"<[^>]+>", "", raw_title)).split())
        url = _unwrap_ddg_url(href)
        snippet = " ".join(unescape(re.sub(r"<[^>]+>", "", snippets[idx] if idx < len(snippets) else "")).split())
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


# 从 JSON 搜索结果字典列表提取统一结构化条目
def _extract_json_items(records: object, snippet_key: str, limit: int) -> list[dict[str, str]]:
    if not isinstance(records, list):
        return []
    return [
        {
            "title": str(r.get("title") or ""),
            "url": str(r.get("url") or ""),
            "snippet": " ".join(str(r.get(snippet_key) or "").split())[:260],
        }
        for r in records[:limit]
        if isinstance(r, dict)
    ]


# 执行联网搜索：优先 Tavily，次选 Brave，默认免 Key 走 DuckDuckGo 与 Bing RSS
def execute_web_search(
    query: str, policy: WebPolicy, *, http_get: HttpGetFn | None = None, http_post_json: HttpPostJsonFn | None = None
) -> str:
    clean_query = " ".join((query or "").split())
    if not clean_query:
        return "错误：搜索关键词不能为空。"

    getter = http_get or (lambda u, h, t, m: _default_http_get(u, h, t, m, allow_fake_ip=policy.allow_fake_ip))
    poster = http_post_json or (
        lambda u, p, h, t: _default_http_post_json(
            u, p, h, t, allow_fake_ip=policy.allow_fake_ip, max_bytes=policy.max_download_bytes
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
            items = _extract_json_items(data.get("results"), "content", policy.max_results)
            return _format_search_results(clean_query, "tavily", items, policy.max_output_bytes)

        if brave_key:
            url = f"https://api.search.brave.com/res/v1/web/search?q={quote(clean_query)}&count={policy.max_results}"
            _, _, raw = getter(
                url, {"Accept": "application/json", "X-Subscription-Token": brave_key}, policy.timeout, policy.max_download_bytes
            )
            data = json.loads(raw.decode("utf-8", errors="replace"))
            items = _extract_json_items((data.get("web") or {}).get("results"), "description", policy.max_results)
            return _format_search_results(clean_query, "brave", items, policy.max_output_bytes)

        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q={quote(clean_query)}"
            _, _, ddg_raw = getter(ddg_url, {}, policy.timeout, policy.max_download_bytes)
            ddg_items = parse_duckduckgo_html(ddg_raw.decode("utf-8", errors="replace"), limit=policy.max_results)
            if ddg_items:
                return _format_search_results(clean_query, "duckduckgo", ddg_items, policy.max_output_bytes)
        except (HTTPError, URLError, http.client.HTTPException, TimeoutError, OSError, ValueError):
            pass

        rss_url = f"https://www.bing.com/search?format=rss&q={quote(clean_query)}"
        _, _, raw = getter(rss_url, {}, policy.timeout, policy.max_download_bytes)
        items = parse_bing_rss(raw.decode("utf-8", errors="replace"), limit=policy.max_results)
        return _format_search_results(clean_query, "bing_rss", items, policy.max_output_bytes)
    except HTTPError as error:
        return f"错误：联网搜索请求失败（HTTP {error.code} {error.reason}）。"
    except (URLError, http.client.HTTPException, TimeoutError, OSError, ET.ParseError, json.JSONDecodeError, ValueError) as error:
        return f"错误：联网搜索失败（{error}）。"


# 插件工具入口：联网搜索实时信息或最新资料，返回标题、链接与摘要
@tool(
    name="web_search",
    description="联网搜索实时信息、最新技术文档或公开网页，返回排名前几条的标题、链接与摘要。",
    parameters={"query": "搜索关键词，例如 'Python 3.14 新特性'"},
    risk="network",
)
def web_search(query: str, *, web_config: Mapping[str, object] | None = None) -> str:
    return execute_web_search(query, WebPolicy.from_dict(web_config))


# 插件工具入口：抓取指定公开网页并提取标题与纯文本正文（超长自动落盘分页）
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
    url: str, *, offload_dir: Path | None = None, web_config: Mapping[str, object] | None = None
) -> str:
    return execute_fetch_webpage(url, WebPolicy.from_dict(web_config, offload_dir=offload_dir))
