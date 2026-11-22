"""联网搜索与网页正文提取插件（零第三方依赖）：支持防 SSRF 抓网页与多后端联网搜索。"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from personal_assistant.agent.tools import tool
from personal_assistant.assistant.common import (
    DEFAULT_SOFT_WRAP_CHARS,
    truncate_or_offload,
    truncate_utf8,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
LOCAL_HOST_SUFFIXES = (".local", ".localhost", ".internal")
FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")
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

    # 从已解析的 web 配置字典直接构建 WebPolicy（支持绑定实例专属 offload_dir）
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


# 严格校验目标 URL 仅限公开 http/https 地址，防止通过私网/回环/云元数据 IP 触发 SSRF
def validate_public_url(
    url: str,
    *,
    resolve_dns: DnsResolver = socket.getaddrinfo,
) -> str:
    clean = (url or "").strip()
    if not clean or any(ch in clean for ch in ("\x00", "\r", "\n")):
        raise ValueError("URL 不能为空且不能包含换行符。")
    parsed = urlparse(clean)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"只允许 http 或 https 协议，已拒绝：{parsed.scheme or '空协议'}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("URL 缺少有效主机名。")
    if host == "localhost" or host.endswith(LOCAL_HOST_SUFFIXES):
        raise ValueError(f"禁止访问本机或内网域名：{host}")

    # 直接检查字面 IP 或 DNS 解析后的全部目标 IP，必须为公网地址（域名解析兼容 TUN 代理 Fake-IP 网段）
    ips: list[str] = []
    is_literal_ip = False
    try:
        ipaddress.ip_address(host)
        ips.append(host)
        is_literal_ip = True
    except ValueError:
        try:
            infos = resolve_dns(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        except OSError as error:
            raise ValueError(f"域名解析失败（{host}）：{error}") from error
        for info in infos:
            sockaddr = info[4]
            if isinstance(sockaddr, tuple) and sockaddr:
                ips.append(str(sockaddr[0]))

    if not ips:
        raise ValueError(f"未能解析出有效 IP 地址：{host}")
    for raw_ip in ips:
        ip_obj = ipaddress.ip_address(raw_ip.split("%", 1)[0])
        allow_fake_ip = (
            not is_literal_ip
            and isinstance(ip_obj, ipaddress.IPv4Address)
            and ip_obj in FAKE_IP_NETWORK
        )
        if not ip_obj.is_global and not allow_fake_ip:
            raise ValueError(f"禁止访问非公网地址（SSRF 防护）：{host} -> {ip_obj}")
    return clean


# 安全的 HTTP 重定向处理器：对每一跳 301/302/307/308 目标都执行公网地址校验
class _SafeRedirectHandler(HTTPRedirectHandler):
    # 保存自定义 DNS 解析器用于校验重定向目标
    def __init__(self, resolve_dns: DnsResolver = socket.getaddrinfo) -> None:
        super().__init__()
        self._resolve_dns = resolve_dns

    # 重定向前校验跳转 URL 必须为公网 http/https，拦截公网跳内网绕过
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        target = urljoin(req.full_url, newurl)
        validate_public_url(target, resolve_dns=self._resolve_dns)
        return super().redirect_request(req, fp, code, msg, headers, target)


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


# 从 Content-Type 响应头提取字符编码，缺省回退 utf-8
def _extract_charset(content_type: str) -> str:
    match = re.search(r"charset=([A-Za-z0-9._-]+)", content_type, flags=re.IGNORECASE)
    return match.group(1) if match else "utf-8"


# 使用标准库 urllib 安全发起 HTTP GET（校验重定向，多读 1 字节用于探测是否触达下载上限）
def _default_http_get(
    url: str,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int,
    *,
    resolve_dns: DnsResolver = socket.getaddrinfo,
) -> tuple[str, str, bytes]:
    opener = build_opener(_SafeRedirectHandler(resolve_dns=resolve_dns))
    req = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT, **headers}, method="GET")
    with opener.open(req, timeout=timeout) as resp:
        final_url = resp.geturl()
        content_type = resp.headers.get("Content-Type", "text/plain")
        data = resp.read(max_bytes + 1)
        return final_url, content_type, data


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


# 使用标准库 urllib 发送 JSON POST 请求（供 Tavily 搜索接口使用）
def _default_http_post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, object]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": DEFAULT_USER_AGENT, **headers},
        method="POST",
    )
    opener = build_opener(_SafeRedirectHandler())
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# 抓取公开网页并提取标题与纯文本正文（含 SSRF 校验、类型过滤与字节截断）
def execute_fetch_webpage(
    url: str,
    policy: WebPolicy,
    *,
    resolve_dns: DnsResolver = socket.getaddrinfo,
    http_get: HttpGetFn | None = None,
) -> str:
    try:
        safe_url = validate_public_url(url, resolve_dns=resolve_dns)
    except ValueError as error:
        return f"错误：{error}"

    getter = http_get or (
        lambda u, h, t, m: _default_http_get(u, h, t, m, resolve_dns=resolve_dns)
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

    getter = http_get or _default_http_get
    poster = http_post_json or _default_http_post_json
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


# 插件工具入口：联网搜索实时信息或最新资料，返回标题、链接与摘要（支持由 build_tool 自动注入 config_path）
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
) -> str:
    return execute_web_search(query, WebPolicy.from_config(config_path))


# 插件工具入口：抓取指定公开网页（http/https）并提取标题与纯文本正文（支持由 build_tool 自动注入 config_path / offload_dir）
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
) -> str:
    return execute_fetch_webpage(url, WebPolicy.from_config(config_path, offload_dir=offload_dir))
