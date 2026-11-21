"""联网搜索与网页正文提取插件（plugins/web_tools.py）单元测试（TDD）。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.memory import MemoryStore
from plugins.fts5_retriever import FTS5Retriever
from plugins.web_tools import (
    WebPolicy,
    execute_fetch_webpage,
    execute_web_search,
    extract_text_from_html,
    parse_bing_rss,
    parse_duckduckgo_html,
    validate_public_url,
)


# 测试插件能通过 config/tools.json 自动加载 web_search 与 fetch_webpage，且可由白名单随时开关
def test_web_plugin_loads_via_tools_config_and_can_be_disabled(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = ToolsConfig.load(repo_root / "config" / "tools.json", project_root=repo_root)
    assert "web_search" in cfg.enabled_tools
    assert "fetch_webpage" in cfg.enabled_tools
    registry = cfg.build_registry()
    assert "web_search" in registry.names
    assert "fetch_webpage" in registry.names

    # 从 enabled_tools 移除后立即关闭
    off_cfg = tmp_path / "tools.json"
    off_cfg.write_text(
        json.dumps({"enabled_tools": ["get_current_time"], "plugins": ["plugins"], "commands": []}),
        encoding="utf-8",
    )
    off_reg = ToolsConfig.load(off_cfg, project_root=repo_root).build_registry()
    assert "web_search" not in off_reg.names
    assert "fetch_webpage" not in off_reg.names


# 测试 SSRF 护栏：拦截非 http(s) 协议、本地主机名以及解析到私网/回环/云元数据地址的域名
def test_validate_public_url_blocks_ssrf_and_non_http_schemes() -> None:
    # 构造返回固定 IP 的假 DNS 解析器
    def make_resolver(ip: str):
        # 返回与 socket.getaddrinfo 兼容的元组列表
        def _resolve(host: str, port: object, *args: object):
            return [(0, 0, 0, "", (ip, 0))]

        return _resolve

    for bad_url in (
        "",
        "file:///etc/passwd",
        "ftp://example.com/a.txt",
        "javascript:alert(1)",
        "http:///etc/passwd",
        "http://localhost:8765/api/status",
        "http://mybox.local/secret",
        "https://service.internal/admin",
    ):
        with pytest.raises(ValueError):
            validate_public_url(bad_url, resolve_dns=make_resolver("93.184.216.34"))

    # 域名解析到任意非公网 IP 时必须拦截
    for private_ip in (
        "127.0.0.1",
        "10.0.0.8",
        "172.16.0.1",
        "192.168.1.10",
        "169.254.169.254",
        "::1",
        "fc00::1",
        "fe80::1",
        "0.0.0.0",
    ):
        with pytest.raises(ValueError, match="公网"):
            validate_public_url("https://evil.example.com/", resolve_dns=make_resolver(private_ip))

    assert validate_public_url("https://example.com/docs", resolve_dns=make_resolver("93.184.216.34")) == (
        "https://example.com/docs"
    )
    # 兼容 macOS TUN 代理 Fake-IP 网段（仅允许公开域名 DNS 解析到 198.18.0.0/15，严禁字面 IP 直连）
    assert validate_public_url("https://docs.python.org/", resolve_dns=make_resolver("198.18.1.170")) == (
        "https://docs.python.org/"
    )
    with pytest.raises(ValueError, match="公网"):
        validate_public_url("http://198.18.1.170/", resolve_dns=make_resolver("198.18.1.170"))


# 测试 HTML 正文提取：保留标题和正文段落，剔除 script/style/nav/footer 噪声
def test_extract_text_from_html_strips_scripts_styles_and_extracts_title() -> None:
    html = """
    <html>
      <head>
        <title>测试文章标题 &amp; 副标题</title>
        <style>body { color: red; }</style>
        <script>console.log("secret_script");</script>
      </head>
      <body>
        <nav>首页 | 登录 | 广告</nav>
        <main>
          <h1>核心结论</h1>
          <p>Python 3.12 支持零依赖构建个人 Agent。</p>
          <p>标准库足够轻量。</p>
        </main>
        <footer>版权所有 2026</footer>
      </body>
    </html>
    """
    title, text = extract_text_from_html(html)
    assert title == "测试文章标题 & 副标题"
    assert "核心结论" in text
    assert "Python 3.12 支持零依赖构建个人 Agent。" in text
    assert "secret_script" not in text
    assert "color: red" not in text
    assert "首页 | 登录" not in text
    assert "版权所有" not in text


# 测试 fetch_webpage 的正常抓取、二进制拦截、TOAST 卸载保护与私网重定向拦截
def test_fetch_webpage_parses_html_enforces_limits_and_blocks_binary(tmp_path: Path) -> None:
    offload_dir = tmp_path / "workspace" / "tool_outputs"
    policy = WebPolicy(timeout=5.0, max_download_bytes=4096, max_output_bytes=200, offload_dir=offload_dir)

    # 固定公网 DNS
    def public_dns(host: str, port: object, *args: object):
        return [(0, 0, 0, "", ("93.184.216.34", 0))]

    # 1. HTML 正常抓取与输出截断
    long_html = "<html><head><title>示例</title></head><body><p>" + ("正文内容" * 80) + "</p></body></html>"
    out_html = execute_fetch_webpage(
        "https://example.com/article",
        policy,
        resolve_dns=public_dns,
        http_get=lambda url, headers, timeout, max_bytes: (
            url,
            "text/html; charset=utf-8",
            long_html.encode("utf-8"),
        ),
    )
    assert "标题：示例" in out_html
    assert "正文内容" in out_html
    assert "输出已截断" in out_html
    assert "data/workspace/tool_outputs/" in out_html

    # 2. 二进制文件（如 PDF/图片）应明确拒绝
    out_bin = execute_fetch_webpage(
        "https://example.com/file.pdf",
        policy,
        resolve_dns=public_dns,
        http_get=lambda url, headers, timeout, max_bytes: (url, "application/pdf", b"%PDF-1.7"),
    )
    assert out_bin.startswith("错误：")
    assert "不支持" in out_bin

    # 3. 尝试抓取本机回环地址应直接拦截（绝不发网络请求）
    called = {"flag": False}

    # 若被调用说明 SSRF 拦截失效
    def should_not_run(*args: object):
        called["flag"] = True
        return ("", "text/plain", b"")

    out_ssrf = execute_fetch_webpage(
        "http://127.0.0.1:8765/api/status",
        policy,
        http_get=should_not_run,
    )
    assert out_ssrf.startswith("错误：")
    assert called["flag"] is False


# 测试 fetch_webpage 的 TOAST 卸载分页续读、2MB 下载上限与超限显式提醒、以及 JSON 自动美化
def test_fetch_webpage_toast_offload_download_cap_notice_and_json_pretty(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_policy = WebPolicy.from_config(repo_root / "config" / "tools.json")
    assert default_policy.max_download_bytes >= 2 * 1024 * 1024

    offload_dir = tmp_path / "workspace" / "tool_outputs"
    policy = WebPolicy(timeout=5.0, max_download_bytes=64 * 1024, max_output_bytes=450, offload_dir=offload_dir)

    # 固定公网 DNS
    def public_dns(host: str, port: object, *args: object):
        return [(0, 0, 0, "", ("93.184.216.34", 0))]

    # 1. 长文章触发 TOAST 卸载：主输出保留开头预览，后半段可通过 read_markdown 与 search_markdown 完整读取
    paragraphs = [f"<p>第 {i:02d} 段：分布式存储与一致性协议的详细工程说明。</p>" for i in range(1, 35)]
    paragraphs[29] = "<p>第 30 段：隐藏在文章尾部的核心基准指标 TOAST_ARTICLE_MARKER_881。</p>"
    article_html = f"<html><head><title>长文测试</title></head><body>{''.join(paragraphs)}</body></html>"
    out = execute_fetch_webpage(
        "https://example.com/long-post",
        policy,
        resolve_dns=public_dns,
        http_get=lambda url, headers, timeout, max_bytes: (
            url,
            "text/html; charset=utf-8",
            article_html.encode("utf-8"),
        ),
    )
    assert "标题：长文测试" in out
    assert "输出已截断" in out
    assert "TOAST_ARTICLE_MARKER_881" not in out
    path_match = re.search(r"(data/workspace/tool_outputs/[A-Za-z0-9._-]+\.txt)", out)
    next_match = re.search(r"start_line=(\d+)", out)
    assert path_match is not None and next_match is not None

    memory = MemoryStore(tmp_path, retriever=FTS5Retriever())
    rest = memory.read_markdown(path_match.group(1), start_line=int(next_match.group(1)), end_line=100)
    assert "TOAST_ARTICLE_MARKER_881" in rest
    hits = memory.search_markdown("TOAST_ARTICLE_MARKER_881", root="data/workspace/tool_outputs")
    assert path_match.group(1) in hits

    # 2. 原始响应超过 max_download_bytes 时必须显式提示源码已截断（不再静默截半）
    small_cap_policy = WebPolicy(timeout=5.0, max_download_bytes=120, max_output_bytes=4096, offload_dir=offload_dir)
    oversized_html = ("<html><body><p>" + ("超长源码" * 60) + "</p></body></html>").encode("utf-8")
    out_dl_cap = execute_fetch_webpage(
        "https://example.com/huge",
        small_cap_policy,
        resolve_dns=public_dns,
        http_get=lambda url, headers, timeout, max_bytes: (url, "text/html; charset=utf-8", oversized_html),
    )
    assert "下载上限" in out_dl_cap

    # 3. 压缩单行 JSON 响应自动缩进换行，便于按行阅读与分页
    minified_json = json.dumps(
        {"items": [{"id": idx, "name": f"entry_{idx}", "detail": "x" * 20} for idx in range(25)]},
        separators=(",", ":"),
    ).encode("utf-8")
    out_json = execute_fetch_webpage(
        "https://api.example.com/data",
        policy,
        resolve_dns=public_dns,
        http_get=lambda url, headers, timeout, max_bytes: (url, "application/json", minified_json),
    )
    assert "输出已截断" in out_json
    json_path_match = re.search(r"(data/workspace/tool_outputs/[A-Za-z0-9._-]+\.txt)", out_json)
    assert json_path_match is not None
    saved_json_lines = (tmp_path / "workspace" / "tool_outputs" / Path(json_path_match.group(1)).name).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(saved_json_lines) > 50
    assert any('"name": "entry_24"' in line for line in saved_json_lines)


# 测试 web_search：DuckDuckGo HTML 解包与优先搜索、Bing RSS 兜底、Tavily/Brave 切换以及空查询拦截
def test_web_search_supports_duckduckgo_bing_tavily_and_brave(monkeypatch) -> None:
    ddg_html = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Fwhatsnew%2F3.14.html&amp;rut=abc">
        What&#x27;s new in Python 3.14
      </a>
      <a class="result__snippet">Template strings (PEP 750) and deferred annotations.</a>
    </div>
    """
    ddg_items = parse_duckduckgo_html(ddg_html, limit=5)
    assert len(ddg_items) == 1
    assert ddg_items[0]["title"] == "What's new in Python 3.14"
    assert ddg_items[0]["url"] == "https://docs.python.org/3/whatsnew/3.14.html"
    assert "PEP 750" in ddg_items[0]["snippet"]

    rss_xml = """<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Python 官方文档</title>
        <link>https://docs.python.org/3/</link>
        <description>Python 标准库与语言参考。</description>
      </item>
      <item>
        <title>PEP 701</title>
        <link>https://peps.python.org/pep-0701/</link>
        <description>f-string 语法增强说明。</description>
      </item>
    </channel></rss>
    """
    parsed = parse_bing_rss(rss_xml, limit=5)
    assert len(parsed) == 2
    assert parsed[0]["title"] == "Python 官方文档"
    assert parsed[0]["url"] == "https://docs.python.org/3/"

    policy = WebPolicy(max_results=5)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    # 默认免 Key 模式优先走 DuckDuckGo HTML
    out_ddg = execute_web_search(
        "Python 3.14",
        policy,
        http_get=lambda url, headers, timeout, max_bytes: (url, "text/html", ddg_html.encode("utf-8")),
    )
    assert "What's new in Python 3.14" in out_ddg
    assert "https://docs.python.org/3/whatsnew/3.14.html" in out_ddg

    # DuckDuckGo 返回空结果时自动降级到 Bing RSS
    out_fallback = execute_web_search(
        "Python 3.12",
        policy,
        http_get=lambda url, headers, timeout, max_bytes: (
            (url, "text/html", b"<html></html>")
            if "duckduckgo" in url
            else (url, "application/xml", rss_xml.encode("utf-8"))
        ),
    )
    assert "Python 官方文档" in out_fallback
    assert "https://docs.python.org/3/" in out_fallback

    # 配置 TAVILY_API_KEY 时自动走 Tavily JSON API
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    out_tavily = execute_web_search(
        "ReAct agent",
        policy,
        http_post_json=lambda url, payload, headers, timeout: {
            "results": [
                {"title": "ReAct Paper", "url": "https://arxiv.org/abs/2210.03629", "content": "Synergizing reasoning and acting."}
            ]
        },
    )
    assert "ReAct Paper" in out_tavily
    assert "arxiv.org" in out_tavily

    # 配置 BRAVE_API_KEY（无 Tavily）时走 Brave JSON API
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test")
    out_brave = execute_web_search(
        "SQLite WAL",
        policy,
        http_get=lambda url, headers, timeout, max_bytes: (
            url,
            "application/json",
            json.dumps(
                {
                    "web": {
                        "results": [
                            {"title": "WAL Mode", "url": "https://sqlite.org/wal.html", "description": "Write-Ahead Logging."}
                        ]
                    }
                }
            ).encode("utf-8"),
        ),
    )
    assert "WAL Mode" in out_brave
    assert "sqlite.org/wal.html" in out_brave

    # 空查询报错
    assert execute_web_search("   ", policy).startswith("错误：")
