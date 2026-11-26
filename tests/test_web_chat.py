"""本地网页对话脚本（scripts/web_chat.py）单元测试（TDD）。"""
from __future__ import annotations

import http.client
import json
import socket
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest

from personal_assistant.assistant.app import AssistantApp
from personal_assistant.ai import LLMClient, TokenUsage
from scripts.web_chat import create_web_server


# 用于网页接口集成测试的脚本化假模型
class WebTestLLM(LLMClient):
    # 保存预设回复队列并初始化 last_usage
    def __init__(self, responses: list[tuple[dict[str, object], TokenUsage | None]]) -> None:
        self._responses = list(responses)
        self.last_usage: TokenUsage | None = None

    # 弹出下一条预设回复并更新 last_usage
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        reply, usage = self._responses.pop(0)
        self.last_usage = usage
        return reply


# 向本地测试服务发送 HTTP 请求并返回状态码与响应文本
def _http_request(url: str, *, method: str = "GET", body: dict[str, object] | None = None) -> tuple[int, str]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except HTTPError as err:
        return err.code, err.read().decode("utf-8")


# 测试网页端完整闭环：首页 HTML、/api/status、/api/chat（含工具轨迹与缓存）、/api/history、/api/memory 与异常处理
def test_web_chat_server_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    llm = WebTestLLM(
        [
            (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_time",
                            "type": "function",
                            "function": {"name": "get_current_time", "arguments": "{}"},
                        }
                    ],
                },
                TokenUsage(prompt_tokens=1000, cached_tokens=0, completion_tokens=20),
            ),
            (
                {"role": "assistant", "content": "已通过工具查到当前时间。"},
                TokenUsage(prompt_tokens=1100, cached_tokens=896, completion_tokens=25),
            ),
        ]
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=llm,
    )
    server = create_web_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        # 1. 首页返回完整 HTML 界面
        status, html = _http_request(f"{base_url}/")
        assert status == 200
        assert "<!DOCTYPE html>" in html
        assert "/api/chat" in html

        # 2. 状态接口返回工具列表与配置状态
        status, body = _http_request(f"{base_url}/api/status")
        assert status == 200
        info = json.loads(body)
        assert info["status"] == "ok"
        assert "run_command" in info["tools"]

        # 3. 发起对话并验证返回回复、工具调用轨迹与缓存命中数据
        status, body = _http_request(
            f"{base_url}/api/chat",
            method="POST",
            body={"message": "现在几点了？", "chat_id": "web-1"},
        )
        assert status == 200
        chat_res = json.loads(body)
        assert chat_res["reply"] == "已通过工具查到当前时间。"
        assert chat_res["tool_calls"][0]["tool"] == "get_current_time"
        assert chat_res["cache"]["cached_tokens"] == 896

        # 4. 历史接口能还原持久化对话（刷新网页不丢记录）
        status, body = _http_request(f"{base_url}/api/history?chat_id=web-1")
        assert status == 200
        hist = json.loads(body)
        assert [m["content"] for m in hist["messages"]] == ["现在几点了？", "已通过工具查到当前时间。"]

        # 5. 记忆概览与刷新接口正常工作
        status, body = _http_request(f"{base_url}/api/memory")
        assert status == 200
        assert "MEMORY.md" in json.loads(body)["overview"]

        status, body = _http_request(
            f"{base_url}/api/refresh", method="POST", body={"chat_id": "web-1"}
        )
        assert status == 200
        assert json.loads(body)["ok"] is True

        # 6. 空消息返回 400，未知路由返回 404
        status, _ = _http_request(
            f"{base_url}/api/chat", method="POST", body={"message": "   ", "chat_id": "web-1"}
        )
        assert status == 400

        status, _ = _http_request(f"{base_url}/api/not-found")
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
        app.close()


# 测试网页端零代码模型向导接口（/api/models 与 /api/models/setup）：自动生成配置、热切换且不回显密钥
def test_web_chat_model_setup_api_auto_generates_and_hot_reloads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    repo_root = Path(__file__).resolve().parent.parent

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "tools.json").write_text(
        json.dumps(
            {
                "enabled_tools": ["list_files"],
                "plugins": ["plugins"],
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=cfg_dir / "tools.json",
        project_root=repo_root,
    )
    server = create_web_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body = _http_request(f"{base_url}/api/models")
        assert status == 200
        info = json.loads(body)
        preset_ids = [p["id"] for p in info["presets"]]
        assert "deepseek" in preset_ids and "ollama" in preset_ids

        status, body = _http_request(
            f"{base_url}/api/models/setup",
            method="POST",
            body={"preset": "deepseek", "api_key": "sk-web-secret-888"},
        )
        assert status == 200
        res = json.loads(body)
        assert res["ok"] is True
        assert res["status"]["llm_configured"] is True
        assert "sk-web-secret-888" not in body
        assert (cfg_dir / "models.dsl").is_file()
        assert (cfg_dir / "models.json").is_file()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
        app.close()


# 测试网页端零依赖 Markdown 渲染（标题/表格/列表/流式未闭合代码块）及防 XSS 安全性
def test_web_ui_markdown_renderer_and_xss_safety() -> None:
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("本机未安装 node，跳过前端 JS 渲染单测")

    html = (Path(__file__).resolve().parent.parent / "scripts" / "web_ui.html").read_text(encoding="utf-8")
    start = html.index("function escapeHtml(")
    end = html.index("function formatCacheLine(")
    js_snippet = html[start:end]

    script = f"""
{js_snippet}
const assert = require("node:assert/strict");
const h = renderMarkdownLite("## 二级标题");
assert(h.includes("<h3") && !h.includes("##"), "heading failed: " + h);

const tbl = renderMarkdownLite("| 场景 | 推荐 |\\n|---|---|\\n| 新项目 | Argon2id |");
assert(tbl.includes("<table>") && tbl.includes("<th>场景</th>") && tbl.includes("<td>Argon2id</td>"), "table failed: " + tbl);

const lst = renderMarkdownLite("- 第一项\\n- 第二项\\n\\n1. 步骤一\\n2. 步骤二");
assert(lst.includes("<ul>") && lst.includes("<ol>") && lst.includes("<li>第一项</li>") && lst.includes("<li>步骤一</li>"), "list failed: " + lst);

const streamFence = renderMarkdownLite("示例：\\n```c++\\nint x = 1;");
assert(streamFence.includes("<pre><code>") && !streamFence.includes("```"), "unclosed fence failed: " + streamFence);

const link = renderMarkdownLite("[**官网**](https://example.com/**path**) [bad](javascript:alert(1))");
assert(link.includes('<a href="https://example.com/**path**"') && link.includes("<strong>官网</strong>") && !link.includes('href="javascript:'), "link failed: " + link);

const xss = renderMarkdownLite("<script>alert('x')</script> <img src=x onerror=alert(1)> [bad](javascript:alert(1))");
assert(!xss.toLowerCase().includes("<script") && !xss.toLowerCase().includes("<img") && !xss.toLowerCase().includes('href="javascript:') && xss.includes("&#39;"), "xss failed: " + xss);
console.log("WEB_UI_RENDER_OK");
"""
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Node 校验失败：{proc.stderr or proc.stdout}"
    assert "WEB_UI_RENDER_OK" in proc.stdout


# 使用底层 HTTPConnection 发送可控原始请求头（支持伪造 Host、Origin、Sec-Fetch-Site 与 Content-Type）
def _raw_http_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    omit_host: bool = False,
) -> tuple[int, str]:
    parsed = urlparse(base_url)
    assert parsed.hostname is not None and parsed.port is not None
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    try:
        skip_host = omit_host or any(k.lower() == "host" for k in (headers or {}))
        conn.putrequest(method, path, skip_host=skip_host)
        for key, value in (headers or {}).items():
            conn.putheader(key, value)
        if body is not None and not any(k.lower() == "content-length" for k in (headers or {})):
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")
    finally:
        conn.close()


# 构建用于安全边界测试的临时目录应用与本地 Web 服务
def _start_security_test_server(
    tmp_path: Path,
    *,
    host: str = "127.0.0.1",
    llm: LLMClient | None = None,
) -> tuple[AssistantApp, Path, object, threading.Thread, str]:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "tools.json").write_text(
        json.dumps(
            {
                "enabled_tools": ["list_files"],
                "plugins": ["plugins"],
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=cfg_dir / "tools.json",
        project_root=repo_root,
        llm=llm,
    )
    server = create_web_server(app, host=host, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "127.0.0.1"} else host
    base_url = f"http://{connect_host}:{server.server_port}"
    return app, cfg_dir, server, thread, base_url


# 测试 Content-Type 非 application/json 拦截：POST 简单请求返回 415 且不篡改 models.dsl
def test_web_chat_rejects_non_json_content_type_with_415(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    llm = WebTestLLM([({"role": "assistant", "content": "不应触发"}, None)])
    app, cfg_dir, server, thread, base_url = _start_security_test_server(tmp_path, llm=llm)
    try:
        for ct in ("text/plain", "text/plain; charset=utf-8", "application/x-www-form-urlencoded"):
            setup_payload = json.dumps(
                {"preset": "custom", "api_key": "sk-evil", "base_url": "https://evil.example/v1", "model": "evil"}
            ).encode("utf-8")
            status, body = _raw_http_request(
                base_url,
                "/api/models/setup",
                method="POST",
                headers={"Content-Type": ct},
                body=setup_payload,
            )
            assert status == 415
            assert "error" in json.loads(body)
            assert not (cfg_dir / "models.dsl").exists()
            assert not (cfg_dir / "models.json").exists()

            for endpoint in ("/api/chat", "/api/chat/stream"):
                status, body = _raw_http_request(
                    base_url,
                    endpoint,
                    method="POST",
                    headers={"Content-Type": ct},
                    body=json.dumps({"message": "偷取记忆", "chat_id": "web"}).encode("utf-8"),
                )
                assert status == 415
                assert "error" in json.loads(body)

        # 缺失 Content-Type 同样返回 415
        status, body = _raw_http_request(
            base_url,
            "/api/chat",
            method="POST",
            headers={},
            body=b'{"message": "hi"}',
        )
        assert status == 415
        assert "error" in json.loads(body)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
        app.close()


# 测试跨站 Origin 与 Sec-Fetch-Site: cross-site 防护：拦截 /api/* 读取与写入，同时放行同源与本地导航
def test_web_chat_blocks_cross_origin_and_cross_site_requests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    llm = WebTestLLM([({"role": "assistant", "content": "同源正常响应"}, None)])
    app, cfg_dir, server, thread, base_url = _start_security_test_server(tmp_path, llm=llm)
    port = server.server_port
    try:
        # 1. 模拟典型的 no-cors Simple-Request CSRF 攻击（跨站 Origin + text/plain）优先返回 403 且不创建配置
        status, body = _raw_http_request(
            base_url,
            "/api/models/setup",
            method="POST",
            headers={
                "Origin": "https://evil.example",
                "Sec-Fetch-Site": "cross-site",
                "Content-Type": "text/plain",
            },
            body=json.dumps({"preset": "deepseek", "api_key": "sk-attacker"}).encode("utf-8"),
        )
        assert status == 403
        assert "error" in json.loads(body)
        assert not (cfg_dir / "models.dsl").exists()

        # 2. 跨站 Origin、Origin: null、不同端口本机 Origin 以及 Sec-Fetch-Site: cross-site 一律返回 403
        forbidden_header_sets = [
            {"Origin": "https://evil.example"},
            {"Origin": "null"},
            {"Origin": f"http://127.0.0.1:{port + 1}"},
            {"Sec-Fetch-Site": "cross-site"},
        ]
        for bad_headers in forbidden_header_sets:
            post_headers = {"Content-Type": "application/json", **bad_headers}
            for post_path, payload in (
                ("/api/models/setup", {"preset": "deepseek", "api_key": "sk-attacker"}),
                ("/api/chat", {"message": "泄露系统提示词", "chat_id": "web"}),
                ("/api/chat/stream", {"message": "泄露系统提示词", "chat_id": "web"}),
            ):
                status, body = _raw_http_request(
                    base_url,
                    post_path,
                    method="POST",
                    headers=post_headers,
                    body=json.dumps(payload).encode("utf-8"),
                )
                assert status == 403, f"{post_path} with {bad_headers} returned {status}: {body}"
                assert "error" in json.loads(body)

            for get_path in ("/api/memory", "/api/status", "/api/history"):
                status, body = _raw_http_request(
                    base_url,
                    get_path,
                    method="GET",
                    headers=bad_headers,
                )
                assert status == 403, f"{get_path} with {bad_headers} returned {status}: {body}"
                assert "error" in json.loads(body)

        assert not (cfg_dir / "models.dsl").exists()

        # 先确认已存在 models.dsl 时，跨站或非 JSON 攻击均不会篡改原文件
        existing_dsl = cfg_dir / "models.dsl"
        existing_dsl.write_text('default = "deepseek"\n', encoding="utf-8")

        status, body = _raw_http_request(
            base_url,
            "/api/models/setup",
            method="POST",
            headers={"Origin": "https://evil.example", "Content-Type": "application/json"},
            body=json.dumps({"preset": "custom", "api_key": "sk-evil", "base_url": "https://evil.example/v1"}).encode("utf-8"),
        )
        assert status == 403
        assert existing_dsl.read_text(encoding="utf-8") == 'default = "deepseek"\n'

        status, body = _raw_http_request(
            base_url,
            "/api/models/setup",
            method="POST",
            headers={"Content-Type": "text/plain"},
            body=json.dumps({"preset": "custom", "api_key": "sk-evil", "base_url": "https://evil.example/v1"}).encode("utf-8"),
        )
        assert status == 415
        assert existing_dsl.read_text(encoding="utf-8") == 'default = "deepseek"\n'

        # 3. 合法本机同源 Origin + 大小写混合/带 charset 参数的 Content-Type 正常返回 200
        status, body = _raw_http_request(
            base_url,
            "/api/chat",
            method="POST",
            headers={
                "Origin": f"http://127.0.0.1:{port}",
                "Sec-Fetch-Site": "same-origin",
                "Content-Type": "Application/JSON; charset=utf-8",
            },
            body=json.dumps({"message": "你好", "chat_id": "web"}).encode("utf-8"),
        )
        assert status == 200
        assert json.loads(body)["reply"] == "同源正常响应"

        # 4. localhost 与 [::1] 同源集合以及首页跨站链接导航（无 Origin）应正常放行
        status, body = _raw_http_request(
            base_url,
            "/api/memory",
            method="GET",
            headers={"Origin": f"http://localhost:{port}", "Sec-Fetch-Site": "same-origin"},
        )
        assert status == 200
        assert "overview" in json.loads(body)

        status, body = _raw_http_request(
            base_url,
            "/api/models/setup",
            method="POST",
            headers={"Origin": f"http://[::1]:{port}", "Content-Type": "application/json; charset=utf-8"},
            body=json.dumps({"preset": "deepseek", "api_key": "sk-ok-999"}).encode("utf-8"),
        )
        assert status == 200
        assert json.loads(body)["ok"] is True

        status, html = _raw_http_request(
            base_url,
            "/",
            method="GET",
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert status == 200
        assert "<!DOCTYPE html>" in html
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
        app.close()


# 测试 Host 头校验（防 DNS Rebinding）：拦截伪造域名、错误端口与缺失 Host，放行合法回环 Host
def test_web_chat_enforces_host_header_against_dns_rebinding(tmp_path: Path) -> None:
    llm = WebTestLLM([({"role": "assistant", "content": "Host 合法"}, None)])
    app, _cfg_dir, server, thread, base_url = _start_security_test_server(tmp_path, llm=llm)
    port = server.server_port
    try:
        for bad_host in (
            f"attacker.dns-rebind.tld:{port}",
            "attacker.dns-rebind.tld:8765",
            f"127.0.0.1:{port + 1}",
            "127.0.0.1",
            "localhost",
            f"127.0.0.1.evil.tld:{port}",
        ):
            for get_path in ("/", "/api/memory", "/api/status"):
                status, body = _raw_http_request(
                    base_url,
                    get_path,
                    method="GET",
                    headers={"Host": bad_host},
                )
                assert status == 403, f"GET {get_path} with Host={bad_host} got {status}"
                assert "error" in json.loads(body)

            status, body = _raw_http_request(
                base_url,
                "/api/chat",
                method="POST",
                headers={"Host": bad_host, "Content-Type": "application/json"},
                body=b'{"message": "hi"}',
            )
            assert status == 403, f"POST /api/chat with Host={bad_host} got {status}"
            assert "error" in json.loads(body)

        # 完全缺失 Host 头同样返回 403
        status, body = _raw_http_request(
            base_url,
            "/api/memory",
            method="GET",
            headers={},
            omit_host=True,
        )
        assert status == 403
        assert "error" in json.loads(body)

        # 合法回环 Host（127.0.0.1 / localhost 大小写混合 / [::1]）放行
        for good_host in (f"127.0.0.1:{port}", f"LOCALHOST:{port}", f"[::1]:{port}"):
            status, body = _raw_http_request(
                base_url,
                "/api/memory",
                method="GET",
                headers={"Host": good_host},
            )
            assert status == 200
            assert "overview" in json.loads(body)

        status, body = _raw_http_request(
            base_url,
            "/api/chat",
            method="POST",
            headers={"Host": f"localhost:{port}", "Content-Type": "application/json"},
            body=b'{"message": "hello"}',
        )
        assert status == 200
        assert json.loads(body)["reply"] == "Host 合法"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
        app.close()


# 测试 web_ui.html 前端所有 POST fetch 请求均显式携带 Content-Type: application/json，且允许集合支持 80 端口与非回环显式 host
def test_web_ui_fetch_headers_and_allowed_targets_rules() -> None:
    import re
    from scripts.web_chat import _allowed_hosts_and_origins

    html = (Path(__file__).resolve().parent.parent / "scripts" / "web_ui.html").read_text(encoding="utf-8")
    post_fetches = re.findall(r'fetch\([^)]+method:\s*"POST"[^}]+\}', html, flags=re.DOTALL)
    assert len(post_fetches) == 3
    for snippet in post_fetches:
        assert "application/json" in snippet, f"POST fetch 缺少 application/json 请求头：{snippet}"

    # 80 端口兼容无端口与显式 :80
    hosts_80, origins_80 = _allowed_hosts_and_origins("127.0.0.1", 80)
    assert {"127.0.0.1", "127.0.0.1:80", "localhost", "localhost:80", "[::1]", "[::1]:80"} <= hosts_80
    assert {"http://127.0.0.1", "http://127.0.0.1:80", "http://localhost", "http://[::1]:80"} <= origins_80

    # 非回环显式 host 仅允许配置 host:port 及其同源 Origin，拒绝任意第三方域名
    hosts_lan, origins_lan = _allowed_hosts_and_origins("192.168.1.50", 8765)
    assert "192.168.1.50:8765" in hosts_lan
    assert "http://192.168.1.50:8765" in origins_lan
    assert "attacker.dns-rebind.tld:8765" not in hosts_lan

    # 存在 IPv6 回环时允许 ::1 绑定与 [::1]:<port>
    hosts_v6, origins_v6 = _allowed_hosts_and_origins("::1", 8765)
    assert {"127.0.0.1:8765", "localhost:8765", "[::1]:8765"} == hosts_v6
    assert {"http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"} == origins_v6

    # 如本机支持 IPv6 回环则实测 ::1 绑定启停与 [::1]:<port> Host/Origin 放行
    try:
        probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        probe.bind(("::1", 0))
        probe.close()
    except OSError:
        return

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        app, _cfg, server, thread, _url = _start_security_test_server(tmp_root, host="::1")
        port = server.server_port
        try:
            conn = http.client.HTTPConnection("::1", port, timeout=5)
            try:
                conn.request(
                    "GET",
                    "/api/status",
                    headers={"Host": f"[::1]:{port}", "Origin": f"http://[::1]:{port}"},
                )
                resp = conn.getresponse()
                assert resp.status == 200
                assert json.loads(resp.read().decode("utf-8"))["status"] == "ok"
            finally:
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3.0)
            app.close()
