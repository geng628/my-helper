"""本地网页对话脚本（scripts/web_chat.py）单元测试（TDD）。"""
from __future__ import annotations

from collections.abc import Sequence
import http.client
import io
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


# 测试网页端流式思维链 thinking_delta（含 XSS 防护、正文到达/done 自动折叠）与 tool_start -> tool_step 就地状态升级
def test_web_ui_thinking_delta_xss_and_tool_start_upgrade() -> None:
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("本机未安装 node，跳过前端流式渲染单测")

    html = (Path(__file__).resolve().parent.parent / "scripts" / "web_ui.html").read_text(encoding="utf-8")
    assert "details.thinking" in html
    assert "applyStreamEvent" in html

    start = html.index("function escapeHtml(")
    end = html.index("async function refreshStatus(")
    js_snippet = html[start:end]

    script = f"""
const assert = require("node:assert/strict");
function makeElement(tag) {{
  const el = {{
    tagName: tag.toUpperCase(),
    className: "",
    open: false,
    _innerHTML: "",
    _textContent: "",
    attrs: {{}},
    children: [],
    parentNode: null,
    style: {{}},
    setAttribute(k, v) {{ this.attrs[k] = String(v); }},
    getAttribute(k) {{ return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }},
    appendChild(child) {{
      child.parentNode = this;
      this.children.push(child);
      return child;
    }},
    insertBefore(newNode, refNode) {{
      newNode.parentNode = this;
      const idx = this.children.indexOf(refNode);
      if (idx === -1) this.children.push(newNode);
      else this.children.splice(idx, 0, newNode);
      return newNode;
    }},
    querySelector(sel) {{
      if (sel.startsWith(".")) {{
        const cls = sel.slice(1);
        return this.children.find((c) => c.className.split(/\\s+/).includes(cls)) || null;
      }}
      return null;
    }},
    get innerHTML() {{ return this._innerHTML; }},
    set innerHTML(val) {{ this._innerHTML = String(val); this._textContent = String(val); }},
    get textContent() {{ return this._textContent; }},
    set textContent(val) {{ this._textContent = String(val); this._innerHTML = String(val); }},
  }};
  return el;
}}
const chatEl = makeElement("main");
chatEl.scrollTop = 0;
chatEl.scrollHeight = 100;
const emptyEl = makeElement("div");
const document = {{ createElement: makeElement }};
{js_snippet}

// 1. 完整流式交互：thinking_delta(含 XSS) -> tool_start -> tool_step(同 call_id 就地替换) -> text_delta(自动折叠) -> done
const wrap = appendMessage("assistant", "思考与执行工具中…");
const bubble = wrap.querySelector(".bubble");
const ctx = {{
  wrap,
  bubble,
  toolState: {{ details: null, summary: null, count: 0, pending: [] }},
  thinkingState: {{ details: null, summary: null, body: null, text: "", collapsed: false }},
  replyText: "",
}};

applyStreamEvent(ctx, {{ type: "thinking_delta", text: "先推导 " }});
applyStreamEvent(ctx, {{ type: "thinking_delta", text: "<img src=x onerror=alert(1)> 结束" }});
assert(ctx.thinkingState.details !== null, "thinking details not created");
assert.equal(ctx.thinkingState.details.className, "thinking");
assert.equal(ctx.thinkingState.details.open, true);
assert.equal(ctx.thinkingState.summary.textContent, "💭 推导思路（思考中…）");
assert(!ctx.thinkingState.body.innerHTML.toLowerCase().includes("<img"), "XSS not escaped: " + ctx.thinkingState.body.innerHTML);
assert(ctx.thinkingState.body.innerHTML.includes("&lt;img src=x onerror=alert(1)&gt;"));
assert.equal(bubble.innerHTML, "<p>思考与执行工具中…</p>");

applyStreamEvent(ctx, {{ type: "tool_start", tool: "web_search", arguments: '{{"q":"algo"}}', call_id: "c1" }});
assert(ctx.toolState.details !== null, "tools details not created");
assert.equal(ctx.toolState.details.open, true);
assert.equal(ctx.toolState.summary.textContent, "🔧 正在调用 web_search…");
assert.equal(wrap.children[0], ctx.thinkingState.details, "thinking must be above tools");
assert.equal(wrap.children[1], ctx.toolState.details, "tools must be above bubble");
assert.equal(wrap.children[2], bubble);
const toolItemsBefore = ctx.toolState.details.children.filter((c) => c.className.includes("tool-item"));
assert.equal(toolItemsBefore.length, 1);
assert.equal(toolItemsBefore[0].getAttribute("data-call-id"), "c1");
assert(toolItemsBefore[0].innerHTML.includes("[调用中]") && toolItemsBefore[0].innerHTML.includes("web_search"));

applyStreamEvent(ctx, {{ type: "tool_step", tool: "web_search", arguments: '{{"q":"algo"}}', observation: "命中结果", duration_ms: 12.3, call_id: "c1" }});
const toolItemsAfter = ctx.toolState.details.children.filter((c) => c.className.includes("tool-item"));
assert.equal(toolItemsAfter.length, 1, "tool_step must replace pending item in place");
assert.equal(ctx.toolState.summary.textContent, "🔧 工具调用轨迹（1 步）");
assert(toolItemsAfter[0].innerHTML.includes("[Act]") && toolItemsAfter[0].innerHTML.includes("(12.3ms)") && toolItemsAfter[0].innerHTML.includes("[Observe] 命中结果"));
assert(!toolItemsAfter[0].innerHTML.includes("[调用中]"));

// 匿名 call_id 走队首替换，不匹配 call_id 时追加新节点
applyStreamEvent(ctx, {{ type: "tool_start", tool: "get_time", arguments: "{{}}", call_id: "" }});
applyStreamEvent(ctx, {{ type: "tool_step", tool: "get_time", arguments: "{{}}", observation: "12:00", duration_ms: 1.0, call_id: "" }});
applyStreamEvent(ctx, {{ type: "tool_step", tool: "extra", arguments: "{{}}", observation: "extra_obs", duration_ms: 2.0, call_id: "unmatched" }});
const toolItemsFinal = ctx.toolState.details.children.filter((c) => c.className.includes("tool-item"));
assert.equal(toolItemsFinal.length, 3);
assert.equal(ctx.toolState.summary.textContent, "🔧 工具调用轨迹（3 步）");

const expectedLen = "先推导 <img src=x onerror=alert(1)> 结束".length;
applyStreamEvent(ctx, {{ type: "text_delta", text: "最终**回答**" }});
assert.equal(ctx.thinkingState.details.open, false, "first text_delta must collapse thinking");
assert.equal(ctx.thinkingState.summary.textContent, `💭 推导思路（${{expectedLen}} 字，点击展开）`);
assert(bubble.innerHTML.includes("<strong>回答</strong>"));

applyStreamEvent(ctx, {{ type: "done", reply: "最终**回答**", thinking: "先推导 <img src=x onerror=alert(1)> 结束" }});
assert.equal(ctx.toolState.details.open, false);
assert.equal(ctx.thinkingState.details.open, false);

// 2. 仅有 thinking_delta 直接到 done（无 text_delta）同样必须自动折叠并标注字数
const wrap2 = appendMessage("assistant", "思考与执行工具中…");
const ctx2 = {{
  wrap: wrap2,
  bubble: wrap2.querySelector(".bubble"),
  toolState: {{ details: null, summary: null, count: 0, pending: [] }},
  thinkingState: {{ details: null, summary: null, body: null, text: "", collapsed: false, stepBreak: false }},
  replyText: "",
  textStepBreak: false,
}};
applyStreamEvent(ctx2, {{ type: "thinking_delta", text: "纯思考" }});
assert.equal(ctx2.thinkingState.details.open, true);
applyStreamEvent(ctx2, {{ type: "done", reply: "直接结果", thinking: "纯思考" }});
assert.equal(ctx2.thinkingState.details.open, false);
assert.equal(ctx2.thinkingState.summary.textContent, "💭 推导思路（3 字，点击展开）");

// 3. 多步推导与跨工具正文分段（thinking -> text -> tool_start -> tool_step -> thinking -> text -> done）
const wrap3 = appendMessage("assistant", "思考与执行工具中…");
const ctx3 = {{
  wrap: wrap3,
  bubble: wrap3.querySelector(".bubble"),
  toolState: {{ details: null, summary: null, count: 0, pending: [] }},
  thinkingState: {{ details: null, summary: null, body: null, text: "", collapsed: false, stepBreak: false }},
  replyText: "",
  textStepBreak: false,
}};
applyStreamEvent(ctx3, {{ type: "thinking_delta", text: "第一步推导" }});
applyStreamEvent(ctx3, {{ type: "text_delta", text: "过渡说明" }});
applyStreamEvent(ctx3, {{ type: "tool_start", tool: "search", arguments: "{{}}", call_id: "s1" }});
assert.equal(ctx3.thinkingState.stepBreak, true);
assert.equal(ctx3.textStepBreak, true);
applyStreamEvent(ctx3, {{ type: "tool_step", tool: "search", arguments: "{{}}", observation: "ok", duration_ms: 2.0, call_id: "s1" }});
applyStreamEvent(ctx3, {{ type: "thinking_delta", text: "第二步推导" }});
assert.equal(ctx3.thinkingState.text, "第一步推导\\n\\n第二步推导");
assert.equal(ctx3.thinkingState.stepBreak, false);
applyStreamEvent(ctx3, {{ type: "text_delta", text: "最终正文" }});
assert.equal(ctx3.replyText, "过渡说明\\n\\n最终正文");
assert.equal(ctx3.textStepBreak, false);
applyStreamEvent(ctx3, {{ type: "done", reply: "最终正文", thinking: "第一步推导第二步推导" }});
assert.equal(ctx3.thinkingState.details.open, false);

// 若正在思考时即进入 tool_start（未先收 text_delta），标题需显示执行工具中…
const wrap4 = appendMessage("assistant", "思考与执行工具中…");
const ctx4 = {{
  wrap: wrap4,
  bubble: wrap4.querySelector(".bubble"),
  toolState: {{ details: null, summary: null, count: 0, pending: [] }},
  thinkingState: {{ details: null, summary: null, body: null, text: "", collapsed: false, stepBreak: false }},
  replyText: "",
  textStepBreak: false,
}};
applyStreamEvent(ctx4, {{ type: "thinking_delta", text: "先查一下" }});
applyStreamEvent(ctx4, {{ type: "tool_start", tool: "calc", arguments: "{{}}", call_id: "c4" }});
assert.equal(ctx4.thinkingState.summary.textContent, "💭 推导思路（执行工具中…）");
console.log("WEB_UI_STREAM_OK");
"""
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Node 流式渲染校验失败：{proc.stderr or proc.stdout}"
    assert "WEB_UI_STREAM_OK" in proc.stdout


# 使用底层 HTTPConnection 发送可控原始请求头（支持多值/重复同名头、伪造 Host/Origin/Sec-Fetch-Site/Content-Type）
def _raw_http_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | Sequence[tuple[str, str]] | None = None,
    body: bytes | None = None,
    omit_host: bool = False,
) -> tuple[int, str]:
    parsed = urlparse(base_url)
    assert parsed.hostname is not None and parsed.port is not None
    header_items = list(headers.items()) if isinstance(headers, dict) else list(headers or ())
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    try:
        skip_host = omit_host or any(k.lower() == "host" for k, _ in header_items)
        conn.putrequest(method, path, skip_host=skip_host)
        for key, value in header_items:
            conn.putheader(key, value)
        if body is not None and not any(k.lower() == "content-length" for k, _ in header_items):
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")
    finally:
        conn.close()


# 通过原始 TCP Socket 发送 HTTP/0.9 等非标准报文并读取完整字节响应
def _raw_socket_request(base_url: str, raw_payload: bytes) -> bytes:
    parsed = urlparse(base_url)
    assert parsed.hostname is not None and parsed.port is not None
    with socket.create_connection((parsed.hostname, parsed.port), timeout=5) as sock:
        sock.sendall(raw_payload)
        sock.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


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
        for ct in (
            "text/plain",
            "text/plain; charset=utf-8",
            "application/x-www-form-urlencoded",
            "",
            "application/json-patch+json",
            "text/plain; application/json",
            "application/json; charset=utf-8, text/plain",
        ):
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
            assert status == 415, f"Content-Type={ct!r} returned {status}"
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
                assert status == 415, f"{endpoint} with Content-Type={ct!r} returned {status}"
                assert "error" in json.loads(body)

        # 缺失 Content-Type 与重复双 Content-Type 头同样返回 415
        for extra_headers in ([], [("Content-Type", "application/json"), ("Content-Type", "text/plain")]):
            status, body = _raw_http_request(
                base_url,
                "/api/chat",
                method="POST",
                headers=extra_headers,
                body=b'{"message": "hi"}',
            )
            assert status == 415, f"headers={extra_headers!r} returned {status}"
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

        # 2. 跨站 Origin、Origin: null、不同端口本机 Origin、重复双 Origin、Sec-Fetch-Site: cross-site/same-site 一律返回 403
        forbidden_header_sets: list[list[tuple[str, str]]] = [
            [("Origin", "https://evil.example")],
            [("Origin", "null")],
            [("Origin", f"http://127.0.0.1:{port + 1}")],
            [("Origin", f"http://127.0.0.1:{port}"), ("Origin", f"http://localhost:{port}")],
            [("Sec-Fetch-Site", "cross-site")],
            [("Sec-Fetch-Site", "same-site")],
        ]
        for bad_headers in forbidden_header_sets:
            post_headers = [("Content-Type", "application/json"), *bad_headers]
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
            f"localhost.:{port}",
            "",
            f"::1:{port}",
            f"127.0.0.1.evil.tld:{port}",
        ):
            for get_path in ("/", "/api/memory", "/api/status"):
                status, body = _raw_http_request(
                    base_url,
                    get_path,
                    method="GET",
                    headers={"Host": bad_host},
                )
                assert status == 403, f"GET {get_path} with Host={bad_host!r} got {status}"
                assert "error" in json.loads(body)

            status, body = _raw_http_request(
                base_url,
                "/api/chat",
                method="POST",
                headers={"Host": bad_host, "Content-Type": "application/json"},
                body=b'{"message": "hi"}',
            )
            assert status == 403, f"POST /api/chat with Host={bad_host!r} got {status}"
            assert "error" in json.loads(body)

        # 完全缺失 Host 头、重复双 Host 头以及 HTTP/0.9 原始请求同样受控返回 403 且不崩溃
        for bad_headers, use_omit in (([], True), ([("Host", f"127.0.0.1:{port}"), ("Host", f"localhost:{port}")], False)):
            status, body = _raw_http_request(
                base_url,
                "/api/memory",
                method="GET",
                headers=bad_headers,
                omit_host=use_omit,
            )
            assert status == 403
            assert "error" in json.loads(body)

        raw_http09_resp = _raw_socket_request(base_url, b"GET /api/memory\r\n")
        http09_json = json.loads(raw_http09_resp.decode("utf-8"))
        assert "error" in http09_json and "overview" not in http09_json

        # 显式验证 dict 类型 headers（无 get_all 方法，模拟 HTTP/0.9 空 dict）也能平稳返回 403 JSON 而不抛 AttributeError
        fake_handler = server.RequestHandlerClass.__new__(server.RequestHandlerClass)
        fake_handler.headers = {}
        fake_handler.server = server
        fake_handler.request_version = "HTTP/0.9"
        fake_handler.requestline = "GET /api/memory"
        fake_handler.wfile = io.BytesIO()
        assert fake_handler._is_json_content_type() is False
        assert fake_handler._check_request_boundary(check_origin=True) is False
        assert "error" in json.loads(fake_handler.wfile.getvalue().decode("utf-8"))

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
    from scripts.web_chat import _allowed_hosts_and_origins, _parse_bind_host

    html = (Path(__file__).resolve().parent.parent / "scripts" / "web_ui.html").read_text(encoding="utf-8")
    post_fetches = re.findall(r'fetch\([^)]+method:\s*"POST"[^}]+\}', html, flags=re.DOTALL)
    assert len(post_fetches) == 3
    for snippet in post_fetches:
        assert "application/json" in snippet, f"POST fetch 缺少 application/json 请求头：{snippet}"

    # 未压缩 IPv6 字面量自动按 RFC 5952 规范化为压缩格式
    assert _parse_bind_host("[0:0:0:0:0:0:0:1]") == ("::1", True)

    # 80 端口兼容无端口与显式 :80
    hosts_80, origins_80 = _allowed_hosts_and_origins("127.0.0.1", 80)
    assert {"127.0.0.1", "127.0.0.1:80", "localhost", "localhost:80", "[::1]", "[::1]:80"} <= hosts_80
    assert {"http://127.0.0.1", "http://127.0.0.1:80", "http://localhost", "http://[::1]:80"} <= origins_80

    # 非回环显式 host 仅允许配置 host:port 及其同源 Origin，并支持 extra_host 合并展开
    hosts_lan, origins_lan = _allowed_hosts_and_origins("192.168.1.50", 8765, extra_host="10.0.0.8")
    assert {"192.168.1.50:8765", "10.0.0.8:8765"} <= hosts_lan
    assert {"http://192.168.1.50:8765", "http://10.0.0.8:8765"} <= origins_lan
    assert "attacker.dns-rebind.tld:8765" not in hosts_lan

    # 存在 IPv6 回环时允许 ::1 绑定与 [::1]:<port>
    hosts_v6, origins_v6 = _allowed_hosts_and_origins("0:0:0:0:0:0:0:1", 8765)
    assert {"127.0.0.1:8765", "localhost:8765", "[::1]:8765"} == hosts_v6
    assert {"http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"} == origins_v6

    # 如本机支持 IPv6 回环则实测 ::1 绑定启停与 [::1]:<port> Host/Origin 放行及 X-Content-Type-Options: nosniff
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
                assert resp.headers.get("X-Content-Type-Options") == "nosniff"
                assert json.loads(resp.read().decode("utf-8"))["status"] == "ok"
            finally:
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3.0)
            app.close()


# 测试响应安全头、socket 超时配置、SSE 断连立即释放锁回滚、非回环绑定警告与 Bearer Token 鉴权
def test_web_chat_security_headers_sse_closing_non_loopback_warn_and_bearer_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import time
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from personal_assistant.ai import StreamDone, TextDelta
    from scripts.web_chat import _is_loopback_host, _warn_if_non_loopback, main as web_main

    # 1. 验证安全响应头（HTML 与 JSON）与 WebChatHandler.timeout == 30
    llm = WebTestLLM(
        [
            ({"role": "assistant", "content": "ok"}, None),
        ]
    )
    app, _cfg_dir, server, thread, base_url = _start_security_test_server(tmp_path, llm=llm)
    try:
        assert getattr(server.RequestHandlerClass, "timeout", None) == 30
        parsed = urlparse(base_url)
        conn = http.client.HTTPConnection(str(parsed.hostname), int(parsed.port or 80), timeout=5)
        try:
            conn.request("GET", "/", headers={"Host": f"127.0.0.1:{parsed.port}"})
            html_resp = conn.getresponse()
            _ = html_resp.read()
            assert html_resp.status == 200
            assert html_resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert html_resp.headers.get("X-Frame-Options") == "DENY"
            assert html_resp.headers.get("Content-Security-Policy") == "frame-ancestors 'none'"

            conn.request("GET", "/api/status", headers={"Host": f"127.0.0.1:{parsed.port}"})
            api_resp = conn.getresponse()
            _ = api_resp.read()
            assert api_resp.status == 200
            assert api_resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert api_resp.headers.get("X-Frame-Options") == "DENY"
            assert api_resp.headers.get("Content-Security-Policy") == "frame-ancestors 'none'"
        finally:
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
        app.close()

    # 2. 验证 /api/chat/stream 客户端断连后 contextlib.closing 确定性触发生成器 finally 回滚与锁释放
    first_chunk_sent = threading.Event()
    second_chunk_gate = threading.Event()

    # 控制流式吐帧节奏：首帧后调用 remember_fact 暂存并在第二帧前等待门栓，随后恢复正常对话
    class SlowDisconnectLLM(LLMClient):
        # 初始化脚本步数
        def __init__(self) -> None:
            self._step = 0

        # 同步响应入口（用于断连恢复后的下一轮对话）
        def respond(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
            return {"role": "assistant", "content": "恢复后正常"}

        # 流式响应：第 1 步调 remember_fact，第 2 步先吐 1 帧后卡住等待客户端断开
        def stream(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None):
            self._step += 1
            if self._step == 1:
                msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c_rem", "type": "function", "function": {"name": "remember_fact", "arguments": "{}"}}],
                }
                yield StreamDone(message=msg)
                return
            if self._step == 2:
                yield TextDelta(text="第一段")
                first_chunk_sent.set()
                second_chunk_gate.wait(timeout=3.0)
                yield TextDelta(text="第二段")
                time.sleep(0.05)
                yield TextDelta(text="第三段")
                yield StreamDone(message={"role": "assistant", "content": "第一段第二段第三段"})
                return
            yield TextDelta(text="恢复后正常")
            yield StreamDone(message={"role": "assistant", "content": "恢复后正常"})

    sse_app = AssistantApp(data_root=tmp_path / "sse_data", llm=SlowDisconnectLLM())
    sse_srv = create_web_server(sse_app, host="127.0.0.1", port=0)
    sse_thread = threading.Thread(target=sse_srv.serve_forever, daemon=True)
    sse_thread.start()
    try:
        sse_url = f"http://127.0.0.1:{sse_srv.server_port}"
        sock = socket.create_connection(("127.0.0.1", sse_srv.server_port), timeout=5)
        body = json.dumps({"message": "记住，断连测试不应保留", "chat_id": "sse-drop"}).encode("utf-8")
        req_raw = (
            f"POST /api/chat/stream HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{sse_srv.server_port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("utf-8") + body
        sock.sendall(req_raw)
        assert first_chunk_sent.wait(timeout=3.0)
        # 读到前几帧后立即暴力关闭客户端 socket，再放行第二帧触发服务端 BrokenPipeError
        _ = sock.recv(4096)
        sock.close()
        second_chunk_gate.set()

        # 等待服务端关闭生成器后验证 facts.md 已回滚、同 chat_id 可立即再次对话且锁引用计数归零
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if "(turn#" not in sse_app.memory.read_recent_facts():
                break
            time.sleep(0.02)
        res_next = sse_app.chat("断开后恢复对话", chat_id="sse-drop")
        assert res_next["reply"] == "恢复后正常"
        assert "断连测试不应保留" not in sse_app.memory.read_recent_facts()
        assert sse_app._chat_locks["sse-drop"][1] == 0
    finally:
        second_chunk_gate.set()
        sse_srv.shutdown()
        sse_srv.server_close()
        sse_thread.join(timeout=3.0)
        sse_app.close()

    # 3. 非回环绑定检测与 stderr 安全警告
    assert _is_loopback_host("127.0.0.1") is True
    assert _is_loopback_host("localhost") is True
    assert _is_loopback_host("::1") is True
    assert _is_loopback_host("0.0.0.0") is False
    assert _is_loopback_host("::") is False
    assert _is_loopback_host("192.168.1.50") is False

    _warn_if_non_loopback("127.0.0.1")
    assert capsys.readouterr().err == ""
    _warn_if_non_loopback("0.0.0.0")
    warn_err = capsys.readouterr().err
    assert "警告" in warn_err and "0.0.0.0" in warn_err

    # 4. Bearer Token 鉴权（环境变量 MY_HELPER_WEB_TOKEN 与显式 auth_token / CLI --auth-token）
    monkeypatch.setenv("MY_HELPER_WEB_TOKEN", "env-secret-token")
    auth_app = AssistantApp(data_root=tmp_path / "auth_data", llm=WebTestLLM([]))
    auth_srv = create_web_server(auth_app, host="127.0.0.1", port=0)
    auth_thread = threading.Thread(target=auth_srv.serve_forever, daemon=True)
    auth_thread.start()
    try:
        auth_base = f"http://127.0.0.1:{auth_srv.server_port}"
        for path in ("/", "/api/status"):
            with pytest.raises(HTTPError) as no_tok_err:
                urlopen(Request(f"{auth_base}{path}"), timeout=5)
            assert no_tok_err.value.code == 401
            with pytest.raises(HTTPError) as bad_tok_err:
                urlopen(Request(f"{auth_base}{path}", headers={"Authorization": "Bearer wrong"}), timeout=5)
            assert bad_tok_err.value.code == 401
            # 含非 ASCII 字节（0x80..0xFF）的 Authorization 头必须优雅返回 401 JSON 而绝不抛 TypeError 断连
            non_ascii_status, non_ascii_body = _raw_http_request(
                auth_base,
                path,
                method="GET",
                headers={"Authorization": "Bearer \x80\xff"},
            )
            assert non_ascii_status == 401
            assert "error" in json.loads(non_ascii_body)
            with urlopen(
                Request(f"{auth_base}{path}", headers={"Authorization": "Bearer env-secret-token"}),
                timeout=5,
            ) as ok_resp:
                assert ok_resp.status == 200
    finally:
        auth_srv.shutdown()
        auth_srv.server_close()
        auth_thread.join(timeout=3.0)
        auth_app.close()

    # 验证 main() 支持 --auth-token 并在非回环绑定时输出警告
    monkeypatch.delenv("MY_HELPER_WEB_TOKEN", raising=False)
    captured_srv: dict[str, object] = {}

    # 拦截真实 HTTP 服务启动以在单测内验证 CLI 参数解析与告警逻辑
    class _FakeServer:
        server_port = 8765

        # 立即返回而不阻塞事件循环
        def serve_forever(self) -> None:
            return None

        # 空关闭方法
        def server_close(self) -> None:
            return None

    # 记录 create_web_server 入参
    def _fake_create_server(app_obj, host: str = "127.0.0.1", port: int = 8765, *, auth_token: str | None = None):
        captured_srv["host"] = host
        captured_srv["auth_token"] = auth_token
        return _FakeServer()

    monkeypatch.setattr("scripts.web_chat.build_app", lambda **kw: auth_app)
    monkeypatch.setattr("scripts.web_chat.create_web_server", _fake_create_server)
    monkeypatch.setattr(auth_app, "close", lambda: None)
    web_main(["--host", "0.0.0.0", "--port", "0", "--auth-token", "cli-secret-999"])
    cli_err = capsys.readouterr().err
    assert "警告" in cli_err and "0.0.0.0" in cli_err
    assert captured_srv == {"host": "0.0.0.0", "auth_token": "cli-secret-999"}

