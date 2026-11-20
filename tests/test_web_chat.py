"""本地网页对话脚本（scripts/web_chat.py）单元测试（TDD）。"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
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
