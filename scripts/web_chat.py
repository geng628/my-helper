"""零第三方依赖的本地网页对话脚本：用标准库 ThreadingHTTPServer 接入 AssistantApp。"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from personal_assistant.app import AssistantApp, build_app
from personal_assistant.cli import load_dotenv
from personal_assistant.llm import LLMNotConfigured

UI_HTML_PATH = Path(__file__).resolve().parent / "web_ui.html"
MAX_REQUEST_BYTES = 256 * 1024


# 创建绑定了 AssistantApp 的本地多线程 HTTP 服务（不依赖任何第三方 Web 框架）
def create_web_server(
    app: AssistantApp,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    ui_html = UI_HTML_PATH.read_text(encoding="utf-8")

    # 本地网页与 JSON API 请求处理器
    class WebChatHandler(BaseHTTPRequestHandler):
        # 屏蔽默认每条请求的嘈杂 stderr 访问日志
        def log_message(self, format: str, *args: object) -> None:
            return

        # 发送统一 UTF-8 JSON 响应
        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        # 读取并校验 JSON 请求体（限制最大字节数防止异常大包）
        def _read_json_body(self) -> dict[str, object]:
            length_raw = self.headers.get("Content-Length", "0").strip()
            try:
                length = int(length_raw)
            except ValueError as error:
                raise ValueError("无效的 Content-Length。") from error
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("请求体大小超出限制。")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("请求体必须是 JSON 对象。")
            return parsed

        # 处理页面加载与只读查询接口（/、/api/status、/api/history、/api/memory）
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                data = ui_html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/status":
                self._send_json(HTTPStatus.OK, app.status())
                return
            if parsed.path == "/api/memory":
                self._send_json(HTTPStatus.OK, {"overview": app.memory_overview()})
                return
            if parsed.path == "/api/history":
                qs = parse_qs(parsed.query)
                chat_id = (qs.get("chat_id") or ["web"])[0].strip() or "web"
                compaction = app.state.get_compaction(chat_id)
                msgs = app.state.active_messages(chat_id, limit=40)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "chat_id": chat_id,
                        "summary": compaction.get("summary") or "",
                        "messages": msgs,
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"未知接口：{parsed.path}"})

        # 处理对话发送与提示词缓存刷新接口（/api/chat、/api/refresh）
        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                body = self._read_json_body()
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return

            if parsed.path == "/api/chat":
                message = str(body.get("message") or "")
                chat_id = str(body.get("chat_id") or "web").strip() or "web"
                try:
                    result = app.chat(message, chat_id=chat_id)
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                except LLMNotConfigured as error:
                    self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
                    return
                except Exception as error:  # noqa: BLE001
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"对话执行失败：{type(error).__name__}: {error}"},
                    )
                    return
                self._send_json(HTTPStatus.OK, result)
                return

            if parsed.path == "/api/refresh":
                chat_id = str(body.get("chat_id") or "web").strip() or "web"
                app.reset_session_prompt(chat_id)
                self._send_json(HTTPStatus.OK, {"ok": True, "chat_id": chat_id})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"未知接口：{parsed.path}"})

    return ThreadingHTTPServer((host, port), WebChatHandler)


# 命令行启动入口：加载 .env、启动本地网页服务并在 Ctrl+C 时优雅收尾后台沉淀任务
def main(argv: list[str] | None = None) -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="启动 My Helper 本地网页对话界面（纯标准库，无需额外插件）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    parser.add_argument("--open", action="store_true", help="启动后自动打开系统默认浏览器")
    args = parser.parse_args(argv)

    app = build_app()
    server = create_web_server(app, host=args.host, port=args.port)
    url = f"http://{args.host}:{server.server_port}"
    print(f"My Helper 本地网页对话已启动：{url}")
    print("按 Ctrl+C 停止服务（退出前会自动等待后台记忆沉淀完成）。")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在收尾并关闭服务…")
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
