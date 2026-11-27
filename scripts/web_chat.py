"""零第三方依赖的本地网页对话脚本：用标准库 ThreadingHTTPServer 接入 AssistantApp。"""
from __future__ import annotations

import argparse
import contextlib
import hmac
import ipaddress
import json
import os
import socket
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TextIO
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from personal_assistant.ai.client import LLMNotConfigured
from personal_assistant.assistant.app import AssistantApp, build_app
from personal_assistant.assistant.common import load_dotenv

UI_HTML_PATH = Path(__file__).resolve().parent / "web_ui.html"
MAX_REQUEST_BYTES = 256 * 1024
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")
ALLOWED_SEC_FETCH_SITES = frozenset({"same-origin", "none"})


# 标记 POST 请求 Content-Type 非 application/json 时的 415 协议错误
class _UnsupportedMediaTypeError(ValueError):
    pass


# 解析绑定主机名（去括号、IP 字面量 RFC 5952 规范化与小写处理）并判断是否为 IPv6 字面量
def _parse_bind_host(host: str) -> tuple[str, bool]:
    clean = host.strip().strip("[]").lower()
    if not clean:
        return "", False
    try:
        ip = ipaddress.ip_address(clean)
        return str(ip), ip.version == 6
    except ValueError:
        return clean, False


# 判断绑定主机名是否仅限本机回环地址（127.0.0.0/8、localhost、::1）
def _is_loopback_host(host: str) -> bool:
    clean, _ = _parse_bind_host(host)
    if clean in {"127.0.0.1", "localhost", "::1"}:
        return True
    if not clean:
        return False
    try:
        return ipaddress.ip_address(clean).is_loopback
    except ValueError:
        return False


# 当绑定非回环地址时向 stderr 打印醒目安全警告，提示开启令牌鉴权
def _warn_if_non_loopback(host: str, stream: TextIO | None = None) -> bool:
    if _is_loopback_host(host):
        return False
    target = stream if stream is not None else sys.stderr
    print(
        f"[安全警告] 本地网页服务正在监听非回环地址 {host!r}，可能暴露到局域网或外部网络；"
        "请设置环境变量 MY_HELPER_WEB_TOKEN 或传入 --auth-token 开启 Bearer 鉴权。",
        file=target,
    )
    return True


# 基于绑定地址与实际监听端口构造允许的 Host 与同源 Origin 白名单（支持补充通配绑定下的落地地址）
def _allowed_hosts_and_origins(
    host: str,
    port: int,
    *,
    extra_host: str | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    clean, is_ipv6 = _parse_bind_host(host)
    is_loopback = _is_loopback_host(host)

    formatted = f"[{clean}]" if is_ipv6 else clean
    if is_loopback:
        base_hosts = set(LOOPBACK_HOSTS)
        if clean not in {"127.0.0.1", "localhost", "::1"}:
            base_hosts.add(formatted)
    elif clean in {"", "0.0.0.0", "::"}:
        base_hosts = set(LOOPBACK_HOSTS)
        if formatted:
            base_hosts.add(formatted)
    else:
        base_hosts = {formatted}

    if extra_host:
        base_hosts.add(extra_host)

    allowed_hosts: set[str] = set()
    allowed_origins: set[str] = set()
    for item in base_hosts:
        host_port = f"{item}:{port}"
        allowed_hosts.add(host_port)
        allowed_origins.add(f"http://{host_port}")
        if port == 80:
            allowed_hosts.add(item)
            allowed_origins.add(f"http://{item}")
    return frozenset(allowed_hosts), frozenset(allowed_origins)


# 创建绑定了 AssistantApp 的本地多线程 HTTP 服务（不依赖任何第三方 Web 框架）
def create_web_server(
    app: AssistantApp,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    ui_html = UI_HTML_PATH.read_text(encoding="utf-8")
    clean_bind_host, is_ipv6 = _parse_bind_host(host)
    is_wildcard_bind = clean_bind_host in {"", "0.0.0.0", "::"}
    effective_token = (
        auth_token if auth_token is not None else os.getenv("MY_HELPER_WEB_TOKEN", "")
    ).strip()

    # 本地网页与 JSON API 请求处理器
    class WebChatHandler(BaseHTTPRequestHandler):
        timeout = 30

        # 屏蔽默认每条请求的嘈杂 stderr 访问日志
        def log_message(self, format: str, *args: object) -> None:
            return

        # 统一写入防点击劫持与防 MIME 嗅探安全响应头
        def _send_security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")

        # 发送统一 UTF-8 JSON 响应
        def _send_json(
            self,
            status: int,
            payload: dict[str, object],
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            if extra_headers:
                for hdr_name, hdr_val in extra_headers.items():
                    self.send_header(hdr_name, hdr_val)
            self.end_headers()
            self.wfile.write(raw)

        # 发送单帧 UTF-8 SSE 数据事件并立即刷新底层缓冲
        def _send_sse_frame(self, payload: dict[str, object]) -> None:
            raw = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(raw)
            self.wfile.flush()

        # 安全读取同名多值请求头列表（兼容 HTTP/0.9 等 dict headers 无 get_all 的情况）
        def _get_header_values(self, name: str) -> list[str]:
            get_all = getattr(self.headers, "get_all", None)
            if callable(get_all):
                return list(get_all(name) or ())
            value = self.headers.get(name) if isinstance(self.headers, dict) else None
            return [value] if isinstance(value, str) else []

        # 在通配地址绑定模式下获取当前连接落地的本机 IP（用于局域网 Host/Origin 放行）
        def _socket_local_host(self) -> str | None:
            try:
                sock_ip = str(self.connection.getsockname()[0]).split("%", 1)[0]
                ip = ipaddress.ip_address(sock_ip)
                if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
                    ip = ip.ipv4_mapped
                return f"[{ip}]" if ip.version == 6 else str(ip)
            except (AttributeError, OSError, IndexError, ValueError):
                return None

        # 当开启访问令牌时校验唯一 Authorization: Bearer <token> 请求头（恒定时间比对）
        def _check_bearer_auth(self) -> bool:
            if not effective_token:
                return True
            auth_headers = self._get_header_values("Authorization")
            if len(auth_headers) != 1:
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": "未授权访问，请提供有效的 Bearer Token。"},
                    extra_headers={"WWW-Authenticate": "Bearer"},
                )
                return False
            scheme, _, token_val = auth_headers[0].strip().partition(" ")
            provided_bytes = token_val.strip().encode("utf-8", errors="surrogateescape")
            expected_bytes = effective_token.encode("utf-8", errors="surrogateescape")
            if scheme.lower() != "bearer" or not hmac.compare_digest(provided_bytes, expected_bytes):
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": "无效的 Bearer Token。"},
                    extra_headers={"WWW-Authenticate": "Bearer"},
                )
                return False
            return True

        # 校验 Host、可选 Bearer 令牌及同源 Origin / Sec-Fetch-Site，拦截 DNS Rebinding、未授权访问与 CSRF
        def _check_request_boundary(self, *, check_origin: bool, check_auth: bool = True) -> bool:
            sock_host = self._socket_local_host() if is_wildcard_bind else None
            allowed_hosts, allowed_origins = _allowed_hosts_and_origins(
                host, self.server.server_port, extra_host=sock_host
            )

            host_headers = self._get_header_values("Host")
            if len(host_headers) != 1 or host_headers[0].strip().lower() not in allowed_hosts:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "非法的 Host 请求头。"})
                return False

            if check_auth and not self._check_bearer_auth():
                return False

            if not check_origin:
                return True

            sec_fetch_sites = [v.strip().lower() for v in self._get_header_values("Sec-Fetch-Site")]
            if len(sec_fetch_sites) > 1 or (sec_fetch_sites and sec_fetch_sites[0] not in ALLOWED_SEC_FETCH_SITES):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "禁止跨站访问本地接口。"})
                return False

            origin_headers = self._get_header_values("Origin")
            if len(origin_headers) > 1:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "非法的 Origin 请求头。"})
                return False
            if origin_headers and origin_headers[0].strip().lower() not in allowed_origins:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "禁止跨源访问本地接口。"})
                return False

            return True

        # 校验 POST 请求是否显式携带唯一且无逗号折叠的 application/json 主媒体类型
        def _is_json_content_type(self) -> bool:
            ct_headers = self._get_header_values("Content-Type")
            if len(ct_headers) != 1 or "," in ct_headers[0]:
                return False
            media_type = ct_headers[0].split(";", 1)[0].strip().lower()
            return media_type == "application/json"

        # 读取并校验 JSON 请求体（先校验 Content-Type 再限制最大字节数防止异常大包）
        def _read_json_body(self) -> dict[str, object]:
            if not self._is_json_content_type():
                raise _UnsupportedMediaTypeError("Content-Type 必须为 application/json。")
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
            is_static_shell = parsed.path in {"/", "/index.html"}
            is_api = parsed.path == "/api" or parsed.path.startswith("/api/")
            if not self._check_request_boundary(check_origin=is_api, check_auth=not is_static_shell):
                return
            if is_static_shell:
                data = ui_html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self._send_security_headers()
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/status":
                self._send_json(HTTPStatus.OK, app.status())
                return
            if parsed.path == "/api/memory":
                self._send_json(HTTPStatus.OK, {"overview": app.memory_overview()})
                return
            if parsed.path == "/api/models":
                self._send_json(HTTPStatus.OK, app.models_overview())
                return
            if parsed.path == "/api/history":
                qs = parse_qs(parsed.query)
                chat_id = (qs.get("chat_id") or ["web"])[0].strip() or "web"
                compaction, msgs = app.state.get_context_snapshot(chat_id, limit=40)
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
            if not self._check_request_boundary(check_origin=True):
                return
            parsed = urlparse(self.path)
            try:
                body = self._read_json_body()
            except _UnsupportedMediaTypeError as error:
                self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": str(error)})
                return
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

            if parsed.path == "/api/chat/stream":
                message = str(body.get("message") or "")
                chat_id = str(body.get("chat_id") or "web").strip() or "web"
                try:
                    frames = app.stream_chat(message, chat_id=chat_id)
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                except LLMNotConfigured as error:
                    self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self._send_security_headers()
                self.end_headers()
                try:
                    with contextlib.closing(frames):
                        for frame in frames:
                            self._send_sse_frame(frame)
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as error:  # noqa: BLE001
                    try:
                        self._send_sse_frame(
                            {"type": "error", "error": f"对话执行失败：{type(error).__name__}: {error}"}
                        )
                    except OSError:
                        return
                return

            if parsed.path == "/api/refresh":
                chat_id = str(body.get("chat_id") or "web").strip() or "web"
                app.reset_session_prompt(chat_id)
                self._send_json(HTTPStatus.OK, {"ok": True, "chat_id": chat_id})
                return

            if parsed.path == "/api/models/setup":
                try:
                    configured = app.configure_model(
                        preset=str(body.get("preset") or ""),
                        api_key=str(body.get("api_key") or ""),
                        model=str(body.get("model") or ""),
                        base_url=str(body.get("base_url") or ""),
                        name=str(body.get("name") or ""),
                    )
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "configured": configured, "status": app.status()},
                )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"未知接口：{parsed.path}"})

    if is_ipv6:
        # 绑定 IPv6 地址的多线程 HTTP 服务子类
        class _IPv6ThreadingHTTPServer(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        return _IPv6ThreadingHTTPServer((clean_bind_host, port), WebChatHandler)

    return ThreadingHTTPServer((clean_bind_host or host.strip(), port), WebChatHandler)


# 命令行启动入口：加载 .env、启动本地网页服务并在 Ctrl+C 时优雅收尾后台沉淀任务
def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="启动 My Helper 本地网页对话界面（纯标准库，无需额外插件）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    parser.add_argument("--model", default=None, help="指定 config/models.dsl 或 models.json 中的模型名称")
    parser.add_argument("--consolidator-model", default=None, help="指定后台记忆沉淀专用模型名称")
    parser.add_argument("--auth-token", default=None, help="设置 Bearer 访问令牌（默认读取 MY_HELPER_WEB_TOKEN）")
    parser.add_argument("--open", action="store_true", help="启动后自动打开系统默认浏览器")
    args = parser.parse_args(argv)

    _warn_if_non_loopback(args.host)
    app = build_app(model_name=args.model, consolidator_model=args.consolidator_model)
    server = create_web_server(app, host=args.host, port=args.port, auth_token=args.auth_token)
    clean_host, is_ipv6 = _parse_bind_host(args.host)
    url_host = f"[{clean_host}]" if is_ipv6 else (clean_host or args.host)
    url = f"http://{url_host}:{server.server_port}"
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
