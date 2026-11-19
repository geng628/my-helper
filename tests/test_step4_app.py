"""Step 4 单元测试：验证 SQLite 对话历史与 AssistantApp 核心编排闭环。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from personal_assistant.app import AssistantApp
from personal_assistant.db import StateStore
from personal_assistant.llm import LLMClient, NoopLLMClient


# 可录制每次模型请求的脚本化假模型，用于确定性测试 AssistantApp
class RecordingLLM(LLMClient):
    # 保存预设回复队列并初始化请求记录表
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[list[dict[str, object]], list[dict[str, object]] | None]] = []

    # 记录请求快照并返回下一条预设回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        self.requests.append(([dict(m) for m in messages], tools))
        return self._responses.pop(0)


# 测试 StateStore 的写入、按会话隔离、时间顺序截取与重启持久化
def test_state_store_persistence_and_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "chat_history.sqlite3"
    store = StateStore(db_path)
    try:
        for idx in range(15):
            role = "user" if idx % 2 == 0 else "assistant"
            store.append_message("chat-a", role, f"msg-{idx}")
        store.append_message("chat-b", "user", "hello-b")

        recent_a = store.recent_messages("chat-a", limit=12)
        assert [m["content"] for m in recent_a] == [f"msg-{i}" for i in range(3, 15)]
        assert store.count_messages("chat-a") == 15
        assert store.count_messages("chat-b") == 1
        assert store.count_messages() == 16
    finally:
        store.close()

    # 重新打开数据库文件，历史记录仍然完整保留
    reopened = StateStore(db_path)
    try:
        assert reopened.count_messages("chat-a") == 15
        assert reopened.recent_messages("chat-b", limit=5)[0]["content"] == "hello-b"
    finally:
        reopened.close()


# 测试 AssistantApp 多轮对话持久化、history_reference 注入、记忆联动与插件调用
def test_assistant_app_chat_history_memory_and_plugins(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = tmp_path / "data"
    llm = RecordingLLM(
        [
            # 第 1 轮：调用 remember_fact 与插件工具 get_current_time
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "remember_fact", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_current_time", "arguments": "{}"},
                    },
                ],
            },
            {"role": "assistant", "content": "已记住你的偏好，并查看了当前时间。"},
            # 第 2 轮：验证历史对话与 MEMORY.md 已自动注入上下文
            {"role": "assistant", "content": "你之前说过你正在开发 my-helper。"},
        ]
    )

    app = AssistantApp(
        data_root=data_root,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=llm,
    )
    try:
        turn1 = app.chat("记住，我正在开发 my-helper 项目", chat_id="tg-1")
        assert turn1["reply"] == "已记住你的偏好，并查看了当前时间。"
        assert [t["tool"] for t in turn1["tool_calls"]] == ["remember_fact", "get_current_time"]
        assert "(msg#1)" in app.memory.read_core_memory()

        turn2 = app.chat("我刚才说自己在做什么？", chat_id="tg-1")
        assert "my-helper" in str(turn2["reply"])

        # 检查第 2 轮发给模型的上下文：System Prompt 含 MEMORY.md，User 消息含 history_reference
        second_system = str(llm.requests[2][0][0]["content"])
        second_user = json.loads(str(llm.requests[2][0][1]["content"]))
        assert "我正在开发 my-helper 项目" in second_system
        assert second_user["current_user_message"] == "我刚才说自己在做什么？"
        assert len(second_user["history_reference"]) == 2
        assert second_user["history_reference"][0]["content"] == "记住，我正在开发 my-helper 项目"

        status = app.status()
        assert status["status"] == "ok"
        assert status["llm_configured"] is True
        assert status["messages"] == 4
        assert "get_current_time" in status["tools"]
        assert "MEMORY.md" in app.memory_overview()
    finally:
        app.close()


# 测试无 LLM 时的离线检索、ingest_note 收存去重、同名防覆盖与非法文件名拦截
def test_assistant_app_ingest_note_and_offline_search(tmp_path: Path) -> None:
    app = AssistantApp(data_root=tmp_path / "data", llm=NoopLLMClient())
    try:
        assert app.status()["llm_configured"] is False

        # .txt 自动转为 .md 存入 inbox
        r1 = app.ingest_note("study.txt", "# 笔记\n自旋锁适合极短临界区。\n")
        assert r1["path"] == "data/knowledge/inbox/study.md"
        assert r1["duplicate"] is False

        # 相同内容再次上传幂等去重
        r2 = app.ingest_note("study.md", "# 笔记\n自旋锁适合极短临界区。\n")
        assert r2["path"] == "data/knowledge/inbox/study.md"
        assert r2["duplicate"] is True

        # 同名但内容不同时追加短哈希后缀，绝不覆盖旧笔记
        r3 = app.ingest_note("study.md", "# 笔记 v2\n新增休眠锁内容。\n")
        assert r3["duplicate"] is False
        assert r3["path"] != "data/knowledge/inbox/study.md"
        assert str(r3["path"]).startswith("data/knowledge/inbox/study-")

        # 离线检索立刻能搜到两份入库笔记
        assert "自旋锁适合极短临界区" in app.search("自旋锁")
        assert "新增休眠锁内容" in app.search("休眠锁")

        # 拦截非法路径或不支持后缀
        for bad_name in ("../evil.md", ".hidden.md", "sub/a.md", "binary.exe"):
            with pytest.raises(ValueError):
                app.ingest_note(bad_name, "hello")
    finally:
        app.close()
