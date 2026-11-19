"""Step 4 单元测试：验证 SQLite 对话历史、KV Cache 前缀冻结与索引式记忆编排。"""
from __future__ import annotations

from pathlib import Path

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


# 测试会话内冻结 System Prompt、标准追加式历史前缀复用（保 KV Cache）以及新会话加载记忆索引
def test_assistant_app_kv_cache_prefix_and_memory_index(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = tmp_path / "data"
    llm = RecordingLLM(
        [
            # 第 1 轮：调用 remember_fact 写入 facts.md，并调用插件 get_current_time
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
            {"role": "assistant", "content": "已记下你的项目，并查看了当前时间。"},
            # 第 2 轮（同一会话）：前缀应与第 1 轮字节级完全一致
            {"role": "assistant", "content": "你刚才说你在开发 my-helper 项目。"},
            # 新会话：system prompt 应加载 MEMORY.md 索引指针，但不把 facts.md 原文硬塞进 system
            {"role": "assistant", "content": "让我先检索记忆。"},
        ]
    )

    app = AssistantApp(
        data_root=data_root,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=llm,
    )
    try:
        turn1 = app.chat("记住，我正在开发 my-helper 项目", chat_id="cli-1")
        assert turn1["reply"] == "已记下你的项目，并查看了当前时间。"
        assert [t["tool"] for t in turn1["tool_calls"]] == ["remember_fact", "get_current_time"]
        assert "(msg#1)" in app.memory.read_recent_facts()
        assert "data/memory/facts.md" in app.memory.read_memory_index()

        turn2 = app.chat("我刚才说自己在做什么？", chat_id="cli-1")
        assert "my-helper" in str(turn2["reply"])

        # 验证 KV Cache 友好性：第 2 轮 system 与第 1 轮字节级完全相同，历史按标准 user/assistant 追加
        turn1_msgs = llm.requests[0][0]
        turn2_msgs = llm.requests[2][0]
        assert turn2_msgs[0] == turn1_msgs[0]
        assert turn2_msgs[1] == {"role": "user", "content": "记住，我正在开发 my-helper 项目"}
        assert turn2_msgs[2] == {"role": "assistant", "content": "已记下你的项目，并查看了当前时间。"}
        assert turn2_msgs[3] == {"role": "user", "content": "我刚才说自己在做什么？"}

        # 开启新会话（或 reset_session_prompt 后）：system 加载 MEMORY.md 目录索引，但不灌入具体事实正文
        app.chat("查一下我的项目备忘", chat_id="cli-2")
        new_session_system = str(llm.requests[3][0][0]["content"])
        assert "data/memory/facts.md" in new_session_system
        assert "我正在开发 my-helper 项目" not in new_session_system

        status = app.status()
        assert status["status"] == "ok"
        assert status["llm_configured"] is True
        assert status["memory_files"] == 2
        assert status["messages"] == 6
        assert "get_current_time" in status["tools"]
        overview = app.memory_overview()
        assert "MEMORY.md" in overview
        assert "facts.md" in overview
    finally:
        app.close()


# 测试无 LLM 配置时的状态返回以及对 memory/workspace 的离线关键词检索
def test_assistant_app_offline_status_and_search(tmp_path: Path) -> None:
    app = AssistantApp(data_root=tmp_path / "data", llm=NoopLLMClient())
    try:
        assert app.status()["llm_configured"] is False
        assert app.status()["memory_files"] == 0

        app.memory.remember_user_statement("记住，代码函数前需要写中文注释", message_id=1)
        app.memory.write_workspace_file("plan.md", "# 计划\n用纯标准库完成最小 Agent。\n", mode="create")

        # remember_fact 会创建 facts.md 与索引 MEMORY.md 共 2 个文件
        assert app.status()["memory_files"] == 2
        assert "中文注释" in app.search("中文注释")
        assert "纯标准库" in app.search("纯标准库", root="data/workspace")
    finally:
        app.close()
