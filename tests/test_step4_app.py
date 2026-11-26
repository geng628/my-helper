"""Step 4 单元测试：验证 SQLite 对话历史、KV Cache 前缀冻结与索引式记忆编排。"""
from __future__ import annotations

from pathlib import Path

from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.db import StateStore
from personal_assistant.ai import LLMClient, NoopLLMClient


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

        # 验证缓存观测：第 1 轮含 2 次模型调用（第 2 步 ReAct 复用前缀），第 2 轮直接复用第 1 轮前缀
        assert turn1["cache"]["llm_calls"] == 2
        assert turn1["cache"]["warm_calls"] == 1
        assert float(turn1["cache"]["mean_prefix_reuse_rate"]) > 0.7
        assert turn2["cache"]["llm_calls"] == 1
        assert turn2["cache"]["warm_calls"] == 1
        assert float(turn2["cache"]["mean_prefix_reuse_rate"]) > 0.7

        status = app.status()
        assert status["status"] == "ok"
        assert status["llm_configured"] is True
        assert status["memory_files"] == 2
        assert status["messages"] == 6
        assert status["cache"]["llm_calls"] == 4
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


# 测试流式断连(close)/中途抛错绝不遗留孤儿 user 消息，并校验严格 user/assistant 交替与历史脏消息合并兜底
def test_stream_failure_and_close_rollback_prevent_consecutive_user_messages(tmp_path: Path) -> None:
    import pytest

    from personal_assistant.ai import LLMStreamEvent, StreamDone, TextDelta

    # 严格校验消息交替契约（如 deepseek-reasoner：禁止出现连续两条同角色消息）的模型桩
    class StrictAlternatingLLM(LLMClient):
        # 初始化剧本（支持流事件列表、异常或文本）
        def __init__(self, script: list[object]) -> None:
            self._script = list(script)
            self.requests: list[list[dict[str, object]]] = []

        # 校验发往模型的消息绝无连续两条 user 或连续两条常规 assistant
        @staticmethod
        def _assert_alternating(messages: list[dict[str, object]]) -> None:
            non_system = [m for m in messages if m.get("role") != "system"]
            for prev, curr in zip(non_system, non_system[1:]):
                prev_role = prev.get("role")
                curr_role = curr.get("role")
                if prev_role == curr_role and prev_role in {"user", "assistant"}:
                    if prev_role == "assistant" and ("tool_calls" in prev or "tool_calls" in curr):
                        continue
                    raise AssertionError(f"检测到连续同角色消息 {prev_role}：{messages}")

        # 同步调用入口（含交替校验）
        def respond(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
            self._assert_alternating(messages)
            self.requests.append([dict(m) for m in messages])
            step = self._script.pop(0)
            if isinstance(step, BaseException):
                raise step
            assert isinstance(step, dict)
            return step

        # 流式调用入口（含交替校验，支持中途抛错或分片流）
        def stream(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
        ):
            self._assert_alternating(messages)
            self.requests.append([dict(m) for m in messages])
            step = self._script.pop(0)
            if isinstance(step, list):
                for ev in step:
                    if isinstance(ev, BaseException):
                        raise ev
                    yield ev
                return
            if isinstance(step, BaseException):
                raise step
            assert isinstance(step, dict)
            yield TextDelta(text=str(step.get("content") or ""))
            yield StreamDone(message=step)

    llm = StrictAlternatingLLM(
        [
            # 1. 正常第 1 轮
            {"role": "assistant", "content": "第1轮正常回复"},
            # 2. 第 2 轮流式中途抛错
            [TextDelta(text="半截回复"), RuntimeError("upstream reset")],
            # 3. 第 3 轮验证：第 2 轮已回滚，绝无连续 user
            {"role": "assistant", "content": "第3轮恢复正常"},
            # 4. 第 4 轮将被外部调用方提前 close()
            [TextDelta(text="断开前碎片"), TextDelta(text="不应到达")],
            # 5. 第 5 轮验证：第 4 轮已回滚，绝无连续 user
            {"role": "assistant", "content": "第5轮依然正常"},
            # 6. 第 6 轮：先调用 remember_fact 写入 (msg#...)，再在第二步工具后模型调用抛错
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc_rf", "type": "function", "function": {"name": "remember_fact", "arguments": "{}"}}],
            },
            RuntimeError("second step 502"),
            # 7. 第 7 轮：再次调用 remember_fact，必须拿到全新不撞车的 msg#id 且不触发幂等误跳过
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc_rf2", "type": "function", "function": {"name": "remember_fact", "arguments": "{}"}}],
            },
            {"role": "assistant", "content": "第7轮记忆已保存"},
        ]
    )

    app = AssistantApp(data_root=tmp_path / "data", llm=llm)
    try:
        r1 = app.chat("你好", chat_id="s-safe")
        assert r1["reply"] == "第1轮正常回复"
        assert app.state.count_messages("s-safe") == 2

        with pytest.raises(RuntimeError, match="upstream reset"):
            app.chat("会失败的提问", chat_id="s-safe")
        assert app.state.count_messages("s-safe") == 2

        r3 = app.chat("失败后的下一句", chat_id="s-safe")
        assert r3["reply"] == "第3轮恢复正常"
        assert app.state.count_messages("s-safe") == 4

        gen = app.stream_chat("浏览器刷新中断", chat_id="s-safe")
        first_frame = next(gen)
        assert first_frame == {"type": "text_delta", "text": "断开前碎片"}
        gen.close()
        assert app.state.count_messages("s-safe") == 4

        r5 = app.chat("重连后的提问", chat_id="s-safe")
        assert r5["reply"] == "第5轮依然正常"
        assert app.state.count_messages("s-safe") == 6

        with pytest.raises(RuntimeError, match="second step 502"):
            app.chat("记住，中断轮偏好", chat_id="s-safe")
        assert app.state.count_messages("s-safe") == 6

        r7 = app.chat("记住，重连后新偏好", chat_id="s-safe")
        assert r7["reply"] == "第7轮记忆已保存"
        facts_text = app.memory.read_recent_facts()
        assert "重连后新偏好" in facts_text
        assert app.state.count_messages("s-safe") == 8
    finally:
        app.close()

    # 第二道防线：即使 SQLite 中已存在历史遗留的连续孤儿 user 消息，拼装上下文也绝不发连续两条 role == "user"
    dirty_llm = StrictAlternatingLLM([{"role": "assistant", "content": "已合并历史孤儿并正常答复"}])
    dirty_app = AssistantApp(data_root=tmp_path / "dirty_data", llm=dirty_llm)
    try:
        dirty_app.state.append_message("s-dirty", "user", "遗留孤儿1")
        dirty_app.state.append_message("s-dirty", "user", "遗留孤儿2")
        merged_hist = AssistantApp._build_history_messages(
            dirty_app.state.active_messages("s-dirty")
        )
        assert len(merged_hist) == 1
        assert merged_hist[0]["role"] == "user"
        assert "遗留孤儿1" in str(merged_hist[0]["content"])
        assert "遗留孤儿2" in str(merged_hist[0]["content"])

        res = dirty_app.chat("最新提问", chat_id="s-dirty")
        assert res["reply"] == "已合并历史孤儿并正常答复"
        sent = dirty_llm.requests[0]
        assert [m["role"] for m in sent] == ["system", "user"]
        assert "遗留孤儿1" in str(sent[1]["content"])
        assert "最新提问" in str(sent[1]["content"])
    finally:
        dirty_app.close()


# 测试同一 chat_id 并发 chat() 严格串行有序落库，不同 chat_id 互不阻塞，且锁/提示词缓存有界
def test_per_chat_id_mutex_serialization_and_bounded_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    import personal_assistant.assistant.app as app_mod

    monkeypatch.setattr(app_mod, "MAX_SESSION_PROMPTS", 4)
    monkeypatch.setattr(app_mod, "MAX_SESSION_LOCKS", 4)

    lock = threading.Lock()
    active_same = 0
    max_active_same = 0
    same_started = threading.Event()
    other_entered = threading.Event()
    release_same = threading.Event()

    # 记录同会话最大并发度，并验证跨会话不互相阻塞
    class ConcurrencyProbeLLM(LLMClient):
        # 根据消息内容模拟慢调用与并发观测
        def respond(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
            nonlocal active_same, max_active_same
            content = str(messages[-1]["content"])
            if content.startswith("same-"):
                with lock:
                    active_same += 1
                    max_active_same = max(max_active_same, active_same)
                same_started.set()
                release_same.wait(timeout=3.0)
                time.sleep(0.02)
                with lock:
                    active_same -= 1
                return {"role": "assistant", "content": f"reply:{content}"}
            if content == "other-fast":
                other_entered.set()
            return {"role": "assistant", "content": f"reply:{content}"}

    app = AssistantApp(data_root=tmp_path / "data", llm=ConcurrencyProbeLLM())
    try:
        errors: list[BaseException] = []

        # 在子线程执行目标会话对话
        def run_chat(msg: str, cid: str) -> None:
            try:
                app.chat(msg, chat_id=cid)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=run_chat, args=("same-1", "chat-same"))
        t2 = threading.Thread(target=run_chat, args=("same-2", "chat-same"))
        t1.start()
        assert same_started.wait(timeout=3.0)
        t2.start()

        # 同一 chat-same 被锁住期间，另一个会话 chat-other 应可立即执行而不被阻塞
        t_other = threading.Thread(target=run_chat, args=("other-fast", "chat-other"))
        t_other.start()
        assert other_entered.wait(timeout=2.0)
        t_other.join(timeout=3.0)

        release_same.set()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert not errors
        assert max_active_same == 1

        stored = app.state.active_messages("chat-same")
        assert [m["role"] for m in stored] == ["user", "assistant", "user", "assistant"]
        assert stored[1]["content"] == f"reply:{stored[0]['content']}"
        assert stored[3]["content"] == f"reply:{stored[2]['content']}"

        # 验证跨多个不同 chat_id 后，_session_prompts 与 _chat_locks 均保持在上限以内
        for i in range(10):
            app.chat(f"ping-{i}", chat_id=f"bounded-{i}")
        assert len(app._session_prompts) <= 4
        assert len(app._chat_locks) <= 4
    finally:
        release_same.set()
        app.close()


# 测试 today_fn 跨自然日自动清空并刷新冻结系统提示词中的日期，同日内多轮保持冻结不变
def test_cross_midnight_today_fn_refreshes_session_prompt_and_freezes_within_same_day(
    tmp_path: Path,
) -> None:
    from datetime import date

    current_day = date(2026, 11, 26)
    llm = RecordingLLM(
        [
            {"role": "assistant", "content": "第1天首轮"},
            {"role": "assistant", "content": "第1天次轮"},
            {"role": "assistant", "content": "第2天首轮"},
        ]
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        llm=llm,
        today_fn=lambda: current_day,
    )
    try:
        # 写入一条 11-26 当天有效、11-27 过期的临时记忆
        app.memory.apply_memory_operations(
            "schedule",
            "临时日程",
            "近期临时安排",
            [{"op": "add", "text": "2026-11-26 晚上技术分享", "until": "2026-11-26"}],
            today=current_day,
        )

        app.chat("今天有什么安排", chat_id="daily-1")
        prompt_day1_turn1 = str(llm.requests[0][0][0]["content"])
        assert "今天是 2026-11-26" in prompt_day1_turn1

        # 同日内新增记忆索引文件后，同日次轮仍保持冻结不变（保护 KV Cache）
        app.memory.remember_user_statement("同日内新增的一条事实", message_id=99)
        app.chat("同一天第二句", chat_id="daily-1")
        prompt_day1_turn2 = str(llm.requests[1][0][0]["content"])
        assert prompt_day1_turn2 == prompt_day1_turn1

        # 跨自然日切换到 2026-11-27：自动归档过期记忆并清空重建系统提示词
        current_day = date(2026, 11, 27)
        app.chat("新的一天早上好", chat_id="daily-1")
        prompt_day2 = str(llm.requests[2][0][0]["content"])
        assert "今天是 2026-11-27" in prompt_day2
        assert "今天是 2026-11-26" not in prompt_day2
        assert "data/memory/facts.md" in prompt_day2
        assert "晚上技术分享" not in (app.memory.memory_dir / "schedule.md").read_text(encoding="utf-8")
    finally:
        app.close()
