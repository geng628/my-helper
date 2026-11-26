"""Step 3 单元测试：纯 Markdown 记忆、规则、工作区写入与路径安全边界。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.ai import LLMClient
from personal_assistant.assistant.memory import FilesPolicy, MemoryStore, MemoryToolsetProvider
from personal_assistant.agent.react import run_react


# 预录回复的假模型，用于确定性验证记忆工具与 ReAct 联动
class ScriptedLLM(LLMClient):
    # 保存预设回复队列
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    # 按序弹出下一条预设回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._responses.pop(0)


# 测试读写路径安全边界：拦截目录穿越、敏感段、越界软链接及只读目录写入
def test_path_security_and_symlink_handling(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = ToolsConfig.load(repo_root / "config" / "tools.json", project_root=repo_root)
    data_root = tmp_path / "data"
    policy = FilesPolicy.from_dict(cfg.files, project_root=tmp_path, data_root=data_root)
    memory = MemoryStore(data_root, policy=policy)

    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (data_root / "memory" / "note.md").write_text("# Note\nhi\n", encoding="utf-8")

    # 拦截路径穿越、绝对路径与敏感目录段
    for bad in ("../.env", "/etc/passwd", ".env", "data/state/db.sqlite3", "data/.git/HEAD"):
        with pytest.raises(ValueError):
            memory.resolve_path(bad, write=False)
        with pytest.raises(ValueError):
            memory.resolve_path(bad, write=True)

    # 拦截指向白名单外部的符号链接
    os.symlink(tmp_path / ".env", data_root / "memory" / "escape.md")
    with pytest.raises(ValueError):
        memory.resolve_path("data/memory/escape.md", write=False)

    # 禁止向非工作区目录写入；允许写入 data/workspace
    for readonly in ("data/memory/MEMORY.md", "memory/MEMORY.md", "docs/x.md"):
        with pytest.raises(ValueError, match="只能写入"):
            memory.resolve_path(readonly, write=True)

    assert memory.resolve_path("drafts/a.md", write=True) == (
        data_root / "workspace" / "drafts" / "a.md"
    ).resolve()
    assert memory.resolve_path("data/memory/note.md", write=False) == (
        data_root / "memory" / "note.md"
    ).resolve()


# 测试记忆索引截断、facts.md 幂等追加不膨胀索引、规则落盘以及工作区覆盖前自动备份
def test_core_memory_rules_and_workspace_backup(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "data")
    assert memory.read_memory_index() == ""
    assert memory.read_rules_summary() == ""
    assert memory.read_recent_facts() == ""

    # 事实记忆写入 facts.md，同 message_id 幂等去重，且只在 MEMORY.md 注册一次索引指针
    assert memory.remember_user_statement("记住，我喜欢早上写代码", message_id=7) == "data/memory/facts.md"
    memory.remember_user_statement("记住，我喜欢早上写代码", message_id=7)
    facts_text = memory.read_recent_facts()
    assert facts_text.count("(msg#7)") == 1
    assert "早上写代码" in facts_text

    # 多次追加事实后，MEMORY.md 索引内容保持字节级不变（不膨胀、不打穿 KV Cache）
    index_after_first = memory.memory_file.read_text(encoding="utf-8")
    assert "data/memory/facts.md" in index_after_first
    assert "早上写代码" not in index_after_first

    for i in range(25):
        memory.remember_user_statement(f"事实 {i}：测试记录", message_id=100 + i)
    assert memory.memory_file.read_text(encoding="utf-8") == index_after_first
    assert "事实 24" in memory.read_recent_facts(limit=5)

    # 手工维护的超长 MEMORY.md 索引按头部行数/字符数截断（保留前部目录）
    memory.memory_file.write_text(
        "# 记忆索引\n" + "\n".join(f"- [主题 {i}](topic_{i}.md) — 说明" for i in range(120)),
        encoding="utf-8",
    )
    truncated = memory.read_memory_index(max_lines=10)
    assert "主题 0" in truncated
    assert "主题 50" not in truncated
    assert "索引已截断" in truncated

    # 保存行为规则并汇总列表
    rule_path = memory.save_rule("回答尽量简洁分条", source="agent")
    assert f"data/memory/rules/{rule_path.name} — 回答尽量简洁分条" in memory.read_rules_summary()

    # 工作区创建、冲突保护以及 overwrite 自动备份
    assert memory.write_workspace_file("out.md", "first", mode="create") == "data/workspace/out.md"
    with pytest.raises(ValueError, match="已存在"):
        memory.write_workspace_file("out.md", "second", mode="create")
    memory.write_workspace_file("out.md", "second", mode="overwrite")
    assert (memory.workspace_dir / "out.md").read_text(encoding="utf-8") == "second"
    backups = list((memory.workspace_dir / ".backups").glob("*-out.md"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "first"


# 测试内置记忆/文件工具在 ReAct 循环中的完整检索、读取、记忆与写入联动
def test_memory_tools_and_react_integration(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = ToolsConfig.load(repo_root / "config" / "tools.json", project_root=repo_root)
    data_root = tmp_path / "data"
    policy = FilesPolicy.from_dict(cfg.files, project_root=tmp_path, data_root=data_root)
    memory = MemoryStore(data_root, policy=policy)
    (memory.memory_dir / "guide.md").write_text(
        "# 安全规范\n\n## 命令执行\n外部命令必须使用固定 argv 数组并设置 shell=False。\n第五行内容。\n",
        encoding="utf-8",
    )

    current_msg = ("记住，回复优先给 Python 标准库示例", 99)
    registry = cfg.build_registry(providers=[MemoryToolsetProvider(memory, get_user_message=lambda: current_msg)])
    assert set(cfg.enabled_tools).issubset(set(registry.names))

    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "search_markdown",
                            "arguments": '{"query": "固定 argv shell=False", "root": "data/memory"}',
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "read_markdown",
                            "arguments": '{"path": "data/memory/guide.md", "start_line": "1", "end_line": "5"}',
                        },
                    },
                    {
                        "id": "c3",
                        "type": "function",
                        "function": {"name": "remember_fact", "arguments": "{}"},
                    },
                    {
                        "id": "c4",
                        "type": "function",
                        "function": {"name": "save_rule", "arguments": '{"statement": "回答前先查证已有记录"}'},
                    },
                    {
                        "id": "c5",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "summary.md", "content": "命令必须设置 shell=False", "mode": "create"}',
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "规范要求外部命令使用固定 argv 并设置 shell=False；我也已记下你的偏好与规则并写入 summary.md。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "外部命令执行有什么安全要求？"}]
    reply = run_react(llm, messages, registry)

    assert "shell=False" in reply
    assert "Python 标准库示例" in memory.read_recent_facts()
    assert "data/memory/facts.md" in memory.read_memory_index()
    assert "回答前先查证已有记录" in memory.read_rules_summary()
    assert (memory.workspace_dir / "summary.md").read_text(encoding="utf-8") == "命令必须设置 shell=False"
    assert len(memory.audit_entries()) == 3


# 测试 MemoryStore 并发写锁：多线程交错调用 remember_user_statement、apply_memory_operations、save_rule 与 record_audit 零丢失
def test_memory_store_concurrent_writes_have_zero_lost_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import threading
    import time
    from datetime import date

    import personal_assistant.assistant.memory as memory_mod

    # 1. 确定性竞态测试：在第一次写入 facts.md 时停顿 50ms，验证并发第二写不会在无锁时覆盖丢失第一条事实
    race_mem = MemoryStore(tmp_path / "race_data")
    assert hasattr(race_mem, "_write_lock")
    orig_atomic_write = memory_mod._atomic_write
    first_entered = threading.Event()

    # 在首次写 facts.md 时制造并发窗口以暴露无锁覆盖丢失
    def delayed_atomic_write(target: Path, content: str) -> None:
        if target == race_mem.facts_file and not first_entered.is_set():
            first_entered.set()
            time.sleep(0.05)
        orig_atomic_write(target, content)

    monkeypatch.setattr(memory_mod, "_atomic_write", delayed_atomic_write)

    errors: list[BaseException] = []

    # 在子线程写入第一条事实
    def write_first() -> None:
        try:
            race_mem.remember_user_statement("并发第一条关键事实", message_id=1)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    # 等待第一条进入写窗口后再并发写入第二条事实
    def write_second() -> None:
        try:
            assert first_entered.wait(timeout=2.0)
            race_mem.remember_user_statement("并发第二条关键事实", message_id=2)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=write_first)
    t2 = threading.Thread(target=write_second)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert not errors
    facts_content = race_mem.facts_file.read_text(encoding="utf-8")
    assert "并发第一条关键事实" in facts_content
    assert "并发第二条关键事实" in facts_content

    monkeypatch.setattr(memory_mod, "_atomic_write", orig_atomic_write)

    # 2. 高并发多线程压测：混合写入 facts.md / 主题文件 / rules / audit.jsonl，校验零丢行且 JSONL 全部合法
    stress_mem = MemoryStore(tmp_path / "stress_data")
    barrier = threading.Barrier(8)
    stress_errors: list[BaseException] = []

    # 并发写入事实备忘
    def facts_worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for idx in range(15):
                stress_mem.remember_user_statement(
                    f"线程{worker_id}事实{idx}", message_id=worker_id * 100 + idx
                )
        except BaseException as exc:  # noqa: BLE001
            stress_errors.append(exc)

    # 并发读写改写同一主题 Markdown 与 MEMORY.md 索引
    def topic_worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for idx in range(15):
                stress_mem.apply_memory_operations(
                    "concurrent_topic",
                    "并发主题",
                    "并发测试主题索引",
                    [{"op": "add", "text": f"主题记录-{worker_id}-{idx}"}],
                    today=date(2026, 11, 26),
                )
        except BaseException as exc:  # noqa: BLE001
            stress_errors.append(exc)

    # 并发落盘规则文件
    def rule_worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for idx in range(10):
                stress_mem.save_rule(f"并发规则-{worker_id}-{idx}", source="agent")
        except BaseException as exc:  # noqa: BLE001
            stress_errors.append(exc)

    # 并发追加审计日志
    def audit_worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for idx in range(20):
                stress_mem.record_audit(
                    action="custom_audit",
                    actor=f"w{worker_id}",
                    target="data/memory/facts.md",
                    old_value=None,
                    new_value=f"item-{worker_id}-{idx}",
                )
        except BaseException as exc:  # noqa: BLE001
            stress_errors.append(exc)

    threads = [
        threading.Thread(target=facts_worker, args=(0,)),
        threading.Thread(target=facts_worker, args=(1,)),
        threading.Thread(target=topic_worker, args=(2,)),
        threading.Thread(target=topic_worker, args=(3,)),
        threading.Thread(target=rule_worker, args=(4,)),
        threading.Thread(target=rule_worker, args=(5,)),
        threading.Thread(target=audit_worker, args=(6,)),
        threading.Thread(target=audit_worker, args=(7,)),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=15.0)

    assert not stress_errors
    all_facts = stress_mem.facts_file.read_text(encoding="utf-8")
    for w in (0, 1):
        for idx in range(15):
            assert f"线程{w}事实{idx}" in all_facts

    topic_body = (stress_mem.memory_dir / "concurrent_topic.md").read_text(encoding="utf-8")
    for w in (2, 3):
        for idx in range(15):
            assert f"主题记录-{w}-{idx}" in topic_body

    index_body = stress_mem.memory_file.read_text(encoding="utf-8")
    assert index_body.count("data/memory/facts.md") == 1
    assert index_body.count("data/memory/concurrent_topic.md") == 1
    assert len(list(stress_mem.rules_dir.glob("*.md"))) == 20

    raw_audit_lines = [
        ln for ln in stress_mem.audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(raw_audit_lines) == 30 + 30 + 20 + 40
    parsed_entries = [json.loads(ln) for ln in raw_audit_lines]
    assert len(parsed_entries) == 120
