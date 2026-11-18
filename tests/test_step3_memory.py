"""Unit tests for Step 3: plain Markdown memory, knowledge search/read, and path safety."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from personal_assistant.commands import ToolsConfig
from personal_assistant.llm import LLMClient
from personal_assistant.memory import FilesPolicy, MemoryStore, memory_tools
from personal_assistant.react import run_react


class ScriptedLLM(LLMClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._responses.pop(0)


def test_path_security_and_symlink_handling(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = ToolsConfig.load(repo_root / "config" / "tools.json", project_root=repo_root)
    data_root = tmp_path / "data"
    policy = FilesPolicy.from_dict(cfg.files, project_root=tmp_path, data_root=data_root)
    memory = MemoryStore(data_root, policy=policy)

    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (data_root / "knowledge" / "note.md").write_text("# Note\nhi\n", encoding="utf-8")

    # Traversal, absolute path, and sensitive segments blocked
    for bad in ("../.env", "/etc/passwd", ".env", "data/state/db.sqlite3", "data/.git/HEAD"):
        with pytest.raises(ValueError):
            memory.resolve_path(bad, write=False)
        with pytest.raises(ValueError):
            memory.resolve_path(bad, write=True)

    # Inner symlink escaping read root is blocked
    os.symlink(tmp_path / ".env", data_root / "knowledge" / "escape.md")
    with pytest.raises(ValueError):
        memory.resolve_path("data/knowledge/escape.md", write=False)

    # Writing outside workspace is blocked; writing inside workspace is allowed
    for readonly in ("data/knowledge/x.md", "knowledge/x.md", "data/memory/MEMORY.md", "docs/x.md"):
        with pytest.raises(ValueError, match="只能写入"):
            memory.resolve_path(readonly, write=True)

    assert memory.resolve_path("drafts/a.md", write=True) == (
        data_root / "workspace" / "drafts" / "a.md"
    ).resolve()
    assert memory.resolve_path("data/knowledge/note.md", write=False) == (
        data_root / "knowledge" / "note.md"
    ).resolve()


def test_symlinked_knowledge_root_works_safely(tmp_path: Path) -> None:
    shared_notes = tmp_path / "shared_v1_notes"
    (shared_notes / "ostep").mkdir(parents=True)
    (shared_notes / "ostep" / "lock.md").write_text("# 锁\n持有自旋锁时禁止 sleep。\n", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("TOP_SECRET", encoding="utf-8")
    os.symlink(outside, shared_notes / "leak.md")

    data_root = tmp_path / "v2" / "data"
    (data_root / "memory" / "rules").mkdir(parents=True)
    (data_root / "workspace").mkdir(parents=True)
    os.symlink(shared_notes, data_root / "knowledge")

    memory = MemoryStore(data_root)
    assert "data/knowledge/ostep/lock.md" in memory.list_files("data/knowledge", "*.md")
    hits = memory.search_markdown("自旋锁", root="data/knowledge")
    assert "data/knowledge/ostep/lock.md:2" in hits
    assert "TOP_SECRET" not in hits
    with pytest.raises(ValueError):
        memory.read_markdown("data/knowledge/leak.md", 1, 10)


def test_core_memory_rules_and_workspace_backup(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "data")
    assert memory.read_core_memory() == ""
    assert memory.read_rules_summary() == ""

    # Remember user statement with idempotency
    assert memory.remember_user_statement("记住，我喜欢早上阅读论文", message_id=7) == "data/memory/MEMORY.md"
    memory.remember_user_statement("记住，我喜欢早上阅读论文", message_id=7)
    core = memory.read_core_memory()
    assert core.count("(msg#7)") == 1
    assert "早上阅读论文" in core

    # Truncation keeps header and recent items
    for i in range(25):
        memory.remember_user_statement(f"偏好 {i}：" + "内容" * 30, message_id=100 + i)
    truncated = memory.read_core_memory(max_chars=500)
    assert "完整记录请调用 read_markdown" in truncated
    assert "偏好 24" in truncated

    # Save rule and list summary
    rule_path = memory.save_rule("回答尽量简洁分条", source="agent")
    assert f"data/memory/rules/{rule_path.name} — 回答尽量简洁分条" in memory.read_rules_summary()

    # Workspace create, conflict guard, overwrite + backup
    assert memory.write_workspace_file("out.md", "v1", mode="create") == "data/workspace/out.md"
    with pytest.raises(ValueError, match="已存在"):
        memory.write_workspace_file("out.md", "v2", mode="create")
    memory.write_workspace_file("out.md", "v2", mode="overwrite")
    assert (memory.workspace_dir / "out.md").read_text(encoding="utf-8") == "v2"
    backups = list((memory.workspace_dir / ".backups").glob("*-out.md"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "v1"


def test_memory_tools_and_react_integration(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = ToolsConfig.load(repo_root / "config" / "tools.json", project_root=repo_root)
    data_root = tmp_path / "data"
    policy = FilesPolicy.from_dict(cfg.files, project_root=tmp_path, data_root=data_root)
    memory = MemoryStore(data_root, policy=policy)
    note_dir = memory.knowledge_dir / "ostep"
    note_dir.mkdir(parents=True)
    (note_dir / "28.lock.md").write_text(
        "# 第28章 锁\n\n## 自旋锁\n持有自旋锁时不能 sleep。\n第五行内容。\n",
        encoding="utf-8",
    )

    current_msg = ("记住，我正在复习 OSTEP", 99)
    registry = cfg.build_registry(*memory_tools(memory, get_user_message=lambda: current_msg))
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
                            "arguments": '{"query": "自旋锁 sleep", "root": "data/knowledge"}',
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
                            "arguments": '{"path": "data/knowledge/ostep/28.lock.md", "start_line": "1", "end_line": "5"}',
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
                        "function": {"name": "save_rule", "arguments": '{"statement": "笔记问题先读原文再答"}'},
                    },
                    {
                        "id": "c5",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "summary.md", "content": "自旋锁不可休眠", "mode": "create"}',
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "根据 OSTEP 第28章，持有自旋锁时不能 sleep；我也已记下你的偏好与规则并写入 summary.md。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "自旋锁能 sleep 吗？"}]
    reply = run_react(llm, messages, registry)

    assert "不能 sleep" in reply
    assert "正在复习 OSTEP" in memory.read_core_memory()
    assert "笔记问题先读原文再答" in memory.read_rules_summary()
    assert (memory.workspace_dir / "summary.md").read_text(encoding="utf-8") == "自旋锁不可休眠"
    assert len(memory.audit_entries()) == 3
