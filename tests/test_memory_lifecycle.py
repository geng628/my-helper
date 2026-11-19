"""记忆生命周期单元测试（TDD）：覆盖冲突替换、过期日边界过滤、自动归档与读旧改旧沉淀。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from personal_assistant.app import AssistantApp
from personal_assistant.consolidator import ConsolidationPolicy, MemoryConsolidator
from personal_assistant.db import StateStore
from personal_assistant.llm import LLMClient
from personal_assistant.memory import MemoryStore, is_line_expired


# 录制每次请求的预设回复模型，用于验证沉淀器传给模型的上下文内容
class CapturingLLM(LLMClient):
    # 保存预设回复并初始化请求记录
    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.requests: list[list[dict[str, object]]] = []

    # 记录请求列表并弹出下一条预设回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        self.requests.append([dict(m) for m in messages])
        return {"role": "assistant", "content": self._contents.pop(0)}


# 测试写入期冲突消解：在极短文件上原地 replace、remove、跳过已过期新增以及拦截保留主题名
def test_memory_upsert_replace_remove_and_reserved_topics(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "data")
    day1 = date(2026, 11, 19)
    day2 = date(2026, 11, 20)

    # 新增一条长期事实和一条临时事实，同时尝试新增一条已过期事实（应被跳过）
    saved = memory.apply_memory_operations(
        "projects",
        "项目备忘",
        "极光项目配置与进展",
        [
            {"op": "add", "text": "极光后端端口定为 9091。"},
            {"op": "add", "text": "- [2026-11-19] 极光日志采用标准库 logging。"},
            {"op": "add", "text": "2026-11-25 下午三点评审极光架构。", "until": "2026-11-26"},
            {"op": "add", "text": "早就结束的旧任务。", "until": "2026-11-01"},
        ],
        today=day1,
    )
    assert saved == "data/memory/projects.md"
    content1 = (memory.memory_dir / "projects.md").read_text(encoding="utf-8")
    lines1 = content1.splitlines()
    assert "- [2026-11-19] 极光后端端口定为 9091。" in lines1
    assert "- [2026-11-19] 极光日志采用标准库 logging。" in lines1
    assert "- [2026-11-19] (until: 2026-11-26) 2026-11-25 下午三点评审极光架构。" in lines1
    assert "早就结束的旧任务" not in content1

    # 文件只有 2 条记录时即发生端口冲突：replace 必须原地替换旧行，不留 9091
    memory.apply_memory_operations(
        "projects",
        "项目备忘",
        "极光项目配置与进展",
        [
            {"op": "replace", "old": "端口定为 9091", "new": "极光后端端口改为 9092。"},
            {"op": "replace", "old": "不存在的旧描述", "new": "极光前端暂定命令行交互。"},
        ],
        today=day2,
    )
    content2 = (memory.memory_dir / "projects.md").read_text(encoding="utf-8")
    assert "9091" not in content2
    assert "极光后端端口改为 9092。" in content2
    assert "极光前端暂定命令行交互。" in content2

    # remove 精确删除作废条目，并写入审计日志
    memory.apply_memory_operations(
        "projects",
        "项目备忘",
        "极光项目配置与进展",
        [{"op": "remove", "old": "极光前端暂定命令行交互"}],
        today=day2,
    )
    content3 = (memory.memory_dir / "projects.md").read_text(encoding="utf-8")
    assert "极光前端暂定命令行交互" not in content3
    assert "极光后端端口改为 9092。" in content3
    assert any(e["action"] == "consolidate_memory" for e in memory.audit_entries())

    # 保留主题名（含 archive）必须被拒绝
    for reserved in ("memory", "rules", "audit", "archive"):
        with pytest.raises(ValueError):
            memory.apply_memory_operations(
                reserved,
                "非法",
                "不应允许",
                [{"op": "add", "text": "x"}],
                today=day2,
            )


# 测试 (until: YYYY-MM-DD) 过期日边界、检索期零成本过滤与自动归档至 archive/
def test_expiry_boundary_filtering_and_archival(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "data")
    temp_line = "- [2026-11-19] (until: 2026-11-26) 2026-11-25 下午三点答辩。"
    perm_line = "- [2026-11-19] 偏好使用 Python 3.12 标准库。"

    # 边界语义：until 当天（2026-11-26）仍有效，次日（2026-11-27）才算过期
    assert is_line_expired(temp_line, today=date(2026, 11, 25)) is False
    assert is_line_expired(temp_line, today=date(2026, 11, 26)) is False
    assert is_line_expired(temp_line, today=date(2026, 11, 27)) is True
    assert is_line_expired(perm_line, today=date(2099, 1, 1)) is False

    memory.apply_memory_operations(
        "facts",
        "事实备忘",
        "长期事实与临时日程",
        [
            {"op": "add", "text": "2026-11-25 下午三点答辩。", "until": "2026-11-26"},
            {"op": "add", "text": "偏好使用 Python 3.12 标准库。"},
        ],
        today=date(2026, 11, 19),
    )

    # 有效期内可正常检索到；过期后即使尚未归档，search_hits 也立刻屏蔽过期行
    hits_valid = memory.search_hits("答辩", root="data/memory", today=date(2026, 11, 26))
    assert len(hits_valid) == 1
    hits_expired = memory.search_hits("答辩", root="data/memory", today=date(2026, 11, 27))
    assert hits_expired == []
    assert len(memory.search_hits("Python 3.12", root="data/memory", today=date(2026, 11, 27))) == 1

    # 执行归档：过期行从 facts.md 移至 data/memory/archive/2026-11.md，永久行不受影响
    archived = memory.archive_expired_memories(today=date(2026, 11, 27))
    assert archived == {"data/memory/facts.md": 1}
    active_text = (memory.memory_dir / "facts.md").read_text(encoding="utf-8")
    assert "答辩" not in active_text
    assert "Python 3.12" in active_text

    archive_file = memory.memory_dir / "archive" / "2026-11.md"
    assert archive_file.is_file()
    archive_text = archive_file.read_text(encoding="utf-8")
    assert "答辩" in archive_text
    assert "facts.md" in archive_text

    # 默认检索 data/memory 不会混入 archive/ 里的陈年旧事，显式搜 data/memory/archive 可追溯
    assert memory.search_hits("答辩", root="data/memory", today=date(2026, 11, 27)) == []
    assert len(memory.search_hits("答辩", root="data/memory/archive", today=date(2026, 11, 27))) == 1
    assert any(e["action"] == "archive_expired" for e in memory.audit_entries())


# 测试沉淀器读旧改旧：喂入已有主题文件与当前日期，完成端口替换与带有效期临时事实写入
def test_consolidator_reads_existing_topics_resolves_conflicts_and_sets_expiry(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "data" / "state" / "chat.sqlite3")
    memory = MemoryStore(tmp_path / "data")
    try:
        memory.apply_memory_operations(
            "projects",
            "项目备忘",
            "极光项目配置",
            [{"op": "add", "text": "极光后端端口定为 9091。"}],
            today=date(2026, 11, 19),
        )
        for u, a in [
            ("极光端口改成 9092 了，9091 不用啦", "好的，记住改成 9092"),
            ("另外 11 月 25 号下午三点要开架构评审会", "明白"),
            ("今天先到这里", "好的"),
        ]:
            state.append_message("life-1", "user", u)
            state.append_message("life-1", "assistant", a)

        llm = CapturingLLM(
            [
                json.dumps(
                    {
                        "session_summary": "极光后端端口已改为 9092，2026-11-25 下午三点架构评审。",
                        "memories": [
                            {
                                "topic": "projects",
                                "title": "项目备忘",
                                "index_summary": "极光项目配置与进展",
                                "operations": [
                                    {
                                        "op": "replace",
                                        "old": "极光后端端口定为 9091",
                                        "new": "极光后端端口改为 9092。",
                                    },
                                    {
                                        "op": "add",
                                        "text": "2026-11-25 下午三点极光架构评审会。",
                                        "until": "2026-11-26",
                                    },
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            ]
        )
        policy = ConsolidationPolicy(max_active_messages=6, keep_recent_messages=2, min_batch_messages=4)
        consolidator = MemoryConsolidator(state, memory, llm, policy=policy)
        result = consolidator.consolidate_once("life-1", today=date(2026, 11, 20))
        assert result is not None

        # 验证沉淀器确实把当前日期与已有 projects.md 正文喂给了模型，支持读旧改旧
        prompt_user = str(llm.requests[0][1]["content"])
        assert "2026-11-20" in prompt_user
        assert "极光后端端口定为 9091" in prompt_user

        projects_text = (memory.memory_dir / "projects.md").read_text(encoding="utf-8")
        assert "9091" not in projects_text
        assert "极光后端端口改为 9092。" in projects_text
        assert "(until: 2026-11-26) 2026-11-25 下午三点极光架构评审会。" in projects_text
    finally:
        state.close()


# 测试 AssistantApp 对话启动时零成本自动归档已过期临时记忆，无需手动触发
def test_assistant_app_auto_archives_expired_memories_on_new_day(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    memory = MemoryStore(data_root)
    memory.apply_memory_operations(
        "facts",
        "事实备忘",
        "临时日程与事实",
        [
            {"op": "add", "text": "历史临时会议。", "until": "2020-01-02"},
            {"op": "add", "text": "永久技术偏好：纯标准库。"},
        ],
        today=date(2020, 1, 1),
    )

    llm = CapturingLLM(["好的"])
    app = AssistantApp(data_root=data_root, llm=llm, memory=memory)
    try:
        app.chat("你好", chat_id="daily-1")
        active_facts = (memory.memory_dir / "facts.md").read_text(encoding="utf-8")
        assert "历史临时会议" not in active_facts
        assert "永久技术偏好：纯标准库" in active_facts
        archive_files = list((memory.memory_dir / "archive").glob("*.md"))
        assert len(archive_files) == 1
        assert "历史临时会议" in archive_files[0].read_text(encoding="utf-8")
    finally:
        app.close()
