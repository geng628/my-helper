"""记忆生命周期单元测试（TDD）：覆盖冲突替换、过期日边界过滤、自动归档与读旧改旧沉淀。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.consolidator import ConsolidationPolicy, MemoryConsolidator
from personal_assistant.assistant.db import StateStore
from personal_assistant.ai import LLMClient
from personal_assistant.assistant.memory import MemoryStore, is_line_expired


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

    # 默认检索与列举 data/memory 不会混入 archive/ 里的陈年旧事，显式指向 data/memory/archive 可追溯
    assert memory.search_hits("答辩", root="data/memory", today=date(2026, 11, 27)) == []
    assert len(memory.search_hits("答辩", root="data/memory/archive", today=date(2026, 11, 27))) == 1
    assert "archive/" not in memory.list_files("data/memory", "*.md")
    assert "data/memory/archive/2026-11.md" in memory.list_files("data/memory/archive", "*.md")
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


# 测试沉淀时自动清理 facts.md 中已被规范化沉淀的原始 (msg#...) 幽灵行，同时绝不误删未沉淀的无关原话
def test_consolidator_cleans_up_raw_remember_fact_ghost_without_losing_unrelated_facts(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "data" / "state" / "chat.sqlite3")
    memory = MemoryStore(tmp_path / "data")
    try:
        m1 = state.append_message("ghost-1", "user", "记住，我养了一只叫橘猫的宠物。")
        memory.remember_user_statement("记住，我养了一只叫橘猫的宠物。", message_id=m1)
        state.append_message("ghost-1", "assistant", "好的，记下了。")
        m3 = state.append_message("ghost-1", "user", "另外明天下午两点我要参加版本发布演练会。")
        memory.remember_user_statement("另外明天下午两点我要参加版本发布演练会。", message_id=m3)
        state.append_message("ghost-1", "assistant", "记下了。")
        state.append_message("ghost-1", "user", "今天就先到这里。")
        state.append_message("ghost-1", "assistant", "好的。")

        # 刻意模拟模型只输出 add（忘了 replace 旧裸行）的最坏情况，验证程序兜底清理幽灵副本
        llm = CapturingLLM(
            [
                json.dumps(
                    {
                        "session_summary": "用户明天下午两点有版本发布演练会。",
                        "memories": [
                            {
                                "topic": "facts",
                                "title": "事实备忘",
                                "index_summary": "临时日程与事实",
                                "operations": [
                                    {
                                        "op": "add",
                                        "text": "2026-11-20 下午两点参加版本发布演练会。",
                                        "until": "2026-11-20",
                                    }
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
        consolidator.consolidate_once("ghost-1", today=date(2026, 11, 19))

        facts_text = (memory.memory_dir / "facts.md").read_text(encoding="utf-8")
        # 已被规范化的 msg#3 裸原话必须被清除，不留永不过期的幽灵副本
        assert f"(msg#{m3})" not in facts_text
        assert "(until: 2026-11-20)" in facts_text
        # 未被本次沉淀提及的无关原话 msg#1（橘猫）必须完好保留
        assert "橘猫" in facts_text
        # 到了过期日后，搜索“演练”必须完全为空
        assert memory.search_hits("演练", root="data/memory", today=date(2026, 11, 25)) == []
    finally:
        state.close()


# 测试 op=replace / op=remove 软退休至 archive/YYYY-MM.md 并记录 superseded_by 替代链（同时保持热记忆纯净与 MEMORY.md 幂等）
def test_replace_and_remove_soft_retire_with_superseded_chain_and_fts5_traceability(
    tmp_path: Path,
) -> None:
    from plugins.fts5_retriever import FTS5Retriever

    retriever = FTS5Retriever()
    memory = MemoryStore(tmp_path / "data", retriever=retriever)
    day1 = date(2023, 5, 10)
    day2 = date(2023, 10, 24)

    # 1. replace 更新团队规模：活跃 work.md 只保留最新值（5 人），旧值（37 人）软退休进 archive 并记录替代链
    memory.apply_memory_operations(
        "work",
        "工作备忘",
        "团队规模与职责",
        [
            {"op": "add", "text": "刚升任架构师，初期带领 37 名工程师。"},
            {"op": "add", "text": "临时兼任季度招聘接口人。"},
        ],
        today=day1,
    )

    memory.apply_memory_operations(
        "work",
        "工作备忘",
        "团队规模与职责",
        [
            {
                "op": "replace",
                "old": "带领 37 名工程师",
                "new": "担任架构师，目前带领 52 名工程师。",
            }
        ],
        today=day2,
    )

    active_work = (memory.memory_dir / "work.md").read_text(encoding="utf-8")
    assert "37 名工程师" not in active_work
    assert "52 名工程师" in active_work

    archive_file = memory.memory_dir / "archive" / "2023-10.md"
    assert archive_file.is_file()
    archive_text_1 = archive_file.read_text(encoding="utf-8")
    assert "[superseded 2023-10-24 | from: work.md]" in archive_text_1
    assert "带领 37 名工程师" in archive_text_1
    assert "-> superseded_by:" in archive_text_1
    assert "带领 52 名工程师" in archive_text_1

    # MEMORY.md 登记一次 archive 入口指针，且再次软退休时字节级不变（不破坏 KV Cache）
    index_after_replace = memory.memory_file.read_text(encoding="utf-8")
    assert "data/memory/archive" in index_after_replace

    # 2. remove 软退休作废条目并标记 -> superseded_by: (removed)
    memory.apply_memory_operations(
        "work",
        "工作备忘",
        "团队规模与职责",
        [{"op": "remove", "old": "季度招聘接口人"}],
        today=day2,
    )
    archive_text_2 = archive_file.read_text(encoding="utf-8")
    assert "季度招聘接口人" in archive_text_2
    assert "-> superseded_by: (removed)" in archive_text_2
    assert memory.memory_file.read_text(encoding="utf-8") == index_after_replace

    # 3. 默认热检索（FTS5 + 线性）不返回已退休旧值；显式检索 root="data/memory/archive" 可查到完整替代链
    assert all("37" not in h.snippet for h in memory.search_hits("37", root="data/memory", today=day2))
    assert all(
        "季度招聘接口人" not in h.snippet
        for h in memory.search_hits("季度招聘接口人", root="data/memory", today=day2)
    )
    fts_archive_hits = memory.search_hits("37", root="data/memory/archive", today=day2)
    assert len(fts_archive_hits) == 1
    assert "superseded_by:" in fts_archive_hits[0].snippet
    assert "52 名工程师" in fts_archive_hits[0].snippet

    linear_memory = MemoryStore(tmp_path / "data", create_dirs=False, retriever=None)
    assert linear_memory.search_hits("37", root="data/memory", today=day2) == []
    linear_archive_hits = linear_memory.search_hits("37", root="data/memory/archive", today=day2)
    assert len(linear_archive_hits) == 1
    assert "37 名工程师" in linear_archive_hits[0].snippet

    # 4. facts.md 中的原始 (msg#...) 裸原话被规范化替换时，不往 archive/ 写入冗余副本
    memory.remember_user_statement("我偏好深色主题。", message_id=42)
    memory.apply_memory_operations(
        "facts",
        "事实备忘",
        "通用事实",
        [{"op": "remove", "old": "(msg#42)"}],
        today=day2,
    )
    assert "(msg#42)" not in archive_file.read_text(encoding="utf-8")


# 测试 op=merge 无损拼接互补维度：保留原时间与原事实正文，支持中英文标点与幂等去重，且不触发 superseded 软退休
def test_merge_operation_enriches_existing_line_without_losing_old_facts_or_retiring(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "data")
    day1 = date(2023, 5, 21)
    day2 = date(2023, 5, 24)

    # 1. 英文 merge：实际完赛成绩与目标时间属互补指标，保留 [2023-05-21] 与两个时间，不写 archive/
    memory.apply_memory_operations(
        "fitness",
        "运动健康",
        "马拉松与训练目标",
        [{"op": "add", "text": "Completed first full marathon in 4:22:00."}],
        today=day1,
    )
    memory.apply_memory_operations(
        "fitness",
        "运动健康",
        "马拉松与训练目标",
        [{"op": "merge", "old": "marathon", "text": "Target time was 4:10:00."}],
        today=day2,
    )
    fitness_lines = [
        line
        for line in (memory.memory_dir / "fitness.md").read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("- ")
    ]
    assert len(fitness_lines) == 1
    assert "[2023-05-21]" in fitness_lines[0]
    assert "4:22:00" in fitness_lines[0]
    assert "; Target time was 4:10:00." in fitness_lines[0]
    assert not (memory.memory_dir / "archive").exists()

    # 2. 幂等重复 merge 不应重复追加相同补充内容
    changed = memory.apply_memory_operations(
        "fitness",
        "运动健康",
        "马拉松与训练目标",
        [{"op": "merge", "old": "marathon", "text": "Target time was 4:10:00."}],
        today=day2,
    )
    assert changed is None

    # 3. 中文 merge：按中文分号拼接，既有 until 标记保持完整且未传 until 时不丢失
    memory.apply_memory_operations(
        "projects",
        "项目备忘",
        "极光项目配置",
        [
            {"op": "add", "text": "极光后端端口定为 9091。"},
            {"op": "add", "text": "2023-05-28 举行架构评审。", "until": "2023-05-29"},
        ],
        today=day1,
    )
    memory.apply_memory_operations(
        "projects",
        "项目备忘",
        "极光项目配置",
        [
            {"op": "merge", "old": "极光后端", "text": "日志采用标准库 logging。"},
            {"op": "merge", "old": "架构评审", "text": "参会人员包括架构组。"},
            {"op": "merge", "old": "极光后端", "text": "周六前提交压测报告。", "until": "2023-05-27"},
            {"op": "merge", "old": "不存在的旧条目", "text": "极光数据库采用 SQLite。"},
        ],
        today=day2,
    )
    proj_text = (memory.memory_dir / "projects.md").read_text(encoding="utf-8")
    assert "- [2023-05-21] (until: 2023-05-27) 极光后端端口定为 9091；日志采用标准库 logging；周六前提交压测报告。" in proj_text
    assert "- [2023-05-21] (until: 2023-05-29) 2023-05-28 举行架构评审；参会人员包括架构组。" in proj_text
    assert "- [2023-05-24] 极光数据库采用 SQLite。" in proj_text
    assert not (memory.memory_dir / "archive").exists()


# 测试多次 merge 剥离尾部标点后的幂等去重、子句边界保护、HH:MM 与无日期行回退，以及多命中首条与过期跳过
def test_merge_multi_merge_idempotency_and_edge_cases(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "data")
    day1 = date(2023, 5, 21)
    day2 = date(2023, 5, 24)
    day3 = date(2023, 5, 25)

    # 1. 英文 3 段多次 merge：中间子句被剥离句号后，再次 merge 原句与中间子句必须严格幂等返回 None
    memory.apply_memory_operations(
        "fitness",
        "运动健康",
        "马拉松训练",
        [{"op": "add", "text": "Completed first full marathon in 4:22:00."}],
        today=day1,
    )
    memory.apply_memory_operations(
        "fitness",
        "运动健康",
        "马拉松训练",
        [{"op": "merge", "old": "marathon", "text": "Target time was 4:10:00."}],
        today=day2,
    )
    memory.apply_memory_operations(
        "fitness",
        "运动健康",
        "马拉松训练",
        [{"op": "merge", "old": "marathon", "text": "Average pace was 6:12/km."}],
        today=day3,
    )
    fitness_lines = [
        line
        for line in (memory.memory_dir / "fitness.md").read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("- ")
    ]
    assert len(fitness_lines) == 1
    assert "4:22:00" in fitness_lines[0]
    assert "Target time was 4:10:00" in fitness_lines[0]
    assert "Average pace was 6:12/km." in fitness_lines[0]

    assert (
        memory.apply_memory_operations(
            "fitness",
            "运动健康",
            "马拉松训练",
            [{"op": "merge", "old": "marathon", "text": "Target time was 4:10:00."}],
            today=day3,
        )
        is None
    )
    assert (
        memory.apply_memory_operations(
            "fitness",
            "运动健康",
            "马拉松训练",
            [{"op": "merge", "old": "marathon", "text": "Completed first full marathon in 4:22:00."}],
            today=day3,
        )
        is None
    )

    # 中文 3 段多次 merge 幂等性
    memory.apply_memory_operations(
        "projects",
        "项目备忘",
        "极光项目",
        [{"op": "add", "text": "极光后端端口定为 9091。"}],
        today=day1,
    )
    memory.apply_memory_operations(
        "projects",
        "项目备忘",
        "极光项目",
        [{"op": "merge", "old": "极光后端", "text": "日志采用标准库 logging。"}],
        today=day2,
    )
    memory.apply_memory_operations(
        "projects",
        "项目备忘",
        "极光项目",
        [{"op": "merge", "old": "极光后端", "text": "周六前提交压测报告。"}],
        today=day3,
    )
    assert (
        memory.apply_memory_operations(
            "projects",
            "项目备忘",
            "极光项目",
            [{"op": "merge", "old": "极光后端", "text": "日志采用标准库 logging。"}],
            today=day3,
        )
        is None
    )

    # 2. 非子串误判边界保护："12." 不得误命中 "12.5 km."，"Port 9091." 不得误命中 "Port 90910."
    memory.apply_memory_operations(
        "runs",
        "跑步记录",
        "距离与端口测试",
        [
            {"op": "add", "text": "Distance was 12.5 km."},
            {"op": "add", "text": "Port 90910."},
        ],
        today=day1,
    )
    assert (
        memory.apply_memory_operations(
            "runs",
            "跑步记录",
            "距离与端口测试",
            [{"op": "merge", "old": "Distance", "text": "12."}],
            today=day2,
        )
        == "data/memory/runs.md"
    )
    assert (
        memory.apply_memory_operations(
            "runs",
            "跑步记录",
            "距离与端口测试",
            [{"op": "merge", "old": "Port", "text": "Port 9091."}],
            today=day2,
        )
        == "data/memory/runs.md"
    )
    runs_text = (memory.memory_dir / "runs.md").read_text(encoding="utf-8")
    assert "Distance was 12.5 km; 12." in runs_text
    assert "Port 90910; Port 9091." in runs_text

    # 3. 带 HH:MM 时间戳与无日期条目附加 until 时不双括号，且无日期行使用 ref_day 盖戳
    events_path = memory.memory_dir / "events.md"
    events_path.write_text(
        "# 活动 (events.md)\n\n- [2023-05-21 14:30] 参加技术分享。\n- 临时备忘。\n",
        encoding="utf-8",
    )
    memory.apply_memory_operations(
        "events",
        "活动",
        "活动日程",
        [
            {"op": "merge", "old": "技术分享", "text": "地点在 3 楼。", "until": "2023-05-27"},
            {"op": "merge", "old": "临时备忘", "text": "补充细节。", "until": "2023-05-27"},
        ],
        today=day2,
    )
    events_lines = events_path.read_text(encoding="utf-8").splitlines()
    assert "- [2023-05-21 14:30] (until: 2023-05-27) 参加技术分享；地点在 3 楼。" in events_lines
    assert "- [2023-05-24] (until: 2023-05-27) 临时备忘；补充细节。" in events_lines

    # 4. 多条命中时只丰富首条 matched_indices[0]，以及已过期 until < ref_day 时跳过 merge
    memory.apply_memory_operations(
        "training",
        "训练记录",
        "日常训练",
        [
            {"op": "add", "text": "晨跑训练 5 公里。"},
            {"op": "add", "text": "力量训练 30 分钟。"},
        ],
        today=day1,
    )
    memory.apply_memory_operations(
        "training",
        "训练记录",
        "日常训练",
        [{"op": "merge", "old": "训练", "text": "配速 5 分 30 秒。"}],
        today=day2,
    )
    training_text = (memory.memory_dir / "training.md").read_text(encoding="utf-8")
    assert "- [2023-05-21] 晨跑训练 5 公里；配速 5 分 30 秒。" in training_text
    assert "- [2023-05-21] 力量训练 30 分钟。" in training_text

    assert (
        memory.apply_memory_operations(
            "training",
            "训练记录",
            "日常训练",
            [{"op": "merge", "old": "训练", "text": "过期补充。", "until": "2023-05-20"}],
            today=day2,
        )
        is None
    )
    assert "过期补充" not in (memory.memory_dir / "training.md").read_text(encoding="utf-8")
