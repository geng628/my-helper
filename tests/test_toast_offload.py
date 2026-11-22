"""极简 TOAST 工具输出卸载原语（common.truncate_or_offload）单元测试与 MemoryStore 联调测试（TDD）。"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pytest

from personal_assistant.assistant.common import truncate_or_offload
from personal_assistant.assistant.memory import MemoryStore
from plugins.fts5_retriever import FTS5Retriever


# 测试未超字节上限时直接透传原文且零落盘，以及非法参数校验
def test_truncate_or_offload_passthrough_and_validation(tmp_path: Path) -> None:
    offload_dir = tmp_path / "workspace" / "tool_outputs"
    short = "短文本输出，不需要落盘。"
    result = truncate_or_offload(short, max_bytes=200, tag="webpage", offload_dir=offload_dir)
    assert result == short
    assert not offload_dir.exists()

    with pytest.raises(ValueError, match="max_bytes"):
        truncate_or_offload("abc", max_bytes=0, offload_dir=offload_dir)
    with pytest.raises(ValueError, match="mode"):
        truncate_or_offload("abc" * 100, max_bytes=10, mode="invalid", offload_dir=offload_dir)


# 测试 head 模式按整行边界截断落盘，并与 MemoryStore.read_markdown / FTS5 search_markdown 完整联调
def test_truncate_or_offload_head_mode_integrates_with_memory_store(tmp_path: Path) -> None:
    offload_dir = tmp_path / "workspace" / "tool_outputs"
    lines = [f"第 {idx:02d} 行：关于分布式共识与 Raft 日志复制的说明内容。" for idx in range(1, 41)]
    lines[34] = "第 35 行：独一无二的关键结论 QUORUM_CHECKPOINT_9942 在文章后半段。"
    full_text = "\n".join(lines)

    out = truncate_or_offload(
        full_text,
        max_bytes=500,
        tag="webpage",
        mode="head",
        offload_dir=offload_dir,
    )
    assert "输出已截断" in out
    assert "第 01 行" in out
    assert "QUORUM_CHECKPOINT_9942" not in out

    # 解析提示中的逻辑路径与续读行号，验证可无缝传给 read_markdown 与 search_markdown
    path_match = re.search(r"(data/workspace/tool_outputs/[A-Za-z0-9._-]+\.txt)", out)
    next_match = re.search(r"start_line=(\d+)", out)
    assert path_match is not None, f"未在提示中找到落盘路径：{out}"
    assert next_match is not None, f"未在提示中找到续读行号：{out}"
    rel_path = path_match.group(1)
    next_line = int(next_match.group(1))
    assert next_line > 1
    assert f"第 {next_line - 1:02d} 行" in out
    assert f"第 {next_line:02d} 行" not in out

    memory = MemoryStore(tmp_path, retriever=FTS5Retriever())
    rest = memory.read_markdown(rel_path, start_line=next_line, end_line=40)
    assert f"{next_line}: 第 {next_line:02d} 行" in rest
    assert "QUORUM_CHECKPOINT_9942" in rest
    assert "40: 第 40 行" in rest

    search_out = memory.search_markdown("QUORUM_CHECKPOINT_9942", root="data/workspace/tool_outputs")
    assert rel_path in search_out
    assert "35: 第 35 行" in search_out


# 测试 head_tail 模式保留首尾预览（如 pytest 失败汇总），中间标出精确省略行号且可按行回查
def test_truncate_or_offload_head_tail_preserves_head_and_tail_summary(tmp_path: Path) -> None:
    offload_dir = tmp_path / "workspace" / "tool_outputs"
    lines = ["=== test session starts ===", "platform darwin -- Python 3.14"]
    lines.extend(f"middle detail line {i:03d}: trace payload value_{i}" for i in range(1, 61))
    lines.append("FAILED tests/test_demo.py::test_alpha - AssertionError: expected 42")
    lines.append("=== 1 failed, 80 passed in 2.31s ===")
    full_text = "\n".join(lines)

    out = truncate_or_offload(
        full_text,
        max_bytes=520,
        tag="shell",
        mode="head_tail",
        offload_dir=offload_dir,
    )
    assert "输出已截断" in out
    assert "=== test session starts ===" in out
    assert "=== 1 failed, 80 passed in 2.31s ===" in out
    assert "FAILED tests/test_demo.py::test_alpha" in out
    assert "middle detail line 030" not in out

    omit_match = re.search(r"中间省略第 (\d+)-(\d+) 行", out)
    path_match = re.search(r"(data/workspace/tool_outputs/[A-Za-z0-9._-]+\.txt)", out)
    assert omit_match is not None
    assert path_match is not None
    omit_start, omit_end = int(omit_match.group(1)), int(omit_match.group(2))
    assert omit_start <= omit_end

    memory = MemoryStore(tmp_path)
    middle = memory.read_markdown(path_match.group(1), start_line=omit_start, end_line=omit_end)
    assert "middle detail line 030" in middle


# 测试超长单行文本软折行（防单行文本导致分页失效）与落盘单文件最大字节硬上限
def test_truncate_or_offload_soft_wraps_long_lines_and_enforces_max_offload_bytes(tmp_path: Path) -> None:
    offload_dir = tmp_path / "workspace" / "tool_outputs"
    one_huge_line = " ".join(f"token{i:04d}" for i in range(400))
    assert "\n" not in one_huge_line

    out = truncate_or_offload(
        one_huge_line,
        max_bytes=300,
        tag="json",
        mode="head",
        offload_dir=offload_dir,
        soft_wrap_chars=120,
        max_offload_bytes=1024,
    )
    assert "输出已截断" in out
    files = list(offload_dir.glob("*.txt"))
    assert len(files) == 1
    saved_bytes = files[0].read_bytes()
    assert len(saved_bytes) <= 1024
    saved_lines = saved_bytes.decode("utf-8").splitlines()
    assert len(saved_lines) >= 5
    assert all(len(line) <= 120 for line in saved_lines[:-1])
    assert "落盘上限" in saved_lines[-1]


# 测试懒清理机制（清理过期文件与超出数量上限的旧文件）以及 tag 路径穿越清洗
def test_truncate_or_offload_lazy_gc_and_tag_sanitization(tmp_path: Path) -> None:
    offload_dir = tmp_path / "workspace" / "tool_outputs"
    offload_dir.mkdir(parents=True, exist_ok=True)

    # 预埋 1 个已过期的陈旧文件
    expired = offload_dir / "20200101-000000-old-deadbeef.txt"
    expired.write_text("stale content\n", encoding="utf-8")
    old_ts = time.time() - 7200
    os.utime(expired, (old_ts, old_ts))

    # 连续写入 4 个新文件，max_files=2 应仅保留最新的 2 个且过期文件被清除
    base_ts = time.time() - 100
    for idx in range(4):
        payload = "\n".join(f"batch {idx} line {j}: payload text" for j in range(30))
        out = truncate_or_offload(
            payload,
            max_bytes=120,
            tag="../../.env/evil_Tag!",
            mode="head",
            offload_dir=offload_dir,
            max_files=2,
            ttl_seconds=3600,
        )
        assert ".." not in out
        assert ".env" not in out
        match = re.search(r"data/workspace/tool_outputs/([A-Za-z0-9._-]+\.txt)", out)
        assert match is not None
        created_path = offload_dir / match.group(1)
        assert created_path.is_file()
        # 使用严格递增的过去时间戳，稳定区分先后顺序且不触发 TTL
        ts = base_ts + idx * 10
        os.utime(created_path, (ts, ts))

    assert not expired.exists()
    remaining = sorted(offload_dir.glob("*.txt"))
    assert len(remaining) == 2
    assert all(p.parent == offload_dir for p in remaining)
    contents = [p.read_text(encoding="utf-8") for p in remaining]
    assert any("batch 2" in text for text in contents)
    assert any("batch 3" in text for text in contents)


# 测试首行或末行本身超过预览字节预算时（含多字节中文），head 与 head_tail 均安全切片不崩溃
def test_truncate_or_offload_handles_lines_larger_than_budget(tmp_path: Path) -> None:
    offload_dir = tmp_path / "workspace" / "tool_outputs"
    long_cn = "甲" * 60 + "\n" + "乙" * 60

    out_head = truncate_or_offload(
        long_cn,
        max_bytes=40,
        tag="cn",
        mode="head",
        offload_dir=offload_dir,
        soft_wrap_chars=0,
    )
    assert "输出已截断" in out_head
    assert "甲" in out_head
    assert "start_line=1" in out_head

    out_ht = truncate_or_offload(
        long_cn,
        max_bytes=40,
        tag="cn",
        mode="head_tail",
        offload_dir=offload_dir,
        soft_wrap_chars=0,
    )
    assert "输出已截断" in out_ht
    assert "甲" in out_ht
    assert "乙" in out_ht


# 测试默认路径遵循 MY_HELPER_DATA 环境变量，且磁盘不可写时安全回退到 truncate_utf8 不崩溃
def test_truncate_or_offload_uses_my_helper_data_and_falls_back_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_HELPER_DATA", str(tmp_path))
    payload = "\n".join(f"env line {i}: hello world" for i in range(30))
    out = truncate_or_offload(payload, max_bytes=120, tag="web")
    assert "输出已截断" in out
    created = list((tmp_path / "workspace" / "tool_outputs").glob("*.txt"))
    assert len(created) == 1

    # 用普通文件占据 offload_dir 路径制造 OSError，确认优雅回退不抛异常
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("block", encoding="utf-8")
    fallback = truncate_or_offload(payload, max_bytes=120, tag="web", offload_dir=blocker)
    assert "输出已截断" in fallback
    assert "data/workspace/tool_outputs/" not in fallback


# 测试 AssistantApp 使用自定义 data_root（未设 MY_HELPER_DATA）时，工具 TOAST 落盘自动对齐该实例并可由 read_markdown 续读
def test_assistant_app_custom_data_root_binds_toast_offload_without_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from personal_assistant.assistant.app import AssistantApp
    from personal_assistant.assistant.common import REPO_ROOT

    monkeypatch.delenv("MY_HELPER_DATA", raising=False)
    custom_data = tmp_path / "isolated_instance_data"
    app = AssistantApp(
        data_root=custom_data,
        config_path=REPO_ROOT / "config" / "tools.json",
        project_root=REPO_ROOT,
    )
    try:
        reg = app.build_tools()
        obs = reg.execute("run_command", '{"command": "cat personal_assistant/assistant/memory.py"}')
        match = re.search(r"(data/workspace/tool_outputs/[A-Za-z0-9_.-]+\.txt)", obs)
        assert match is not None, f"未触发 TOAST 落盘：{obs[:200]}"
        rel_path = match.group(1)
        filename = Path(rel_path).name

        # 落盘文件必须位于 custom_data 下，绝不污染仓库根目录 REPO_ROOT/data
        assert (custom_data / "workspace" / "tool_outputs" / filename).is_file()
        assert not (REPO_ROOT / "data" / "workspace" / "tool_outputs" / filename).exists()

        # 同实例的 read_markdown 与 search_markdown 必须能直接读取与检索该落盘文件
        read_obs = reg.execute(
            "read_markdown",
            f'{{"path": "{rel_path}", "start_line": 1, "end_line": 15}}',
        )
        assert "纯 Markdown 记忆" in read_obs
        assert "文件不存在" not in read_obs

        search_obs = reg.execute(
            "search_markdown",
            '{"query": "MemoryStore", "root": "data/workspace/tool_outputs"}',
        )
        assert filename in search_obs
    finally:
        app.close()
