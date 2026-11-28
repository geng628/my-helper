"""Markdown 结构化分块器单元测试（TDD）：覆盖标题栈面包屑、代码块/表格完整保护、过期条目隔离与内容哈希。"""
from __future__ import annotations

from personal_assistant.assistant.chunker import DocumentChunk, chunk_markdown


# 测试多级标题栈面包屑维护，且代码块内部的 # 注释绝不被误判为标题
def test_chunk_markdown_tracks_heading_hierarchy_and_ignores_code_comments() -> None:
    md = (
        "# 架构设计\n"
        "\n"
        "总览说明段落。\n"
        "\n"
        "## 检索层\n"
        "\n"
        "### FTS5 索引\n"
        "使用 SQLite FTS5 进行词面召回。\n"
        "\n"
        "```python\n"
        "# 这不是 Markdown 标题\n"
        "def score(x: float) -> float:\n"
        "    # 另一行注释\n"
        "    return x * 2.0\n"
        "```\n"
        "\n"
        "## 安全边界\n"
        "禁止访问敏感路径。\n"
    )
    chunks = chunk_markdown("docs/arch.md", md, max_chars=260)
    assert len(chunks) >= 3
    assert all(isinstance(c, DocumentChunk) for c in chunks)

    # 顶层段落归属 H1
    assert chunks[0].section == "架构设计"
    assert "总览说明段落" in chunks[0].text

    # 代码块归属完整三级面包屑，且整块完整保留未截断
    code_chunk = next(c for c in chunks if "def score" in c.text)
    assert code_chunk.section == "架构设计 > 检索层 > FTS5 索引"
    assert "```python" in code_chunk.text
    assert "# 这不是 Markdown 标题" in code_chunk.text
    assert code_chunk.text.rstrip().endswith("```")
    assert code_chunk.start_line <= 10 and code_chunk.end_line >= 15

    # 新二级标题弹出旧三级标题，面包屑回退到 H1 > H2
    sec_chunk = next(c for c in chunks if "禁止访问敏感路径" in c.text)
    assert sec_chunk.section == "架构设计 > 安全边界"


# 测试围栏代码块（含空行与 ~~~ 围栏）和 Markdown 表格保持结构完整，不被拦腰切碎
def test_chunk_markdown_preserves_fenced_code_and_tables_intact() -> None:
    md = (
        "# 实验记录\n"
        "\n"
        "~~~python\n"
        "def alpha():\n"
        "    x = 1\n"
        "\n"
        "    return x + 1\n"
        "~~~\n"
        "\n"
        "| 模型 | Recall | MRR |\n"
        "| --- | --- | --- |\n"
        "| linear | 0.428 | 0.359 |\n"
        "| rag_bm25 | 0.685 | 0.622 |\n"
    )
    chunks = chunk_markdown("docs/exp.md", md, max_chars=120)

    code_chunks = [c for c in chunks if "def alpha()" in c.text]
    assert len(code_chunks) == 1
    assert "return x + 1" in code_chunks[0].text
    assert code_chunks[0].text.startswith("~~~python")
    assert code_chunks[0].text.endswith("~~~")
    assert (code_chunks[0].start_line, code_chunks[0].end_line) == (3, 8)

    table_chunks = [c for c in chunks if "| 模型 |" in c.text]
    assert len(table_chunks) == 1
    assert "| linear |" in table_chunks[0].text
    assert "| rag_bm25 |" in table_chunks[0].text
    assert (table_chunks[0].start_line, table_chunks[0].end_line) == (10, 13)


# 测试 track_expiry=True 时带 (until: YYYY-MM-DD) 的时效条目严格独立成块，不与常驻邻居混块
def test_chunk_markdown_isolates_expiring_lines_when_tracking_expiry() -> None:
    md = (
        "# 日程\n"
        "- [2026-11-01] 常驻项目说明。\n"
        "- [2026-11-15] 参加冬季压测演练。 (until: 2026-11-20)\n"
        "- [2026-10-01] 旧版架构评审会。 (until: 2026-10-15)\n"
        "- [2026-11-16] 另一条常驻备忘。\n"
    )
    chunks = chunk_markdown("data/memory/schedule.md", md, max_chars=600, track_expiry=True)

    exp_nov = [c for c in chunks if "冬季压测演练" in c.text]
    exp_oct = [c for c in chunks if "旧版架构评审会" in c.text]
    assert len(exp_nov) == 1
    assert exp_nov[0].until_date == "2026-11-20"
    assert (exp_nov[0].start_line, exp_nov[0].end_line) == (3, 3)
    assert "常驻项目说明" not in exp_nov[0].text
    assert "旧版架构评审会" not in exp_nov[0].text

    assert len(exp_oct) == 1
    assert exp_oct[0].until_date == "2026-10-15"
    assert (exp_oct[0].start_line, exp_oct[0].end_line) == (4, 4)

    permanent = [c for c in chunks if c.until_date == ""]
    assert all("冬季压测演练" not in c.text and "旧版架构评审会" not in c.text for c in permanent)


# 测试 content_hash 对正文、标题面包屑与路径变化敏感，且长连续正文按行边界切分不丢行
def test_chunk_markdown_content_hash_and_long_text_line_splitting() -> None:
    c1 = chunk_markdown("docs/a.md", "# A\n相同正文内容。\n")[0]
    c2 = chunk_markdown("docs/a.md", "# A\n相同正文内容。\n")[0]
    c3 = chunk_markdown("docs/a.md", "# B\n相同正文内容。\n")[0]
    c4 = chunk_markdown("docs/b.md", "# A\n相同正文内容。\n")[0]
    assert c1.content_hash == c2.content_hash
    assert len(c1.content_hash) == 64
    assert c1.content_hash != c3.content_hash
    assert c1.content_hash != c4.content_hash

    lines = [f"第 {i:02d} 行：分布式共识与日志复制说明。" for i in range(1, 31)]
    long_chunks = chunk_markdown("docs/long.md", "\n".join(lines), max_chars=180)
    assert len(long_chunks) >= 3
    covered = [ln for c in long_chunks for ln in range(c.start_line, c.end_line + 1)]
    assert covered == list(range(1, 31))
    assert [c.chunk_index for c in long_chunks] == list(range(len(long_chunks)))

    # 仅含标题的文件仍产出 1 个可检索块，纯空文件返回空列表
    heading_only = chunk_markdown("docs/empty_body.md", "# 仅有标题\n")
    assert len(heading_only) == 1 and heading_only[0].section == "仅有标题"
    assert chunk_markdown("docs/blank.md", "   \n\n") == []
