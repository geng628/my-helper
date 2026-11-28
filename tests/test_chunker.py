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


# 回归 B3 与 Nit 1：行内三反引号不误开围栏、未闭合围栏不吞过期条目、同名标题绝不合并、代码块/表格双向不与正文混块、多行过期列表与非法日期对齐生命周期
def test_chunk_markdown_adversarial_edge_cases_and_expiry_isolation() -> None:
    # 1. 行内 ```bash``` 不得误判为未闭合代码块围栏从而吞噬后续标题与过期条目
    inline_fence_md = (
        "# 记录\n\n"
        "```bash``` 是常见的代码块围栏标记。\n\n"
        "## 日程\n"
        "- [2026-10-01] 过期敏感事项 EXPIRED_LEAK (until: 2026-10-15)\n"
    )
    inline_chunks = chunk_markdown("data/memory/inline.md", inline_fence_md, track_expiry=True)
    exp_chunks = [c for c in inline_chunks if "EXPIRED_LEAK" in c.text]
    assert len(exp_chunks) == 1
    assert exp_chunks[0].until_date == "2026-10-15"
    assert exp_chunks[0].section == "记录 > 日程"
    assert all("EXPIRED_LEAK" not in c.text for c in inline_chunks if not c.until_date)

    # 2. track_expiry=True 时真正未闭合的围栏同样不得吞噬后续过期条目
    unclosed_exp_md = (
        "# 记录\n"
        "```python\n"
        "x = 1\n"
        "- [2026-10-01] 未闭合后过期项 UNCLOSED_EXP (until: 2026-10-15)\n"
    )
    unclosed_exp_chunks = chunk_markdown("data/memory/unclosed.md", unclosed_exp_md, track_expiry=True)
    exp_after = [c for c in unclosed_exp_chunks if "UNCLOSED_EXP" in c.text]
    assert len(exp_after) == 1 and exp_after[0].until_date == "2026-10-15"

    # 3. 非过期追踪模式下未闭合代码块裁去尾部空行，保持 start_line..end_line 与 split("\n") 严格对齐
    unclosed_code = "```python\nx = 1\n\n"
    uc = chunk_markdown("docs/uc.md", unclosed_code, track_expiry=False)
    assert len(uc) == 1
    assert (uc[0].start_line, uc[0].end_line) == (1, 2)
    assert len(uc[0].text.split("\n")) == uc[0].end_line - uc[0].start_line + 1

    # 4. 同名二级标题在默认 max_chars=500 下绝不合并进同一分块，也不吞并标题行
    dup_heading_md = "# 日记\n\n## 备忘\n第一段内容。\n\n## 备忘\n第二段内容。\n"
    dup_chunks = chunk_markdown("docs/dup.md", dup_heading_md, max_chars=500)
    assert len(dup_chunks) == 2
    assert "## 备忘" not in dup_chunks[0].text and "## 备忘" not in dup_chunks[1].text
    assert "第一段内容" in dup_chunks[0].text and "第二段内容" in dup_chunks[1].text

    # 5. 代码块与表格在前后双向均保持独立块，不与紧跟其后的正文段落混拼
    code_prose_md = "# 笔记\n\n```python\nx = 1\n```\n\n这是紧随代码块之后的正文段落。\n\n| A | B |\n| - | - |\n| 1 | 2 |\n\n表格后的正文。\n"
    cp_chunks = chunk_markdown("docs/cp.md", code_prose_md, max_chars=500)
    assert len(cp_chunks) == 4
    assert cp_chunks[0].text.startswith("```python") and "紧随代码块之后" not in cp_chunks[0].text
    assert cp_chunks[1].text == "这是紧随代码块之后的正文段落。"
    assert cp_chunks[2].text.startswith("| A | B |") and "表格后的正文" not in cp_chunks[2].text
    assert cp_chunks[3].text == "表格后的正文。"

    # 6. 多行缩进过期列表项整体归入同一过期块；非法日期 (until: 2026-00-00) 按无截止日期处理；带过期标签的标题不泄漏进 section
    expiry_edge_md = (
        "# 常规\r\n"
        "- [2026-10-01] 临时会议 (until: 2026-10-15)\r\n"
        "  会议密码 TEMP_PASS_123\r\n"
        "- [2026-10-01] 非法日期保留 (until: 2026-00-00)\r\n"
        "## 过期标题 SECRET_HEADING (until: 2026-10-15)\r\n"
        "普通正文行。\r\n"
        "| 项 | 说明 |\r\n"
        "| --- | --- |\r\n"
        "| 常驻 | ok |\r\n"
        "| 临时 | ROW_EXP (until: 2026-10-15) |\r\n"
    )
    ee_chunks = chunk_markdown("data/memory/edge.md", expiry_edge_md, track_expiry=True)
    pass_chunks = [c for c in ee_chunks if "TEMP_PASS_123" in c.text]
    assert len(pass_chunks) == 1 and pass_chunks[0].until_date == "2026-10-15"
    invalid_date_chunks = [c for c in ee_chunks if "非法日期保留" in c.text]
    assert len(invalid_date_chunks) == 1 and invalid_date_chunks[0].until_date == ""
    heading_exp = [c for c in ee_chunks if "SECRET_HEADING" in c.text]
    assert len(heading_exp) == 1 and heading_exp[0].until_date == "2026-10-15"
    assert all("SECRET_HEADING" not in c.section for c in ee_chunks)
    row_exp = [c for c in ee_chunks if "ROW_EXP" in c.text]
    assert len(row_exp) == 1 and row_exp[0].until_date == "2026-10-15"

