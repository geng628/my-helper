"""Markdown 父子结构化分块器单元测试（TDD）：覆盖标题面包屑、句子级切分、长表格保表头、整句 Overlap 与过期条目隔离。"""
from __future__ import annotations

import pytest

from personal_assistant.assistant.chunker import (
    ChildChunk,
    ParentChunk,
    chunk_markdown_hierarchy,
    format_context_header,
    split_sentences,
    strip_expiring_lines,
)


# 测试中英文句子边界切分：保护小数 3.14、版本号 v2.0、文件名 rag_index.db、配对闭合引号，且不吞并后句直引号起手
def test_split_sentences_protects_decimals_filenames_and_quotes() -> None:
    zh = "第一句话介绍「系统设计」。第二句提到版本号 v2.0 与 3.14 倍加速！第三句发问：真的吗？"
    zh_sents = split_sentences(zh)
    assert zh_sents == [
        "第一句话介绍「系统设计」。",
        "第二句提到版本号 v2.0 与 3.14 倍加速！",
        "第三句发问：真的吗？",
    ]

    en = 'Check rag_index.db at v2.0 first. Ratio reached 3.14 in Q3! He said "Done." Is it fast? Yes.'
    en_sents = split_sentences(en)
    assert en_sents == [
        "Check rag_index.db at v2.0 first.",
        "Ratio reached 3.14 in Q3!",
        'He said "Done."',
        "Is it fast?",
        "Yes.",
    ]

    # 直引号仅在当前句内未成对闭合时吞并，避免把后句开头直引号误吞入前句
    assert split_sentences('第一句。"第二句。"') == ["第一句。", '"第二句。"']


# 测试多级标题栈面包屑维护，代码块内 # 不误判为标题，且父子块携带统一上下文头
def test_hierarchy_tracks_headings_code_blocks_and_context_headers() -> None:
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
    parents, children = chunk_markdown_hierarchy(
        "docs/arch.md", md, parent_max_chars=400, child_max_chars=200, child_overlap_chars=40
    )
    assert len(parents) >= 4
    assert all(isinstance(p, ParentChunk) for p in parents)
    assert all(isinstance(c, ChildChunk) for c in children)

    assert parents[0].section == "架构设计"
    assert "总览说明段落" in parents[0].text

    code_parent = next(p for p in parents if "def score" in p.text)
    assert code_parent.kind == "code"
    assert code_parent.section == "架构设计 > 检索层 > FTS5 索引"
    assert code_parent.text.startswith("```python") and code_parent.text.endswith("```")
    assert (code_parent.start_line, code_parent.end_line) == (10, 15)

    # 代码块 1 父对 1 子，不做拆碎与重叠
    code_children = [c for c in children if c.parent_index == code_parent.parent_index]
    assert len(code_children) == 1
    assert format_context_header("docs/arch.md", "架构设计 > 检索层 > FTS5 索引") in code_children[0].contextual_text

    sec_parent = next(p for p in parents if "禁止访问敏感路径" in p.text)
    assert sec_parent.section == "架构设计 > 安全边界"


# 测试超大表格拆分时续片保留表头前缀（table_header），且普通表格与代码块保持完整独立
def test_hierarchy_preserves_tables_and_repeats_header_on_oversized_table() -> None:
    md_small = (
        "# 实验\n\n"
        "~~~python\n"
        "def alpha():\n"
        "    x = 1\n\n"
        "    return x + 1\n"
        "~~~\n\n"
        "| 模型 | Recall | MRR |\n"
        "| --- | --- | --- |\n"
        "| linear | 0.428 | 0.359 |\n"
        "| rag | 0.685 | 0.651 |\n"
    )
    parents, _ = chunk_markdown_hierarchy("docs/exp.md", md_small, parent_max_chars=200, child_max_chars=120)
    tbl = next(p for p in parents if "| 模型 |" in p.text)
    assert tbl.kind == "table" and tbl.table_header == ""
    assert (tbl.start_line, tbl.end_line) == (10, 13)

    # 构造超出 2 * parent_max_chars 的超长表格，验证拆分续片自动携带 table_header 且不打乱行号
    rows = [f"| row_{i:02d} | 性能指标详情描述与对照数据 {i:02d} |" for i in range(1, 15)]
    big_table = "# 大表\n\n| 编号 | 指标说明 |\n| --- | --- |\n" + "\n".join(rows) + "\n"
    big_parents, big_children = chunk_markdown_hierarchy(
        "docs/big_table.md", big_table, parent_max_chars=120, child_max_chars=120
    )
    assert len(big_parents) >= 2
    assert big_parents[0].table_header == ""
    assert "| 编号 | 指标说明 |" in big_parents[0].text
    for cont in big_parents[1:]:
        assert cont.kind == "table"
        assert cont.table_header == "| 编号 | 指标说明 |\n| --- | --- |"
        assert len(cont.text.split("\n")) == cont.end_line - cont.start_line + 1
        # 续片对应的子块索引文本必须包含表头列名，保证按列名也能搜中后半张表
        cont_child = next(c for c in big_children if c.parent_index == cont.parent_index)
        assert "| 编号 | 指标说明 |" in cont_child.text


# 测试父块内子块按整句边界切分与可配 Overlap（0 vs 60），绝不截断半句话且保证严格前进
def test_hierarchy_sentence_splitting_and_configurable_overlap() -> None:
    single_long_para = (
        "# 长段落\n\n"
        "第一句说明系统采用了三层洋葱微内核架构设计。"
        "第二句解释检索层如何通过父子索引提升召回精度。"
        "第三句介绍持久化派生数据库与权威源的物理隔离。"
        "第四句强调通过重叠切块避免跨句上下文在切缝处断裂。"
        "第五句总结端到端消融评测在多跳问题上的收益。\n"
    )
    # 1. overlap=0：无重复句子，每句完整不腰斩
    p0, c0 = chunk_markdown_hierarchy(
        "docs/long.md",
        single_long_para,
        parent_max_chars=600,
        child_max_chars=55,
        child_overlap_chars=0,
    )
    assert len(p0) == 1
    assert len(c0) >= 3
    assert all(c.parent_index == 0 for c in c0)
    assert all(c.text.endswith("。") for c in c0)
    assert "第二句" not in c0[0].text or "第二句" not in c0[1].text

    # 2. overlap=30：相邻子块共享上一子块尾部整句，且同样绝不截断半句
    p_ov, c_ov = chunk_markdown_hierarchy(
        "docs/long.md",
        single_long_para,
        parent_max_chars=600,
        child_max_chars=55,
        child_overlap_chars=30,
    )
    assert len(p_ov) == 1
    assert len(c_ov) >= 3
    assert all(c.text.endswith("。") for c in c_ov)
    # 至少有一对相邻子块发生整句重叠
    overlapped = any(
        any(sent in c_ov[i + 1].text for sent in split_sentences(c_ov[i].text))
        for i in range(len(c_ov) - 1)
    )
    assert overlapped is True

    # 3. 参数校验：非法 overlap（>= child_max_chars、负数或布尔值）显式报错
    with pytest.raises(ValueError):
        chunk_markdown_hierarchy("docs/a.md", "# A\n内容", child_max_chars=100, child_overlap_chars=100)
    with pytest.raises(ValueError):
        chunk_markdown_hierarchy("docs/a.md", "# A\n内容", child_max_chars=100, child_overlap_chars=-1)
    with pytest.raises(ValueError):
        chunk_markdown_hierarchy("docs/a.md", "# A\n内容", child_overlap_chars=True)


# 测试全部对抗边界：行内 ```bash```、未闭合围栏、同名标题不合并、代码块/表格双向隔离、多行过期列表项与非法日期
def test_hierarchy_adversarial_edge_cases_and_expiry_isolation() -> None:
    # 1. 行内 ```bash``` 不误开围栏吞噬后续过期条目
    inline_md = (
        "# 记录\n\n"
        "```bash``` 是常见的代码块围栏标记。\n\n"
        "## 日程\n"
        "- [2026-10-01] 过期敏感事项 EXPIRED_LEAK (until: 2026-10-15)\n"
    )
    parents, children = chunk_markdown_hierarchy("data/memory/inline.md", inline_md, track_expiry=True)
    exp_p = [p for p in parents if "EXPIRED_LEAK" in p.text]
    assert len(exp_p) == 1 and exp_p[0].until_date == "2026-10-15" and exp_p[0].section == "记录 > 日程"
    assert all("EXPIRED_LEAK" not in c.text for c in children if not c.until_date)

    # 2. 同名二级标题绝不合并跨节，且代码块/表格前后双向独立
    dup_md = (
        "# 日记\n\n"
        "## 备忘\n第一段内容。\n\n"
        "## 备忘\n第二段内容。\n\n"
        "```python\nx = 1\n```\n\n"
        "代码块后正文。\n\n"
        "| A | B |\n| - | - |\n| 1 | 2 |\n\n"
        "表格后正文。\n"
    )
    dp, _ = chunk_markdown_hierarchy("docs/dup.md", dup_md, parent_max_chars=1000)
    assert len(dp) == 6
    assert "## 备忘" not in dp[0].text and "## 备忘" not in dp[1].text
    assert dp[2].kind == "code" and "代码块后正文" not in dp[2].text
    assert dp[4].kind == "table" and "表格后正文" not in dp[4].text

    # 3. 多行缩进过期列表项、非法日期、过期标题、表格过期行、未闭合围栏及纯空/仅标题文件
    edge_md = (
        "# 常规\r\n"
        "- [2026-10-01] 临时会议 (until: 2026-10-15)\r\n"
        "  会议密码 TEMP_PASS_123\r\n"
        "- [2026-10-01] 非法日期保留 (until: 2026-00-00)\r\n"
        "## 过期标题 SECRET_HEADING (until: 2026-10-15)\r\n"
        "普通正文行。\r\n\r\n"
        "| 事项 | 状态 |\r\n"
        "| --- | --- |\r\n"
        "| 常规任务 | 进行中 |\r\n"
        "| 临时压测 ROW_EXP (until: 2026-10-15) | 待删 |\r\n"
    )
    ep, ec = chunk_markdown_hierarchy("data/memory/edge.md", edge_md, track_expiry=True)
    pass_p = [p for p in ep if "TEMP_PASS_123" in p.text]
    assert len(pass_p) == 1 and pass_p[0].until_date == "2026-10-15"
    inv_p = [p for p in ep if "非法日期保留" in p.text]
    assert len(inv_p) == 1 and inv_p[0].until_date == ""
    assert all("SECRET_HEADING" not in p.section for p in ep)
    row_exp = [p for p in ep if "ROW_EXP" in p.text]
    assert len(row_exp) == 1 and row_exp[0].until_date == "2026-10-15"
    assert all("TEMP_PASS_123" not in c.text and "ROW_EXP" not in c.text for c in ec if not c.until_date)

    # strip_expiring_lines 彻底剥离多行过期列表项、过期标题与表格过期行，保留常驻内容
    stripped = strip_expiring_lines(edge_md)
    assert "TEMP_PASS_123" not in stripped
    assert "SECRET_HEADING" not in stripped
    assert "ROW_EXP" not in stripped
    assert "非法日期保留" in stripped and "普通正文行。" in stripped and "常规任务" in stripped

    # 4. 未闭合围栏：track_expiry=True 不吞噬后续过期行；track_expiry=False 裁尾空行精确对齐行号
    unclosed_md = "# 未闭合\n```python\nx = 1\n- 过期任务 UNCLOSED_EXP (until: 2026-10-15)\n\n\n"
    up_track, uc_track = chunk_markdown_hierarchy("data/memory/unclosed.md", unclosed_md, track_expiry=True)
    u_exp = [p for p in up_track if "UNCLOSED_EXP" in p.text]
    assert len(u_exp) == 1 and u_exp[0].until_date == "2026-10-15"
    assert all("UNCLOSED_EXP" not in c.text for c in uc_track if not c.until_date)

    up_plain, _ = chunk_markdown_hierarchy("docs/unclosed.md", unclosed_md, track_expiry=False)
    assert len(up_plain) == 1 and up_plain[0].kind == "code"
    assert len(up_plain[0].text.split("\n")) == up_plain[0].end_line - up_plain[0].start_line + 1
    assert not up_plain[0].text.endswith("\n")

    hp, hc = chunk_markdown_hierarchy("docs/h.md", "# 仅有标题\n")
    assert len(hp) == 1 and len(hc) == 1 and hp[0].section == "仅有标题"
    assert chunk_markdown_hierarchy("docs/blank.md", "  \n\n") == ([], [])


# 测试单句超过 child_max_chars 时的软切分与硬切分：零丢字、子块长度不越界、且不向连续 ASCII 标识符注入假空格
def test_hierarchy_soft_and_hard_split_for_oversized_single_sentence() -> None:
    comma_long = (
        "# 长句\n\n"
        "这是一段没有句号的超长说明，包含中文词组 alpha_token，随后继续补充架构细节，再追加一段状态说明，最后交代验收口径\n"
    )
    parents, children = chunk_markdown_hierarchy(
        "docs/comma.md",
        comma_long,
        parent_max_chars=400,
        child_max_chars=24,
        child_overlap_chars=0,
    )
    assert len(parents) == 1 and len(children) >= 3
    assert all(len(c.text) <= 24 for c in children)
    assert "alpha_token" in "".join(c.text for c in children)
    # 中英文之间的真实空格得以保留
    assert any("中文词组 alpha_token" in c.text or c.text.endswith("中文词组") for c in children)

    ident = "SUPER_LONG_ASCII_IDENTIFIER_" + "X" * 60
    _, id_children = chunk_markdown_hierarchy(
        "docs/ident.md",
        f"# 标识符\n\n{ident}\n",
        parent_max_chars=200,
        child_max_chars=30,
        child_overlap_chars=0,
    )
    assert len(id_children) >= 3
    assert all(len(c.text) <= 30 for c in id_children)
    # 硬切连续标识符不得注入虚假空格，拼回与原文逐字符相等
    assert "".join(c.text for c in id_children) == ident
