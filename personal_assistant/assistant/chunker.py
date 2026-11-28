"""Markdown 父子结构化分块器：多级标题面包屑、句子边界切分、长表保表头、可配整句 Overlap 与过期条目隔离。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from personal_assistant.assistant.common import parse_iso_date
from personal_assistant.assistant.memory_lifecycle import UNTIL_PATTERN

DEFAULT_PARENT_MAX_CHARS = 1000
DEFAULT_CHILD_MAX_CHARS = 320
DEFAULT_CHILD_OVERLAP_CHARS = 64

_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})(?:[ \t]+|$)(.*)$")
_FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_TABLE_SEP_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$"
)
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
_CLOSING_DIRECTIONAL_WRAPPERS = frozenset("”’」』）)]】》")
_STRAIGHT_QUOTES = frozenset("\"'")
_CJK_SENTENCE_ENDERS = frozenset("。！？")
_ASCII_SENTENCE_ENDERS = frozenset(".!?")
_EN_ABBREVS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e", "fig", "no", "st", "vol"}
)
_SOFT_CLAUSE_BREAKS = frozenset("；;，, ")


# 单条父级上下文分块：供命中子块后回送完整上下文，text 与 start_line..end_line 严格一一对齐
@dataclass(frozen=True)
class ParentChunk:
    path: str
    parent_index: int
    start_line: int
    end_line: int
    section: str
    kind: str
    text: str
    contextual_text: str
    content_hash: str
    until_date: str = ""
    table_header: str = ""


# 单条子级检索分块：归属唯一父块，按句子/行边界切分并可配相邻整句 Overlap，用于高精度 FTS5/向量匹配
@dataclass(frozen=True)
class ChildChunk:
    path: str
    parent_index: int
    child_index: int
    start_line: int
    end_line: int
    section: str
    text: str
    contextual_text: str
    content_hash: str
    until_date: str = ""


# 内部原子结构块（围栏代码块、Markdown 表格、列表项、段落或独立过期条目，携带单调递增 section_id 与可选表头）
@dataclass(frozen=True)
class _AtomicBlock:
    start_line: int
    end_line: int
    section_id: int
    section: str
    kind: str
    until_date: str = ""
    standalone: bool = False
    table_header: str = ""


# 内部子块切分原子单元（对应单句或单行列表项，记录原行号与相对前驱单元的精确分隔符）
@dataclass(frozen=True)
class _ChildUnit:
    line_no: int
    text: str
    sep_from_prev: str


# 统一渲染分块头部元数据（来源文件路径 + 章节面包屑 + 可选离线 LLM 背景摘要）
def format_context_header(path: str, section: str, context_summary: str = "") -> str:
    parts = [f"来源: {path}"]
    clean_section = section.strip()
    if clean_section:
        parts.append(f"章节: {clean_section}")
    clean_summary = " ".join(context_summary.split())
    if clean_summary:
        parts.append(f"背景: {clean_summary}")
    return "[" + " | ".join(parts) + "]"


# 判断字符是否为 CJK 汉字或全角中文标点（用于识别中英混排中无空格的句末边界）
def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0x3000 <= code <= 0x303F)


# 提取当前句子缓冲区中位于句点前的最后一个英文词或缩写片段（用于保护 Dr. / e.g. / 1. 等）
def _trailing_ascii_token(segment: str) -> str:
    idx = len(segment) - 1
    while idx >= 0 and (segment[idx].isalnum() or segment[idx] == "."):
        idx -= 1
    return segment[idx + 1 :]


# 将单行文本按中英文句界切分为 (句文本, 与前一句之间的原始空白分隔符) 列表，保护小数、版本号、文件名、缩写与闭合引号
def _split_sentences_with_seps(text: str) -> list[tuple[str, str]]:
    n = len(text)
    if n == 0 or not text.strip():
        return []

    results: list[tuple[str, str]] = []
    pos = 0
    while pos < n and text[pos].isspace():
        pos += 1
    seg_start = pos
    pending_sep = ""

    while pos < n:
        ch = text[pos]
        is_cjk_end = ch in _CJK_SENTENCE_ENDERS
        is_ascii_end = ch in _ASCII_SENTENCE_ENDERS
        if not is_cjk_end and not is_ascii_end:
            pos += 1
            continue

        # 吞并连续的句末标点（如 "？！"、"..."）
        punct_end = pos + 1
        while punct_end < n and (text[punct_end] in _CJK_SENTENCE_ENDERS or text[punct_end] in _ASCII_SENTENCE_ENDERS):
            punct_end += 1

        # 吞并紧跟在句末标点后的闭合引号与括号；直引号仅在当前句内未成对闭合（奇数次出现）时才吞并
        wrap_end = punct_end
        while wrap_end < n:
            wch = text[wrap_end]
            if wch in _CLOSING_DIRECTIONAL_WRAPPERS:
                wrap_end += 1
            elif wch in _STRAIGHT_QUOTES and text[seg_start:wrap_end].count(wch) % 2 == 1:
                wrap_end += 1
            else:
                break

        should_split = False
        if is_cjk_end:
            should_split = True
        elif ch in {"!", "?"}:
            # ASCII !/? 仅在末尾、后接空白或后接汉字时切分，避免误切 URL 查询参数（如 ?a=1）
            should_split = wrap_end >= n or text[wrap_end].isspace() or _is_cjk_char(text[wrap_end])
        else:
            # ASCII 句点 .：仅在末尾、后接空白或后接汉字时视为潜在句末（天然保护 3.14 / v2.0 / rag_index.db）
            if wrap_end >= n or text[wrap_end].isspace() or _is_cjk_char(text[wrap_end]):
                if punct_end - pos > 1:
                    should_split = True
                else:
                    prefix = text[seg_start:pos]
                    prefix_stripped = prefix.strip()
                    last_tok = _trailing_ascii_token(prefix).lower()
                    # 保护行首有序列表前缀（如 "1. "）及常见英文缩写（如 "Dr. " / "e.g. "）
                    is_list_num = prefix_stripped.isdigit() and len(prefix_stripped) <= 3 and wrap_end < n
                    is_abbrev = last_tok in _EN_ABBREVS and wrap_end < n
                    should_split = not is_list_num and not is_abbrev

        if should_split:
            sent = text[seg_start:wrap_end].strip()
            if sent:
                results.append((sent, pending_sep if results else ""))
            ws_start = wrap_end
            ws_end = wrap_end
            while ws_end < n and text[ws_end].isspace():
                ws_end += 1
            pending_sep = text[ws_start:ws_end]
            pos = ws_end
            seg_start = ws_end
        else:
            pos = punct_end

    if seg_start < n:
        tail = text[seg_start:].strip()
        if tail:
            results.append((tail, pending_sep if results else ""))
    return results


# 将中英文文本按自然句子边界切分为有序句子列表，保护小数、版本号、文件名与闭合引号不被误切
def split_sentences(text: str) -> list[str]:
    return [sent for sent, _ in _split_sentences_with_seps(text)]


# 当单个超长无句号从句超过 child_max_chars 时，优先在分号/逗号/空格处软切分，按真实空白还原片段分隔符
def _soft_split_long_sentence(sentence: str, max_chars: int) -> list[tuple[str, str]]:
    if len(sentence) <= max_chars:
        return [(sentence, "")]
    pieces: list[tuple[str, str]] = []
    start = 0
    total = len(sentence)
    pending_sep = ""
    while start < total:
        if total - start <= max_chars:
            part = sentence[start:].strip()
            if part:
                pieces.append((part, pending_sep if pieces else ""))
            break
        limit_end = start + max_chars
        cut = -1
        for probe in range(limit_end, start + max(1, max_chars // 3), -1):
            if sentence[probe - 1] in _SOFT_CLAUSE_BREAKS:
                cut = probe
                break
        if cut == -1:
            cut = limit_end
        raw_part = sentence[start:cut]
        part = raw_part.strip()
        next_start = cut
        while next_start < total and sentence[next_start].isspace():
            next_start += 1
        # 仅在原文切缝处确实存在空白时保留单空格，既不向硬切长标识符注空格，也不丢失中英间的真实空格
        had_ws = (raw_part[-1:].isspace()) or (next_start > cut)
        if part:
            pieces.append((part, pending_sep if pieces else ""))
        pending_sep = " " if had_ws else ""
        start = next_start
    return pieces


# 解析单行 ATX 标题，返回 (标题级别 1..6, 清洗后的标题文本)；非标题行返回 None
def _parse_heading(line: str) -> tuple[int, str] | None:
    match = _HEADING_RE.match(line)
    if match is None:
        return None
    level = len(match.group(1))
    title = match.group(2).strip()
    title = re.sub(r"[ \t]+#+$", "", title).strip()
    if not title:
        return None
    return level, title


# 提取并校验单行文本中的有效 (until: YYYY-MM-DD) 截止日期，非法日期按无截止日期处理
def _extract_until_date(line: str) -> str:
    match = UNTIL_PATTERN.search(line)
    if match is None:
        return ""
    parsed = parse_iso_date(match.group(1))
    return parsed.isoformat() if parsed is not None else ""


# 检测当前行是否为合法 CommonMark 围栏代码块起手（反引号围栏 info 串不得包含反引号），并返回闭合行下标
def _match_fenced_code(
    lines: list[str],
    start_idx: int,
    *,
    require_closed: bool,
) -> int | None:
    line = lines[start_idx]
    fence_match = _FENCE_OPEN_RE.match(line)
    if fence_match is None:
        return None
    fence_token = fence_match.group(1)
    fence_char = fence_token[0]
    fence_len = len(fence_token)
    # CommonMark §4.5：反引号围栏的 info string 不能包含反引号
    if fence_char == "`" and "`" in line[fence_match.end() :]:
        return None

    close_re = re.compile(rf"^[ \t]{{0,3}}{re.escape(fence_char)}{{{fence_len},}}[ \t]*$")
    total = len(lines)
    probe = start_idx + 1
    while probe < total:
        if close_re.match(lines[probe]) is not None:
            return probe + 1
        probe += 1
    if require_closed:
        return None
    end_idx = total
    while end_idx > start_idx + 1 and not lines[end_idx - 1].strip():
        end_idx -= 1
    return end_idx


# 判断当前行与下一行是否构成 GFM Markdown 表格起手（表头含 | 且次行为分隔行）
def _is_table_start(lines: list[str], idx: int) -> bool:
    if idx + 1 >= len(lines):
        return False
    header = lines[idx]
    sep = lines[idx + 1]
    if "|" not in header or "|" not in sep:
        return False
    return _TABLE_SEP_RE.match(sep) is not None


# 计算单行文本的缩进列宽（用于判断列表项的续行归属）
def _leading_indent(line: str) -> int:
    width = 0
    for ch in line:
        if ch == " ":
            width += 1
        elif ch == "\t":
            width += 4
        else:
            break
    return width


# 将整篇文本解析为带单调 section_id、标题面包屑与行号范围的有序原子结构块列表
def _parse_atomic_blocks(
    lines: list[str],
    *,
    track_expiry: bool,
) -> tuple[list[_AtomicBlock], list[tuple[int, str]]]:
    blocks: list[_AtomicBlock] = []
    heading_stack: list[tuple[int, str]] = []
    headings_seen: list[tuple[int, str]] = []
    section_id = 0
    total = len(lines)
    idx = 0

    while idx < total:
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue

        section = " > ".join(t for _, t in heading_stack)

        # 1. 追踪过期模式优先检查合法 (until: YYYY-MM-DD)，防止带过期标签的标题或列表项泄漏进常驻索引
        if track_expiry:
            until_date = _extract_until_date(line)
            if until_date:
                start_idx = idx
                is_list = _LIST_ITEM_RE.match(line) is not None
                base_indent = _leading_indent(line)
                idx += 1
                if is_list:
                    while idx < total:
                        nxt = lines[idx]
                        if not nxt.strip():
                            break
                        if _extract_until_date(nxt):
                            break
                        if _parse_heading(nxt) is not None:
                            break
                        if _match_fenced_code(lines, idx, require_closed=True) is not None:
                            break
                        if _leading_indent(nxt) <= base_indent:
                            break
                        idx += 1
                blocks.append(
                    _AtomicBlock(
                        start_line=start_idx + 1,
                        end_line=idx,
                        section_id=section_id,
                        section=section,
                        kind="expiring",
                        until_date=until_date,
                        standalone=True,
                    )
                )
                continue

        # 2. ATX 标题：更新多级标题栈面包屑并递增单调 section_id（确保同名标题绝不合并跨节）
        heading = _parse_heading(line)
        if heading is not None:
            level, title = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            section_id += 1
            section_now = " > ".join(t for _, t in heading_stack)
            headings_seen.append((idx + 1, section_now))
            idx += 1
            continue

        # 3. 围栏代码块（``` 或 ~~~）：保持整块原子性；track_expiry=True 时未闭合围栏不吞噬后续过期条目
        fence_end = _match_fenced_code(lines, idx, require_closed=track_expiry)
        if fence_end is not None:
            blocks.append(
                _AtomicBlock(
                    start_line=idx + 1,
                    end_line=fence_end,
                    section_id=section_id,
                    section=section,
                    kind="code",
                )
            )
            idx = fence_end
            continue

        # 4. GFM Markdown 表格：从表头 + 分隔行起连续吸收数据行
        if _is_table_start(lines, idx):
            start_idx = idx
            idx += 2
            while idx < total:
                nxt = lines[idx]
                if not nxt.strip() or "|" not in nxt:
                    break
                if track_expiry and _extract_until_date(nxt):
                    break
                if _parse_heading(nxt) is not None:
                    break
                if _match_fenced_code(lines, idx, require_closed=track_expiry) is not None:
                    break
                idx += 1
            blocks.append(
                _AtomicBlock(
                    start_line=start_idx + 1,
                    end_line=idx,
                    section_id=section_id,
                    section=section,
                    kind="table",
                )
            )
            continue

        # 5. Markdown 列表项：吸收当前条目及其缩进续行作为一个原子项
        if _LIST_ITEM_RE.match(line) is not None:
            start_idx = idx
            base_indent = _leading_indent(line)
            idx += 1
            while idx < total:
                nxt = lines[idx]
                if not nxt.strip():
                    break
                if track_expiry and _extract_until_date(nxt):
                    break
                if _parse_heading(nxt) is not None:
                    break
                if _match_fenced_code(lines, idx, require_closed=track_expiry) is not None:
                    break
                if _leading_indent(nxt) <= base_indent:
                    break
                idx += 1
            blocks.append(
                _AtomicBlock(
                    start_line=start_idx + 1,
                    end_line=idx,
                    section_id=section_id,
                    section=section,
                    kind="list",
                )
            )
            continue

        # 6. 普通正文段落：连续非空、非特殊结构行归为一个段落块
        start_idx = idx
        idx += 1
        while idx < total:
            nxt = lines[idx]
            if not nxt.strip():
                break
            if track_expiry and _extract_until_date(nxt):
                break
            if (
                _parse_heading(nxt) is not None
                or _match_fenced_code(lines, idx, require_closed=track_expiry) is not None
                or _is_table_start(lines, idx)
                or _LIST_ITEM_RE.match(nxt) is not None
            ):
                break
            idx += 1
        blocks.append(
            _AtomicBlock(
                start_line=start_idx + 1,
                end_line=idx,
                section_id=section_id,
                section=section,
                kind="prose",
            )
        )

    return blocks, headings_seen


# 将超出字数上限的单个超长表格按行拆分，续片自动记录 table_header 保留表头语义且不打乱原行号
def _split_oversized_table(
    lines: list[str],
    block: _AtomicBlock,
    *,
    max_chars: int,
) -> list[_AtomicBlock]:
    data_start = block.start_line + 2
    if data_start > block.end_line:
        return [block]
    header_str = f"{lines[block.start_line - 1]}\n{lines[block.start_line]}"
    header_len = len(header_str)

    parts: list[_AtomicBlock] = []
    # 首片包含真实表头两行（table_header 为空），尽量装入数据行
    cur_start = block.start_line
    cur_chars = header_len
    first_slice = True
    has_data_row = False

    for line_no in range(data_start, block.end_line + 1):
        row_len = len(lines[line_no - 1])
        effective_base = cur_chars if first_slice else (header_len + 1 + cur_chars if cur_chars > 0 else header_len)
        added = effective_base + 1 + row_len
        if has_data_row and added > max_chars:
            parts.append(
                _AtomicBlock(
                    start_line=cur_start,
                    end_line=line_no - 1,
                    section_id=block.section_id,
                    section=block.section,
                    kind="table",
                    until_date=block.until_date,
                    standalone=block.standalone,
                    table_header="" if first_slice else header_str,
                )
            )
            first_slice = False
            cur_start = line_no
            cur_chars = row_len
            has_data_row = True
        else:
            cur_chars = cur_chars + 1 + row_len
            has_data_row = True

    parts.append(
        _AtomicBlock(
            start_line=cur_start,
            end_line=block.end_line,
            section_id=block.section_id,
            section=block.section,
            kind="table",
            until_date=block.until_date,
            standalone=block.standalone,
            table_header="" if first_slice else header_str,
        )
    )
    return parts


# 将超出字数上限的单个长块按整行边界拆分（表格超 2× 上限保表头拆分；代码块 3× 上限内保持完整）
def _split_oversized_block(
    lines: list[str],
    block: _AtomicBlock,
    *,
    max_chars: int,
) -> list[_AtomicBlock]:
    block_text = "\n".join(lines[block.start_line - 1 : block.end_line])
    if block.standalone or len(block_text) <= max_chars:
        return [block]
    if block.kind == "code" and len(block_text) <= max_chars * 3:
        return [block]
    if block.kind == "table":
        if len(block_text) <= max_chars * 2:
            return [block]
        return _split_oversized_table(lines, block, max_chars=max_chars)

    parts: list[_AtomicBlock] = []
    cur_start = block.start_line
    cur_chars = 0
    for line_no in range(block.start_line, block.end_line + 1):
        line_len = len(lines[line_no - 1])
        added = line_len if cur_chars == 0 else (cur_chars + 1 + line_len)
        if cur_chars > 0 and added > max_chars:
            parts.append(
                _AtomicBlock(
                    start_line=cur_start,
                    end_line=line_no - 1,
                    section_id=block.section_id,
                    section=block.section,
                    kind=block.kind,
                    until_date=block.until_date,
                    standalone=block.standalone,
                )
            )
            cur_start = line_no
            cur_chars = line_len
        else:
            cur_chars = added
    parts.append(
        _AtomicBlock(
            start_line=cur_start,
            end_line=block.end_line,
            section_id=block.section_id,
            section=block.section,
            kind=block.kind,
            until_date=block.until_date,
            standalone=block.standalone,
        )
    )
    return parts


# 构造单个不可变 ParentChunk，统一拼接来源+章节上下文头（续片表格附加表头）并计算 SHA-256 哈希
def _build_parent_chunk(
    path: str,
    parent_index: int,
    start_line: int,
    end_line: int,
    section: str,
    kind: str,
    lines: list[str],
    *,
    until_date: str = "",
    table_header: str = "",
) -> ParentChunk:
    body = "\n".join(lines[start_line - 1 : end_line])
    header = format_context_header(path, section)
    contextual = f"{header}\n{table_header}\n{body}" if table_header else f"{header}\n{body}"
    digest = hashlib.sha256(contextual.encode("utf-8")).hexdigest()
    return ParentChunk(
        path=path,
        parent_index=parent_index,
        start_line=start_line,
        end_line=end_line,
        section=section,
        kind=kind,
        text=body,
        contextual_text=contextual,
        content_hash=digest,
        until_date=until_date,
        table_header=table_header,
    )


# 将单个 ParentChunk 的正文提取为句子/行级原子切分单元（短列表行保持整行不拆散说话人前缀，普通正文按整句切分）
def _extract_child_units(parent: ParentChunk, *, child_max_chars: int) -> list[_ChildUnit]:
    units: list[_ChildUnit] = []
    parent_lines = parent.text.split("\n")
    prev_line_no: int | None = None

    for offset, raw_line in enumerate(parent_lines):
        if not raw_line.strip():
            continue
        line_no = parent.start_line + offset
        line_sep = "" if prev_line_no is None else "\n" * min(2, max(1, line_no - prev_line_no))

        # 列表条目在单行长度未超 child_max_chars 时保持整行完整，避免割裂 "- [D1:1] Speaker: ..." 前缀
        if _LIST_ITEM_RE.match(raw_line) is not None and len(raw_line) <= child_max_chars:
            units.append(_ChildUnit(line_no=line_no, text=raw_line, sep_from_prev=line_sep))
            prev_line_no = line_no
            continue

        sent_pairs = _split_sentences_with_seps(raw_line)
        if not sent_pairs:
            units.append(_ChildUnit(line_no=line_no, text=raw_line.strip(), sep_from_prev=line_sep))
            prev_line_no = line_no
            continue

        first_on_line = True
        for sent, intra_sep in sent_pairs:
            sub_pieces = _soft_split_long_sentence(sent, child_max_chars)
            for piece_idx, (piece_text, sub_sep) in enumerate(sub_pieces):
                if first_on_line:
                    sep = line_sep
                    first_on_line = False
                elif piece_idx == 0:
                    sep = intra_sep
                else:
                    sep = sub_sep
                units.append(_ChildUnit(line_no=line_no, text=piece_text, sep_from_prev=sep))
        prev_line_no = line_no

    return units


# 基于单个 ParentChunk 生成按整句/整行边界切分且带可配 Overlap 的 ChildChunk 列表
def _build_children_for_parent(
    parent: ParentChunk,
    *,
    child_max_chars: int,
    child_overlap_chars: int,
) -> list[ChildChunk]:
    header = format_context_header(parent.path, parent.section)

    # 代码块、表格、过期条目、纯标题块或短父块保持 1 父对 1 子，不做拆碎与冗余重叠
    if (
        parent.kind in {"code", "table", "expiring", "heading"}
        or bool(parent.until_date)
        or len(parent.text) <= child_max_chars
    ):
        child_text = f"{parent.table_header}\n{parent.text}" if parent.table_header else parent.text
        contextual = f"{header}\n{child_text}"
        digest = hashlib.sha256(contextual.encode("utf-8")).hexdigest()
        return [
            ChildChunk(
                path=parent.path,
                parent_index=parent.parent_index,
                child_index=0,
                start_line=parent.start_line,
                end_line=parent.end_line,
                section=parent.section,
                text=child_text,
                contextual_text=contextual,
                content_hash=digest,
                until_date=parent.until_date,
            )
        ]

    units = _extract_child_units(parent, child_max_chars=child_max_chars)
    if not units:
        contextual = f"{header}\n{parent.text}"
        digest = hashlib.sha256(contextual.encode("utf-8")).hexdigest()
        return [
            ChildChunk(
                path=parent.path,
                parent_index=parent.parent_index,
                child_index=0,
                start_line=parent.start_line,
                end_line=parent.end_line,
                section=parent.section,
                text=parent.text,
                contextual_text=contextual,
                content_hash=digest,
                until_date=parent.until_date,
            )
        ]

    children: list[ChildChunk] = []
    total_units = len(units)
    pos = 0

    while pos < total_units:
        end = pos + 1
        cur_len = len(units[pos].text)
        while end < total_units:
            added = cur_len + len(units[end].sep_from_prev) + len(units[end].text)
            if added > child_max_chars:
                break
            cur_len = added
            end += 1

        span = units[pos:end]
        parts = [span[0].text]
        for u in span[1:]:
            parts.append(u.sep_from_prev)
            parts.append(u.text)
        child_text = "".join(parts)
        contextual = f"{header}\n{child_text}"
        digest = hashlib.sha256(contextual.encode("utf-8")).hexdigest()
        children.append(
            ChildChunk(
                path=parent.path,
                parent_index=parent.parent_index,
                child_index=len(children),
                start_line=span[0].line_no,
                end_line=span[-1].line_no,
                section=parent.section,
                text=child_text,
                contextual_text=contextual,
                content_hash=digest,
                until_date=parent.until_date,
            )
        )

        if end >= total_units:
            break

        if child_overlap_chars == 0:
            pos = end
            continue

        # 向回吸纳完整尾部单元作为重叠前缀（严格保证 overlap_start > pos，且为新单元 units[end] 留足空间）
        overlap_start = end
        overlap_chars = 0
        for probe in range(end - 1, pos, -1):
            u_len = len(units[probe].text)
            sep_len = 0 if overlap_start == end else len(units[probe + 1].sep_from_prev)
            cand_chars = overlap_chars + sep_len + u_len
            if cand_chars > child_overlap_chars:
                break
            overlap_chars = cand_chars
            overlap_start = probe

        # 确保下一子块必定装入至少一个新单元 units[end]，避免退化为纯尾部重复子块
        next_unit_need = len(units[end].sep_from_prev) + len(units[end].text)
        while overlap_start < end and (overlap_chars + next_unit_need) > child_max_chars:
            dropped_len = len(units[overlap_start].text)
            sep_after = 0 if overlap_start + 1 == end else len(units[overlap_start + 1].sep_from_prev)
            overlap_chars = max(0, overlap_chars - dropped_len - sep_after)
            overlap_start += 1

        pos = overlap_start

    return children


# 剔除记忆文档中所有带合法 (until: YYYY-MM-DD) 时效标签的完整行区间（含多行缩进过期列表项、过期标题与表格过期行）
def strip_expiring_lines(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    blocks, _ = _parse_atomic_blocks(lines, track_expiry=True)
    drop_lines: set[int] = set()
    for blk in blocks:
        if blk.until_date or blk.kind == "expiring":
            for ln in range(blk.start_line, blk.end_line + 1):
                drop_lines.add(ln)
    if not drop_lines:
        return text
    return "\n".join(line for idx, line in enumerate(lines, start=1) if idx not in drop_lines)


# 将单篇 Markdown/纯文本切分为两层父子结构块 (ParentChunk 列表, ChildChunk 列表)
def chunk_markdown_hierarchy(
    path: str,
    text: str,
    *,
    parent_max_chars: int = DEFAULT_PARENT_MAX_CHARS,
    child_max_chars: int = DEFAULT_CHILD_MAX_CHARS,
    child_overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS,
    track_expiry: bool = False,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    if isinstance(parent_max_chars, bool) or not isinstance(parent_max_chars, int) or parent_max_chars < 1:
        raise ValueError("parent_max_chars 必须是正整数。")
    if isinstance(child_max_chars, bool) or not isinstance(child_max_chars, int) or child_max_chars < 1:
        raise ValueError("child_max_chars 必须是正整数。")
    if (
        isinstance(child_overlap_chars, bool)
        or not isinstance(child_overlap_chars, int)
        or child_overlap_chars < 0
        or child_overlap_chars >= child_max_chars
    ):
        raise ValueError("child_overlap_chars 必须是小于 child_max_chars 的非负整数。")

    lines = text.splitlines()
    if not lines or not any(line.strip() for line in lines):
        return [], []

    raw_blocks, headings_seen = _parse_atomic_blocks(lines, track_expiry=track_expiry)
    if not raw_blocks:
        if not headings_seen:
            return [], []
        last_line, last_section = headings_seen[-1]
        only_parent = _build_parent_chunk(
            path,
            0,
            headings_seen[0][0],
            last_line,
            last_section,
            "heading",
            lines,
        )
        only_children = _build_children_for_parent(
            only_parent,
            child_max_chars=child_max_chars,
            child_overlap_chars=child_overlap_chars,
        )
        return [only_parent], only_children

    normalized_blocks: list[_AtomicBlock] = []
    for blk in raw_blocks:
        normalized_blocks.extend(_split_oversized_block(lines, blk, max_chars=parent_max_chars))

    parents: list[ParentChunk] = []
    buf_start: int | None = None
    buf_end: int | None = None
    buf_section_id: int | None = None
    buf_section = ""
    buf_kind = "prose"

    # 将当前同章节缓冲区落成一条连续行范围的 ParentChunk
    def _flush() -> None:
        nonlocal buf_start, buf_end, buf_section_id, buf_section, buf_kind
        if buf_start is None or buf_end is None:
            return
        parents.append(
            _build_parent_chunk(
                path,
                len(parents),
                buf_start,
                buf_end,
                buf_section,
                buf_kind,
                lines,
            )
        )
        buf_start = None
        buf_end = None
        buf_section_id = None
        buf_section = ""
        buf_kind = "prose"

    for blk in normalized_blocks:
        # 独立时效块及代码块/表格在前后双向均保持独立边界，不与相邻段落混拼
        if blk.standalone or blk.kind in {"code", "table"}:
            _flush()
            parents.append(
                _build_parent_chunk(
                    path,
                    len(parents),
                    blk.start_line,
                    blk.end_line,
                    blk.section,
                    blk.kind,
                    lines,
                    until_date=blk.until_date,
                    table_header=blk.table_header,
                )
            )
            continue

        if buf_start is None or buf_end is None:
            buf_start = blk.start_line
            buf_end = blk.end_line
            buf_section_id = blk.section_id
            buf_section = blk.section
            buf_kind = blk.kind
            continue

        # 跨标题章节（含同名新标题，按单调 section_id 判定）绝不合并
        if blk.section_id != buf_section_id:
            _flush()
            buf_start = blk.start_line
            buf_end = blk.end_line
            buf_section_id = blk.section_id
            buf_section = blk.section
            buf_kind = blk.kind
            continue

        merged_len = len("\n".join(lines[buf_start - 1 : blk.end_line]))
        if merged_len <= parent_max_chars:
            buf_end = blk.end_line
            if buf_kind != blk.kind:
                buf_kind = "prose"
        else:
            _flush()
            buf_start = blk.start_line
            buf_end = blk.end_line
            buf_section_id = blk.section_id
            buf_section = blk.section
            buf_kind = blk.kind

    _flush()

    children: list[ChildChunk] = []
    for parent in parents:
        children.extend(
            _build_children_for_parent(
                parent,
                child_max_chars=child_max_chars,
                child_overlap_chars=child_overlap_chars,
            )
        )
    return parents, children
