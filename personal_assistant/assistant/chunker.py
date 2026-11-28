"""Markdown 结构化分块器：维护多级标题面包屑、完整保护代码块与表格、隔离过期条目并生成内容哈希。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from personal_assistant.assistant.common import parse_iso_date
from personal_assistant.assistant.memory_lifecycle import UNTIL_PATTERN

DEFAULT_CHUNK_MAX_CHARS = 500

_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})(?:[ \t]+|$)(.*)$")
_FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_TABLE_SEP_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$"
)
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")


# 单条结构化文档分块：记录展示路径、精确起止行号、标题层级面包屑、原文、上下文文本与 SHA-256 哈希
@dataclass(frozen=True)
class DocumentChunk:
    path: str
    chunk_index: int
    start_line: int
    end_line: int
    section: str
    text: str
    contextual_text: str
    content_hash: str
    until_date: str = ""


# 内部原子结构块（围栏代码块、Markdown 表格、列表项、段落或独立过期条目，携带单调递增 section_id）
@dataclass(frozen=True)
class _AtomicBlock:
    start_line: int
    end_line: int
    section_id: int
    section: str
    kind: str
    until_date: str = ""
    standalone: bool = False


# 解析单行 ATX 标题，返回 (标题级别 1..6, 清洗后的标题文本)；非标题行返回 None
def _parse_heading(line: str) -> tuple[int, str] | None:
    match = _HEADING_RE.match(line)
    if match is None:
        return None
    level = len(match.group(1))
    title = match.group(2).strip()
    # 剥离尾部装饰性闭合 #（如 "## 标题 ##"）
    title = re.sub(r"[ \t]+#+$", "", title).strip()
    if not title:
        return None
    return level, title


# 提取并校验单行文本中的有效 (until: YYYY-MM-DD) 截止日期，非法日期（如 2026-00-00）按无截止日期处理
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
    # CommonMark §4.5：反引号围栏的 info string 不能包含反引号（避免把行内 ```code``` 误判为未闭合代码块）
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
    # 未闭合代码块收尾裁去尾部纯空行，保证 end_line 与可见正文行数严格对齐
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
                # 带过期标签的列表项同步吸收其缩进续行，避免子行密码/细节在过期后残留
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
                        if _LIST_ITEM_RE.match(nxt) is not None and _leading_indent(nxt) <= base_indent:
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

        # 4. GFM Markdown 表格：从表头 + 分隔行起连续吸收数据行，保证整表结构不被切碎
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
                if _LIST_ITEM_RE.match(nxt) is not None and _leading_indent(nxt) <= base_indent:
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


# 将超出字数上限的单个长块按整行边界切分（绝不腰斩单行；代码块/表格在 3 倍上限内保持完整）
def _split_oversized_block(
    lines: list[str],
    block: _AtomicBlock,
    *,
    max_chars: int,
) -> list[_AtomicBlock]:
    block_text = "\n".join(lines[block.start_line - 1 : block.end_line])
    if block.standalone or len(block_text) <= max_chars:
        return [block]
    # 代码块与表格在 3×max_chars 软上限内优先保持完整结构不被切碎
    if block.kind in {"code", "table"} and len(block_text) <= max_chars * 3:
        return [block]

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


# 构造单个不可变 DocumentChunk，统一拼接路径/标题面包屑上下文并计算 SHA-256 内容哈希
def _build_chunk(
    path: str,
    chunk_index: int,
    start_line: int,
    end_line: int,
    section: str,
    lines: list[str],
    *,
    until_date: str = "",
) -> DocumentChunk:
    body = "\n".join(lines[start_line - 1 : end_line])
    header = f"[{path}] {section}" if section else f"[{path}]"
    contextual = f"{header}\n{body}"
    digest = hashlib.sha256(contextual.encode("utf-8")).hexdigest()
    return DocumentChunk(
        path=path,
        chunk_index=chunk_index,
        start_line=start_line,
        end_line=end_line,
        section=section,
        text=body,
        contextual_text=contextual,
        content_hash=digest,
        until_date=until_date,
    )


# 将单篇 Markdown/纯文本按标题层级与块边界切分为有序 DocumentChunk 列表
def chunk_markdown(
    path: str,
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    track_expiry: bool = False,
) -> list[DocumentChunk]:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("chunk_max_chars 必须是正整数。")
    lines = text.splitlines()
    if not lines or not any(line.strip() for line in lines):
        return []

    raw_blocks, headings_seen = _parse_atomic_blocks(lines, track_expiry=track_expiry)
    if not raw_blocks:
        if not headings_seen:
            return []
        last_line, last_section = headings_seen[-1]
        return [
            _build_chunk(
                path,
                0,
                headings_seen[0][0],
                last_line,
                last_section,
                lines,
            )
        ]

    normalized_blocks: list[_AtomicBlock] = []
    for blk in raw_blocks:
        normalized_blocks.extend(_split_oversized_block(lines, blk, max_chars=max_chars))

    chunks: list[DocumentChunk] = []
    buf_start: int | None = None
    buf_end: int | None = None
    buf_section_id: int | None = None
    buf_section = ""

    # 将当前同章节缓冲区落成一条连续行范围的 DocumentChunk
    def _flush() -> None:
        nonlocal buf_start, buf_end, buf_section_id, buf_section
        if buf_start is None or buf_end is None:
            return
        chunks.append(
            _build_chunk(
                path,
                len(chunks),
                buf_start,
                buf_end,
                buf_section,
                lines,
            )
        )
        buf_start = None
        buf_end = None
        buf_section_id = None
        buf_section = ""

    for blk in normalized_blocks:
        # 独立时效块及代码块/表格在前后双向均保持独立边界，不与相邻段落混拼
        if blk.standalone or blk.kind in {"code", "table"}:
            _flush()
            chunks.append(
                _build_chunk(
                    path,
                    len(chunks),
                    blk.start_line,
                    blk.end_line,
                    blk.section,
                    lines,
                    until_date=blk.until_date,
                )
            )
            continue

        if buf_start is None or buf_end is None:
            buf_start = blk.start_line
            buf_end = blk.end_line
            buf_section_id = blk.section_id
            buf_section = blk.section
            continue

        # 跨标题章节（含同名新标题，按单调 section_id 判定）绝不合并
        if blk.section_id != buf_section_id:
            _flush()
            buf_start = blk.start_line
            buf_end = blk.end_line
            buf_section_id = blk.section_id
            buf_section = blk.section
            continue

        merged_len = len("\n".join(lines[buf_start - 1 : blk.end_line]))
        if merged_len <= max_chars:
            buf_end = blk.end_line
        else:
            _flush()
            buf_start = blk.start_line
            buf_end = blk.end_line
            buf_section_id = blk.section_id
            buf_section = blk.section

    _flush()
    return chunks
