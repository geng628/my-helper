"""记忆生命周期模块：负责有效期判断、主题冲突替换（add/merge/replace/remove）与过期归档。"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

UNTIL_PATTERN = re.compile(r"\(until:\s*(\d{4}-\d{2}-\d{2})\)")
MSG_TAG_PATTERN = re.compile(r"\(msg#(\d+)\)")
ECHOED_PREFIX_PATTERN = re.compile(
    r"^(?:-\s+)?(?:\[\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?\]\s*)*(?:\(until:\s*\d{4}-\d{2}-\d{2}\)\s*)?"
)
RESERVED_MEMORY_TOPICS = frozenset({"memory", "audit", "rules", "archive"})
ARCHIVE_INDEX_PATH = "data/memory/archive"
ARCHIVE_INDEX_LINE = (
    "- [历史归档与变更账本](data/memory/archive) — "
    "存放已过期日程与被 replace/remove 软退休（superseded）的历史旧事实及替代链；"
    '仅当追问最初/以前/曾经或对比历史演进时，使用 search_markdown(root="data/memory/archive") 查阅'
)

WriteFn = Callable[[Path, str], None]
IndexFn = Callable[[str, str], None]
AuditFn = Callable[..., dict[str, object]]


# 获取北京时间当天日期，用于统一的记忆有效期与时间戳判断
def today_shanghai() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


# 安全解析 YYYY-MM-DD 格式日期，非法输入返回 None
def parse_iso_date(raw: object) -> date | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return None


# 判断单行记忆是否带 (until: YYYY-MM-DD) 且已严格过期（until < today 为过期，当天仍有效）
def is_line_expired(line: str, today: date | None = None) -> bool:
    match = UNTIL_PATTERN.search(line)
    if not match:
        return False
    until_day = parse_iso_date(match.group(1))
    if until_day is None:
        return False
    ref_day = today or today_shanghai()
    return until_day < ref_day


# 清洗模型输出中可能误复读的列表符、旧日期与旧 until 前缀，避免出现重复时间戳
def clean_memory_body(text: str) -> str:
    collapsed = " ".join(text.split())
    return ECHOED_PREFIX_PATTERN.sub("", collapsed).strip()


# 格式化单条主题记忆行（可选附加 (until: YYYY-MM-DD) 有效期标记）
def format_memory_bullet(text: str, stamp_day: date, until_day: date | None = None) -> str:
    body = clean_memory_body(text)
    if until_day is not None:
        return f"- [{stamp_day.isoformat()}] (until: {until_day.isoformat()}) {body}"
    return f"- [{stamp_day.isoformat()}] {body}"


# 按关键词定位主题文件中匹配的已提交列表条目行号（跳过在途未提交的 turn# 暂存行）
def _find_matching_indices(lines: list[str], old_key: str) -> list[int]:
    if not old_key or "(turn#" in old_key.lower():
        return []
    lowered = old_key.lower()
    return [
        idx
        for idx, line in enumerate(lines)
        if line.lstrip().startswith("- ") and "(turn#" not in line and lowered in line.lower()
    ]


# 判断操作关键词是否命中尚未提交的 (turn#...) 暂存行，避免后台沉淀误改或触发回退新增
def _targets_inflight_turn(lines: list[str], old_key: str) -> bool:
    lowered = old_key.lower()
    return bool(lowered) and (
        "(turn#" in lowered
        or any(line.lstrip().startswith("- ") and "(turn#" in line and lowered in line.lower() for line in lines)
    )


# 判断已有记忆行正文是否已包含目标补充子句（兼容多次 merge 剥离尾部标点，同时避免子串边界误判）
def _bullet_contains_clause(line: str, clause: str) -> bool:
    body = clean_memory_body(line).lower()
    core = clean_memory_body(clause).rstrip("。；;.").strip().lower()
    if not core:
        return True
    pattern = rf"(?:^|[。；;，,]\s*|\.\s+){re.escape(core)}[。；;.]*(?:$|\s*[；;，,。]|\.\s+)"
    return re.search(pattern, body) is not None


# 将补充维度无损拼接至已有记忆行尾部，保留原日期、原正文与既有 until 标记
def _merge_memory_bullet(
    existing_line: str,
    addition: str,
    *,
    stamp_day: date,
    until_day: date | None = None,
) -> str:
    clean_addition = clean_memory_body(addition)
    base = existing_line.rstrip()
    if until_day is not None and not UNTIL_PATTERN.search(base):
        until_tag = f"(until: {until_day.isoformat()}) "
        base = re.sub(
            r"^(\s*-\s+\[\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?\]\s+)",
            rf"\1{until_tag}",
            base,
            count=1,
        )
        if until_tag not in base:
            indent = base[: len(base) - len(base.lstrip())]
            stripped_body = re.sub(r"^\s*-\s+", "", base).strip()
            base = f"{indent}- [{stamp_day.isoformat()}] {until_tag}{stripped_body}"
    sep = "；" if re.search(r"[\u4e00-\u9fff]", base + clean_addition) else "; "
    base = base.rstrip("。；;.")
    return f"{base}{sep}{clean_addition}"


# 读取全部活跃主题记忆文件快照（排除 MEMORY.md 与 archive/），供沉淀器读旧改旧消冲突
def read_topic_snapshots(memory_dir: Path, max_total_chars: int = 6000) -> str:
    if not memory_dir.is_dir():
        return ""
    chunks: list[str] = []
    used = 0
    for path in sorted(memory_dir.glob("*.md")):
        if path.name == "MEMORY.md" or path.name.startswith(".") or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if not content:
            continue
        block = f"### data/memory/{path.name}\n{content}"
        if used + len(block) > max_total_chars:
            remain = max_total_chars - used
            if remain > 120:
                chunks.append(block[:remain] + "\n…（已截断）")
            break
        chunks.append(block)
        used += len(block) + 2
    return "\n\n".join(chunks)


# 将被 replace/remove 软退休的旧事实及替代链追加写入 archive/YYYY-MM.md（跳过正在规范化的 msg# 裸原话）
def _append_superseded_records(
    archive_dir: Path,
    source_name: str,
    retired: list[tuple[str, str]],
    *,
    write_fn: WriteFn,
    ref_day: date,
) -> str:
    effective = [(old_line, repl) for old_line, repl in retired if not MSG_TAG_PATTERN.search(old_line)]
    if not effective:
        return ""
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{ref_day.strftime('%Y-%m')}.md"
    archive_path = archive_dir / archive_name
    existing_archive = (
        archive_path.read_text(encoding="utf-8")
        if archive_path.is_file()
        else f"# 记忆归档与历史变更账本 ({ref_day.strftime('%Y-%m')})\n\n"
    )
    sep = "" if not existing_archive or existing_archive.endswith("\n") else "\n"
    moved = "".join(
        f"- [superseded {ref_day.isoformat()} | from: {source_name}] {old_line} -> superseded_by: {repl}\n"
        for old_line, repl in effective
    )
    write_fn(archive_path, f"{existing_archive}{sep}{moved}")
    return f"data/memory/archive/{archive_name}"


# 对指定主题记忆文件执行 add / merge / replace / remove 操作，写入期消解冲突并将退场旧事实软退休进 archive 账本
def apply_topic_operations(
    memory_dir: Path,
    *,
    topic: str,
    title: str,
    index_summary: str,
    operations: list[dict[str, object]],
    write_fn: WriteFn,
    ensure_index_fn: IndexFn,
    audit_fn: AuditFn,
    archive_dir: Path | None = None,
    today: date | None = None,
) -> str | None:
    slug = re.sub(r"[^a-z0-9_-]+", "-", topic.strip().lower()).strip("-_")[:40]
    if not slug or slug in RESERVED_MEMORY_TOPICS:
        raise ValueError(f"无效的记忆主题名：{topic}")
    if not operations:
        return None

    ref_day = today or today_shanghai()
    memory_dir.mkdir(parents=True, exist_ok=True)
    target = memory_dir / f"{slug}.md"
    rel_path = f"data/memory/{slug}.md"
    clean_title = " ".join(title.split()) or slug
    lines = (
        target.read_text(encoding="utf-8").splitlines()
        if target.is_file()
        else [f"# {clean_title} ({slug}.md)", ""]
    )

    changed = False
    removed_log: list[str] = []
    added_log: list[str] = []
    superseded_records: list[tuple[str, str]] = []

    for op_entry in operations:
        if not isinstance(op_entry, dict):
            continue
        op = str(op_entry.get("op") or "add").strip().lower()
        until_day = parse_iso_date(op_entry.get("until"))

        if op == "add":
            body = clean_memory_body(str(op_entry.get("text") or op_entry.get("new") or ""))
            if not body:
                continue
            if until_day is not None and until_day < ref_day:
                continue
            if any(body in existing_line for existing_line in lines):
                continue
            lines.append(format_memory_bullet(body, ref_day, until_day))
            added_log.append(body)
            changed = True

        elif op == "merge":
            old_key = clean_memory_body(str(op_entry.get("old") or ""))
            addition = clean_memory_body(str(op_entry.get("text") or op_entry.get("new") or ""))
            if not addition:
                continue
            if until_day is not None and until_day < ref_day:
                continue
            matched_indices = _find_matching_indices(lines, old_key)
            if matched_indices:
                idx = matched_indices[0]
                existing_line = lines[idx]
                if _bullet_contains_clause(existing_line, addition):
                    continue
                lines[idx] = _merge_memory_bullet(
                    existing_line, addition, stamp_day=ref_day, until_day=until_day
                )
                added_log.append(addition)
                changed = True
            elif _targets_inflight_turn(lines, old_key):
                continue
            elif not any(
                existing_line.lstrip().startswith("- ")
                and _bullet_contains_clause(existing_line, addition)
                for existing_line in lines
            ):
                lines.append(format_memory_bullet(addition, ref_day, until_day))
                added_log.append(addition)
                changed = True

        elif op == "replace":
            old_key = clean_memory_body(str(op_entry.get("old") or ""))
            new_body = clean_memory_body(str(op_entry.get("new") or op_entry.get("text") or ""))
            if not new_body:
                continue
            if until_day is not None and until_day < ref_day:
                continue
            formatted = format_memory_bullet(new_body, ref_day, until_day)
            matched_indices = _find_matching_indices(lines, old_key)
            if matched_indices:
                first_idx = matched_indices[0]
                for idx in matched_indices:
                    removed_log.append(lines[idx])
                    superseded_records.append((lines[idx], formatted))
                lines[first_idx] = formatted
                for idx in reversed(matched_indices[1:]):
                    lines.pop(idx)
                added_log.append(new_body)
                changed = True
            elif _targets_inflight_turn(lines, old_key):
                continue
            elif not any(new_body in existing_line for existing_line in lines):
                lines.append(formatted)
                added_log.append(new_body)
                changed = True

        elif op == "remove":
            old_key = clean_memory_body(str(op_entry.get("old") or op_entry.get("text") or ""))
            matched_set = set(_find_matching_indices(lines, old_key))
            if not matched_set:
                continue
            kept: list[str] = []
            for idx, line in enumerate(lines):
                if idx in matched_set:
                    removed_log.append(line)
                    superseded_records.append((line, "(removed)"))
                    changed = True
                else:
                    kept.append(line)
            lines = kept

    if not changed:
        return None

    write_fn(target, "\n".join(lines).rstrip() + "\n")
    summary_line = " ".join(index_summary.split()) or f"自动沉淀的 {clean_title} 记录，请按需检索"
    ensure_index_fn(rel_path, f"- [{clean_title}]({rel_path}) — {summary_line}")
    if archive_dir is not None and superseded_records:
        archive_rel = _append_superseded_records(
            archive_dir,
            f"{slug}.md",
            superseded_records,
            write_fn=write_fn,
            ref_day=ref_day,
        )
        if archive_rel:
            ensure_index_fn(ARCHIVE_INDEX_PATH, ARCHIVE_INDEX_LINE)
    audit_fn(
        action="consolidate_memory",
        actor="consolidator",
        target=rel_path,
        old_value="; ".join(removed_log) if removed_log else None,
        new_value="; ".join(added_log) if added_log else None,
    )
    return rel_path


# 扫描活跃主题记忆文件，将已过期的 (until: YYYY-MM-DD) 条目原子归档至 archive/YYYY-MM.md
def archive_expired_topics(
    memory_dir: Path,
    archive_dir: Path,
    *,
    write_fn: WriteFn,
    audit_fn: AuditFn,
    ensure_index_fn: IndexFn | None = None,
    today: date | None = None,
) -> dict[str, int]:
    if not memory_dir.is_dir():
        return {}
    ref_day = today or today_shanghai()
    archived_counts: dict[str, int] = {}
    archive_name = f"{ref_day.strftime('%Y-%m')}.md"
    archive_path = archive_dir / archive_name
    archive_rel = f"data/memory/archive/{archive_name}"

    for path in sorted(memory_dir.glob("*.md")):
        if path.name == "MEMORY.md" or path.name.startswith(".") or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        active_lines: list[str] = []
        expired_lines: list[str] = []
        for line in lines:
            if line.lstrip().startswith("- ") and "(turn#" not in line and is_line_expired(line, today=ref_day):
                expired_lines.append(line)
            else:
                active_lines.append(line)
        if not expired_lines:
            continue

        write_fn(path, "\n".join(active_lines).rstrip() + "\n")
        archive_dir.mkdir(parents=True, exist_ok=True)
        existing_archive = (
            archive_path.read_text(encoding="utf-8")
            if archive_path.is_file()
            else f"# 过期记忆归档 ({ref_day.strftime('%Y-%m')})\n\n"
        )
        sep = "" if not existing_archive or existing_archive.endswith("\n") else "\n"
        moved = "".join(f"{line} (from: {path.name})\n" for line in expired_lines)
        write_fn(archive_path, f"{existing_archive}{sep}{moved}")

        rel_path = f"data/memory/{path.name}"
        archived_counts[rel_path] = len(expired_lines)
        audit_fn(
            action="archive_expired",
            actor="lifecycle",
            target=rel_path,
            old_value="; ".join(expired_lines),
            new_value=archive_rel,
        )
    if archived_counts and ensure_index_fn is not None:
        ensure_index_fn(ARCHIVE_INDEX_PATH, ARCHIVE_INDEX_LINE)
    return archived_counts


# 提取文本的中英文核心检索单元（英文词与中文二元组），用于判断裸原话是否已被规范化沉淀
def _overlap_tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens: set[str] = set(re.findall(r"[a-z0-9_+-]{2,}", lowered))
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(run) == 1:
            tokens.add(run)
        else:
            for idx in range(len(run) - 1):
                tokens.add(run[idx : idx + 2])
    return tokens


# 清理 facts.md 中已被本次沉淀规范化的原始 (msg#...) 裸原话，防止临时事实留存永不过期的幽灵副本
def prune_raw_facts_for_batch(
    memory_dir: Path,
    batch_msg_ids: set[int],
    written_texts: list[str],
    *,
    write_fn: WriteFn,
    audit_fn: AuditFn,
) -> int:
    facts_path = memory_dir / "facts.md"
    if not batch_msg_ids or not written_texts or not facts_path.is_file():
        return 0
    written_tokens: set[str] = set()
    for text in written_texts:
        written_tokens.update(_overlap_tokens(text))
    if not written_tokens:
        return 0

    lines = facts_path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    removed: list[str] = []
    for line in lines:
        match = MSG_TAG_PATTERN.search(line)
        if not match or int(match.group(1)) not in batch_msg_ids:
            kept.append(line)
            continue
        raw_body = line[match.end() :].strip()
        raw_tokens = _overlap_tokens(raw_body)
        shared = raw_tokens & written_tokens
        # 只有原始句与本次写入的规范化记忆有明显语义重合时才移除，绝不误删无关原话
        overlap_ratio = (len(shared) / len(raw_tokens)) if raw_tokens else 0.0
        if len(shared) >= 2 and overlap_ratio >= 0.25:
            removed.append(line)
        else:
            kept.append(line)

    if not removed:
        return 0

    write_fn(facts_path, "\n".join(kept).rstrip() + "\n")
    audit_fn(
        action="dedupe_raw_fact",
        actor="consolidator",
        target="data/memory/facts.md",
        old_value="; ".join(removed),
        new_value=None,
    )
    return len(removed)
