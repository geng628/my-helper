"""Layer 3 无内部依赖基础原语：仓库根目录、时间/日期、正整数校验、原子写盘、.env 加载与 TOAST 输出卸载。"""
from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import threading
import time
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
_CN_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
OFFLOAD_SUBDIR = "tool_outputs"
DEFAULT_MAX_OFFLOAD_BYTES = 1024 * 1024
DEFAULT_SOFT_WRAP_CHARS = 1000
DEFAULT_OFFLOAD_MAX_FILES = 20
DEFAULT_OFFLOAD_TTL_SECONDS = 24 * 3600
_OFFLOAD_CAP_NOTICE = "…（原始输出超过落盘上限，后续字节未保存）"
_OFFLOAD_LOCK = threading.Lock()


# 获取当前北京时间（Asia/Shanghai 带时区 datetime）
def now_shanghai() -> datetime:
    return datetime.now(SHANGHAI_TZ)


# 获取当前北京时间自然日日期
def today_shanghai() -> date:
    return datetime.now(SHANGHAI_TZ).date()


# 安全解析可选日期对象或 YYYY-MM-DD 字符串，非法或空值返回 None
def parse_iso_date(raw: object) -> date | None:
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    candidate = text[:10] if len(text) > 10 and text[10] in {"T", " "} else text
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        return None


# 严格校验参数为正整数（>= 1 且拒绝布尔值）
def require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数。")
    return value


# 严格校验参数为非负有限浮点数（>= 0 且拒绝布尔值、NaN 与 Inf）
def require_non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} 必须是非负有限数字。")
    return float(value)


# 通过同目录 .tmp- 临时文件与 os.replace 原子落盘 UTF-8 文本（可选在写内容前通过 fchmod 设置权限如 0o600）
def atomic_write_text(target: Path, content: str, *, mode: int | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), mode)
            handle.write(content)
        if mode is not None and not hasattr(os, "fchmod"):
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
        if mode is not None:
            os.chmod(target, mode)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# 将日期格式化为带中文星期与时区的统一展示文本（如 2026-11-16（星期一，北京时间））
def format_date_with_weekday(day: date) -> str:
    return f"{day.isoformat()}（{_CN_WEEKDAYS[day.weekday()]}，北京时间）"


# 读取本地 .env 文件（默认仓库根目录 .env）并将未设置的 KEY=VALUE 注入当前进程环境变量
def load_dotenv(dotenv_path: Path | None = None) -> None:
    target = dotenv_path if dotenv_path is not None else (REPO_ROOT / ".env")
    if not target.is_file():
        return
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


# 按 UTF-8 字节上限安全截断长文本，避免单次工具输出撑爆上下文
def truncate_utf8(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore") + "\n…（输出已截断）"


# 返回默认工具大输出落盘目录（遵循 MY_HELPER_DATA 环境变量，回退至仓库 data/workspace/tool_outputs）
def default_offload_dir() -> Path:
    data_root = Path(os.getenv("MY_HELPER_DATA", str(REPO_ROOT / "data"))).resolve()
    return data_root / "workspace" / OFFLOAD_SUBDIR


# 清洗落盘文件名前缀标签，剔除路径穿越字符与非安全符号
def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", (tag or "").strip().lower()).strip("-_")
    return cleaned[:24] or "output"


# 对超长无换行单行做软折行，确保 read_markdown 按行分页与 FTS5 检索有效
def _soft_wrap_lines(text: str, max_line_chars: int) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if max_line_chars <= 0:
        return normalized
    wrapped: list[str] = []
    for line in normalized.split("\n"):
        if len(line) <= max_line_chars:
            wrapped.append(line)
            continue
        pos = 0
        length = len(line)
        while pos < length:
            end = min(pos + max_line_chars, length)
            if end < length:
                split_at = line.rfind(" ", pos + max_line_chars // 2, end + 1)
                if split_at > pos:
                    end = split_at
            piece = line[pos:end].rstrip(" ") if end < length else line[pos:end]
            wrapped.append(piece if piece else line[pos:end])
            pos = end + 1 if (end < length and line[end] == " ") else end
    return "\n".join(wrapped)


# 将待落盘文本严格裁剪到 max_offload_bytes 以内，保证不超过 FTS5 检索文件上限
def _cap_offload_text(text: str, max_offload_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_offload_bytes:
        return text
    notice = f"\n{_OFFLOAD_CAP_NOTICE}"
    notice_bytes = len(notice.encode("utf-8"))
    if max_offload_bytes <= notice_bytes:
        return raw[:max_offload_bytes].decode("utf-8", errors="ignore")
    budget = max_offload_bytes - notice_bytes
    trimmed = raw[:budget].decode("utf-8", errors="ignore")
    if "\n" in trimmed:
        head_part = trimmed.rsplit("\n", 1)[0]
        if head_part:
            trimmed = head_part
    return trimmed + notice


# 懒清理落盘目录中的过期或超量临时输出文件，防止磁盘堆积与 FTS5 内存膨胀
def _cleanup_offload_dir(
    target_dir: Path,
    *,
    keep_name: str,
    max_files: int,
    ttl_seconds: float,
) -> None:
    now = time.time()
    others: list[tuple[int, str, Path]] = []
    for item in target_dir.iterdir():
        if item.name == keep_name or not item.name.endswith(".txt") or item.name.startswith("."):
            continue
        try:
            if not item.is_file() or item.is_symlink():
                continue
            st = item.stat()
        except OSError:
            continue
        if ttl_seconds > 0 and (now - st.st_mtime) > ttl_seconds:
            try:
                item.unlink()
            except OSError:
                pass
            continue
        others.append((st.st_mtime_ns, item.name, item))

    if max_files > 0:
        others.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        for _, _, path in others[max(0, max_files - 1) :]:
            try:
                path.unlink()
            except OSError:
                pass


# 将超长工具输出在互斥锁保护下原子落盘到目标目录并触发懒清理，避免并发写入与清理互删
def _write_offload_file(
    target_dir: Path,
    *,
    tag: str,
    content: str,
    max_files: int,
    ttl_seconds: float,
) -> str:
    payload = content.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    with _OFFLOAD_LOCK:
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        filename = f"{stamp}-{_sanitize_tag(tag)}-{digest}.txt"
        atomic_write_text(target_dir / filename, content)
        _cleanup_offload_dir(
            target_dir,
            keep_name=filename,
            max_files=max_files,
            ttl_seconds=ttl_seconds,
        )
        return filename


# 按 UTF-8 字节上限安全截取文本前缀，不切碎多字节字符
def _slice_utf8_prefix(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


# 按 UTF-8 字节上限安全截取文本后缀，不切碎多字节字符
def _slice_utf8_suffix(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    raw = text.encode("utf-8")
    return raw[-max_bytes:].decode("utf-8", errors="ignore")


# 在字节预算内从头部贪心选取尽可能多的完整行，返回选取行数
def _take_head_lines(lines: list[str], budget: int) -> int:
    used = 0
    count = 0
    for line in lines:
        size = len(line.encode("utf-8")) + (1 if count > 0 else 0)
        if used + size > budget:
            break
        used += size
        count += 1
    return count


# 在字节预算内从尾部贪心选取尽可能多的完整行（不越过 min_index），返回选取行数
def _take_tail_lines(lines: list[str], budget: int, min_index: int = 0) -> int:
    used = 0
    count = 0
    for idx in range(len(lines) - 1, min_index - 1, -1):
        size = len(lines[idx].encode("utf-8")) + (1 if count > 0 else 0)
        if used + size > budget:
            break
        used += size
        count += 1
    return count


# 统一拼装 TOAST 输出截断说明与分页续读/全文检索调用提示
def _format_offload_notice(
    *,
    summary: str,
    total: int,
    display_path: str,
    display_root: str,
    start_line: int,
    end_line: int,
    action: str = "分页读取",
) -> str:
    return (
        f"…（输出已截断：{summary}完整共 {total} 行，已保存至 {display_path}；"
        f'可用 read_markdown(path="{display_path}", start_line={start_line}, end_line={end_line}) {action}，'
        f'或用 search_markdown(query="关键词", root="{display_root}") 检索）'
    )


# 渲染 head 模式整行预览与 TOAST 续读指针（适合网页正文与长文章）
def _render_head_preview(
    lines: list[str],
    max_bytes: int,
    display_path: str,
    display_root: str,
) -> str:
    total = len(lines)
    head_count = _take_head_lines(lines, max_bytes)
    if head_count == 0:
        preview = _slice_utf8_prefix(lines[0], max_bytes)
        notice = _format_offload_notice(
            summary="以上仅含第 1 行前缀，",
            total=total,
            display_path=display_path,
            display_root=display_root,
            start_line=1,
            end_line=min(total, 200),
        )
        return f"{preview}\n{notice}"

    preview = "\n".join(lines[:head_count])
    notice = _format_offload_notice(
        summary=f"以上为第 1-{head_count} 行预览，",
        total=total,
        display_path=display_path,
        display_root=display_root,
        start_line=min(total, head_count + 1),
        end_line=min(total, head_count + 200),
        action="续读",
    )
    return f"{preview}\n{notice}"


# 渲染 head_tail 模式首尾整行预览与中间省略行号指针（适合 pytest 与命令日志）
def _render_head_tail_preview(
    lines: list[str],
    max_bytes: int,
    display_path: str,
    display_root: str,
) -> str:
    total = len(lines)
    head_budget = max_bytes // 2
    tail_budget = max_bytes - head_budget
    head_count = _take_head_lines(lines, head_budget)
    tail_count = _take_tail_lines(lines, tail_budget, min_index=head_count)

    # 若首尾恰好贴合（仅因拼接换行多出 1 字节触发卸载），退让 1 行以保证存在明确的省略区间
    if head_count + tail_count >= total:
        if head_count > 1:
            head_count -= 1
        elif tail_count > 1:
            tail_count -= 1
        else:
            return _render_head_preview(lines, max_bytes, display_path, display_root)

    if head_count >= 1 and tail_count >= 1:
        omit_start = head_count + 1
        omit_end = total - tail_count
        omitted = omit_end - omit_start + 1
        head_block = "\n".join(lines[:head_count])
        tail_block = "\n".join(lines[total - tail_count :])
        notice = _format_offload_notice(
            summary=f"中间省略第 {omit_start}-{omit_end} 行，共省略 {omitted} 行；",
            total=total,
            display_path=display_path,
            display_root=display_root,
            start_line=omit_start,
            end_line=min(omit_end, omit_start + 199),
            action="查看中间内容",
        )
        return f"{head_block}\n\n{notice}\n\n{tail_block}"

    head_block = "\n".join(lines[:head_count]) if head_count > 0 else _slice_utf8_prefix(lines[0], head_budget)
    tail_block = (
        "\n".join(lines[total - tail_count :])
        if tail_count > 0
        else _slice_utf8_suffix(lines[-1], tail_budget)
    )
    start_hint = max(1, head_count + 1)
    notice = _format_offload_notice(
        summary="",
        total=total,
        display_path=display_path,
        display_root=display_root,
        start_line=start_hint,
        end_line=min(total, start_hint + 199),
    )
    return f"{head_block}\n\n{notice}\n\n{tail_block}"


# 极简 TOAST 输出卸载原语：未超限直接透传，超限时原子落盘并返回带行号指针的预览文本
def truncate_or_offload(
    text: str,
    max_bytes: int,
    *,
    tag: str = "output",
    mode: str = "head",
    offload_dir: Path | None = None,
    display_root: str = f"data/workspace/{OFFLOAD_SUBDIR}",
    max_offload_bytes: int = DEFAULT_MAX_OFFLOAD_BYTES,
    soft_wrap_chars: int = DEFAULT_SOFT_WRAP_CHARS,
    max_files: int = DEFAULT_OFFLOAD_MAX_FILES,
    ttl_seconds: float = DEFAULT_OFFLOAD_TTL_SECONDS,
) -> str:
    max_bytes = require_positive_int(max_bytes, "max_bytes")
    if mode not in {"head", "head_tail"}:
        raise ValueError("mode 只能是 head 或 head_tail。")
    if len(text.encode("utf-8")) <= max_bytes:
        return text

    prepared = _cap_offload_text(_soft_wrap_lines(text, soft_wrap_chars), max_offload_bytes)
    if len(prepared.encode("utf-8")) <= max_bytes:
        return prepared

    target_dir = offload_dir.resolve() if offload_dir is not None else default_offload_dir()
    try:
        filename = _write_offload_file(
            target_dir,
            tag=tag,
            content=prepared,
            max_files=max_files,
            ttl_seconds=ttl_seconds,
        )
    except OSError:
        return truncate_utf8(text, max_bytes)

    clean_root = display_root.rstrip("/") or f"data/workspace/{OFFLOAD_SUBDIR}"
    display_path = f"{clean_root}/{filename}"
    lines = prepared.splitlines() or [""]
    if mode == "head_tail":
        return _render_head_tail_preview(lines, max_bytes, display_path, clean_root)
    return _render_head_preview(lines, max_bytes, display_path, clean_root)
