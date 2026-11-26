"""零微内核改动学习助手知识卡片与间隔复习插件：纯 Markdown 落盘、原子写入与精简间隔重复。"""
from __future__ import annotations

import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from personal_assistant.agent.tools import tool

VALID_GRADES = frozenset({"again", "hard", "good", "easy"})
_STUDY_LOCK = threading.Lock()
_TOPIC_PREFIX = "# 学习卡片："
_DISALLOWED_SLUG_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9_-]+")
_META_RE = re.compile(r"^- 掌握度:\s*(again|hard|good|easy)\s*\|\s*复习次数:\s*(\d+)\s*\|\s*间隔天数:\s*(\d+)\s*\|\s*下次复习:\s*(\d{4}-\d{2}-\d{2})\s*\|\s*最近更新:\s*(\d{4}-\d{2}-\d{2})\s*$")


# 单条学习卡片结构：记录知识点标题、掌握度、间隔参数与无损多行要点
@dataclass(frozen=True)
class StudyCard:
    concept: str
    grade: str
    reps: int
    interval_days: int
    due: date
    updated: date
    summary: str


# 解析逻辑当前日期：缺省取北京时间当日，支持 date 实例与 YYYY-MM-DD 字符串
def _resolve_today(today: date | str | None) -> date:
    if today is None:
        return datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if isinstance(today, datetime):
        return today.date()
    if isinstance(today, date):
        return today
    if isinstance(today, str) and today.strip():
        try:
            return date.fromisoformat(today.strip())
        except ValueError as error:
            raise ValueError(f"日期格式无效，应为 YYYY-MM-DD：{today}") from error
    raise ValueError("逻辑日期 today 必须是 date 或 YYYY-MM-DD 字符串。")


# 校验专题名称并生成安全文件短名 slug 与规范化单行展示标题
def _slugify_topic(topic: str) -> tuple[str, str]:
    if not isinstance(topic, str) or "\x00" in topic:
        raise ValueError("专题名称必须是非空有效文本。")
    stripped = topic.strip()
    if not stripped or set(stripped) <= {"."} or "/" in stripped or "\\" in stripped or ".." in stripped:
        raise ValueError("专题名称不能为空、纯点号或包含路径分隔符/..。")
    title = " ".join(stripped.split())
    slug = re.sub(r"-{2,}", "-", _DISALLOWED_SLUG_RE.sub("-", title)).strip("-_").lower()
    if not slug or set(slug) <= {"."}:
        raise ValueError("专题名称未包含有效的中文、字母或数字字符。")
    return slug, title


# 校验并解析学习卡片目录 data/workspace/study，拒绝受控父链与目录符号链接
def _resolve_study_dir(data_root: Path | None, project_root: Path | None) -> Path:
    if project_root is not None and Path(project_root).is_symlink():
        raise ValueError("安全校验失败：project_root 不能是符号链接。")
    raw_base = Path(data_root) if data_root is not None else ((Path(project_root).resolve() / "data") if project_root is not None else Path("data"))
    if raw_base.is_symlink():
        raise ValueError("安全校验失败：数据目录不能是符号链接。")
    if data_root is not None and project_root is not None:
        abs_proj, abs_base = Path(os.path.abspath(project_root)), Path(os.path.abspath(raw_base))
        cur = abs_proj
        for part in (abs_base.relative_to(abs_proj).parts if abs_base.is_relative_to(abs_proj) else ()):
            if (cur := cur / part).is_symlink():
                raise ValueError(f"安全校验失败：父目录不能包含符号链接（{part}）。")
    base_data = raw_base.resolve()
    study_dir = base_data / "workspace" / "study"
    for node in (base_data, base_data / "workspace", study_dir):
        if node.is_symlink() or (node.exists() and not node.is_dir()):
            raise ValueError(f"安全校验失败：路径不能包含符号链接或非目录节点（{node.name}）。")
    if not study_dir.resolve(strict=False).is_relative_to(base_data):
        raise ValueError("安全校验失败：学习卡片目录越界。")
    return study_dir


# 校验目标卡片文件不含符号链接且严格位于 study_dir 目录边界内
def _assert_safe_target(study_dir: Path, target_file: Path) -> None:
    if target_file.is_symlink() or (target_file.exists() and target_file.is_dir()):
        raise ValueError(f"安全校验失败：学习卡片文件不能是符号链接或目录（{target_file.name}）。")
    if not target_file.resolve(strict=False).is_relative_to(study_dir.resolve()):
        raise ValueError("安全校验失败：卡片文件路径越出学习目录边界。")


# 通过同目录临时文件 + os.replace 原子写入卡片文本，防止半截文件损坏
def _atomic_write(study_dir: Path, target_file: Path, content: str) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    if study_dir.is_symlink():
        raise ValueError("安全校验失败：study 目录不能是符号链接。")
    _assert_safe_target(study_dir, target_file)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=study_dir, prefix=f".{target_file.stem}.", suffix=".tmp", delete=False) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
        os.replace(tmp_path, target_file)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


# 解析单个专题 Markdown 文件，精确还原多行要点并防止内嵌标题切碎卡片边界
def parse_study_file(text: str, *, default_title: str = "") -> tuple[str, list[StudyCard]]:
    lines = text.splitlines()
    topic_title = next((ln[len(_TOPIC_PREFIX) :].strip() or default_title for ln in lines if ln.startswith(_TOPIC_PREFIX)), default_title)
    starts = [idx for idx, line in enumerate(lines) if line.startswith("## ")]
    cards: list[StudyCard] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        if not (concept := " ".join(lines[start][3:].split())):
            continue
        meta_match = next((m for i in range(start + 1, end) if (m := _META_RE.match(lines[i])) is not None), None)
        summary_start = next((i + 1 for i in range(start + 1, end) if lines[i].rstrip() == "- 要点与易错点:"), -1)
        if meta_match is None or summary_start < 0:
            continue
        raw_body = list(lines[summary_start:end])
        while raw_body and not raw_body[-1].strip():
            raw_body.pop()
        restored = "\n".join(ln[2:] if ln.startswith("  ") else ("" if not ln.strip() else ln) for ln in raw_body)
        cards.append(StudyCard(concept, meta_match.group(1), int(meta_match.group(2)), int(meta_match.group(3)), date.fromisoformat(meta_match.group(4)), date.fromisoformat(meta_match.group(5)), restored))
    return topic_title, cards


# 将专题标题与卡片列表序列化为人类可读且无损往返的 Markdown 文本
def render_study_file(topic_title: str, cards: list[StudyCard]) -> str:
    blocks = [f"{_TOPIC_PREFIX}{topic_title}"]
    for card in cards:
        indented = "\n".join(f"  {ln}" if ln else "" for ln in card.summary.split("\n"))
        blocks.append(f"## {card.concept}\n- 掌握度: {card.grade} | 复习次数: {card.reps} | 间隔天数: {card.interval_days} | 下次复习: {card.due.isoformat()} | 最近更新: {card.updated.isoformat()}\n- 要点与易错点:\n{indented}")
    return "\n\n".join(blocks) + "\n"


# 按精简 Leitner/SM-2 规则根据掌握度计算下一轮复习次数、间隔天数与到期日
def _compute_schedule(grade: str, prev_reps: int, prev_interval: int, ref_day: date) -> tuple[int, int, date]:
    if grade == "again":
        return 0, 1, ref_day + timedelta(days=1)
    reps = prev_reps + 1
    if grade == "hard":
        interval = 1 if prev_reps == 0 else max(2, round(prev_interval * 1.2))
    elif grade == "good":
        interval = 1 if prev_reps == 0 else (3 if prev_reps == 1 else min(180, prev_interval * 2))
    else:
        interval = 4 if prev_reps == 0 else min(365, max(prev_interval + 1, round(prev_interval * 2.5)))
    return reps, interval, ref_day + timedelta(days=interval)


# 计算单张卡片相对 ref_day 的可读到期状态标签与要点单行预览
def _format_card_status(due: date, ref_day: date, summary: str) -> tuple[str, str]:
    delta = (ref_day - due).days
    label = f"逾期 {delta} 天" if delta > 0 else ("今日到期" if delta == 0 else f"{-delta} 天后到期")
    compact = " ".join(summary.split())
    return label, (compact if len(compact) <= 160 else compact[:159].rstrip() + "…")


# 统一渲染专题卡片列表文本（供 all 概览与单专题筛选复用）
def _render_topic_sections(header_lines: list[str], topics: list[tuple[str, str, list[StudyCard]]], ref_day: date, *, with_preview: bool) -> str:
    lines = list(header_lines)
    for slug, title, cards in topics:
        lines.append(f"\n### {title}（data/workspace/study/{slug}.md，共 {len(cards)} 张）")
        for c in cards:
            label, preview = _format_card_status(c.due, ref_day, c.summary)
            row = f"- {c.concept}｜{label}｜掌握度: {c.grade} | 复习: {c.reps} 次 | 间隔: {c.interval_days} 天 | 下次: {c.due.isoformat()}"
            lines.append(f"{row} | 最近: {c.updated.isoformat()}\n  要点: {preview}" if with_preview else row)
    return "\n".join(lines)


# 插件工具：在 data/workspace/study/ 下按专题创建或幂等更新 Markdown 知识卡片并推进间隔复习计划
@tool(description="仅在用户学习课程/技术概念、整理错题笔记或完成复习自测时调用：在 data/workspace/study/ 下按专题创建或幂等更新 Markdown 知识卡片，并按掌握度自动计算下次间隔复习日期。日常个人偏好、杂事或临时日程切勿调用此工具。", parameters={"topic": "学科或专题短名（如 agent-loop、linear-algebra）", "concept": "具体知识点或错题标题", "summary": "核心推导要点、理解总结或易错点", "grade": "掌握度评级：again（重学）/ hard（吃力）/ good（掌握）/ easy（熟练）"}, param_enums={"grade": ("again", "hard", "good", "easy")}, risk="write")
def save_study_card(topic: str, concept: str, summary: str, grade: str, *, data_root: Path | None = None, project_root: Path | None = None, today: date | str | None = None) -> str:
    slug, topic_title = _slugify_topic(topic)
    if not isinstance(concept, str) or "\x00" in concept or not (norm_concept := " ".join(concept.split())):
        raise ValueError("知识点标题必须是非空有效文本。")
    if not isinstance(summary, str) or "\x00" in summary or not (clean_summary := summary.replace("\r\n", "\n").replace("\r", "\n").strip("\n")).strip():
        raise ValueError("要点总结必须是非空有效文本且不能包含 NUL。")
    clean_grade = grade.strip().lower() if isinstance(grade, str) else ""
    if clean_grade not in VALID_GRADES:
        raise ValueError("掌握度 grade 必须是 again、hard、good 或 easy 之一。")
    ref_day = _resolve_today(today)
    with _STUDY_LOCK:
        study_dir = _resolve_study_dir(data_root, project_root)
        target_file = study_dir / f"{slug}.md"
        _assert_safe_target(study_dir, target_file)
        existing_title, cards = parse_study_file(target_file.read_text(encoding="utf-8"), default_title=topic_title) if target_file.exists() else (topic_title, [])
        topic_title = existing_title or topic_title
        idx = next((i for i, c in enumerate(cards) if c.concept.casefold() == norm_concept.casefold()), -1)
        prev = cards[idx] if idx >= 0 else None
        reps, interval, due = _compute_schedule(clean_grade, prev.reps if prev else 0, prev.interval_days if prev else 0, ref_day)
        card = StudyCard(norm_concept, clean_grade, reps, interval, due, ref_day, clean_summary)
        if idx >= 0:
            cards[idx], action = card, "更新"
        else:
            cards.append(card)
            action = "创建"
        _atomic_write(study_dir, target_file, render_study_file(topic_title, cards))
    return f"已{action}学习卡片：data/workspace/study/{slug}.md\n- 专题: {topic_title} | 知识点: {norm_concept}\n- 掌握度: {clean_grade} | 复习次数: {reps} | 间隔天数: {interval} 天 | 下次复习: {due.isoformat()}"


# 插件工具：查询学习卡片与间隔复习队列（due / all / 指定专题）
@tool(description="查询 data/workspace/study/ 中的学习卡片与间隔复习队列（传 'due' 查看今日及逾期需复习卡片，传 'all' 查看全局学习概览与统计，或传具体专题名筛选该专题卡片），用于主动出题考评与查漏补缺。", parameters={"scope": "查询范围：'due'（今日及逾期需复习）、'all'（全部概览）或具体专题名称"}, risk="read")
def list_due_reviews(scope: str, *, data_root: Path | None = None, project_root: Path | None = None, today: date | str | None = None) -> str:
    if not isinstance(scope, str) or "\x00" in scope or not (raw_scope := scope.strip()):
        raise ValueError("查询范围 scope 必须是非空字符串。")
    mode = raw_scope.lower()
    if mode not in {"due", "all"} and ("/" in raw_scope or "\\" in raw_scope or ".." in raw_scope):
        raise ValueError("查询专题名称不能包含路径分隔符或 ..。")
    ref_day = _resolve_today(today)
    with _STUDY_LOCK:
        study_dir = _resolve_study_dir(data_root, project_root)
        if not study_dir.exists():
            return "学习卡片目录为空（data/workspace/study 暂无卡片），请先使用 save_study_card 记录知识点。"
        probe_slug = ""
        if mode not in {"due", "all"}:
            try:
                probe_slug, _ = _slugify_topic(raw_scope)
                _assert_safe_target(study_dir, study_dir / f"{probe_slug}.md")
            except ValueError as error:
                if "符号链接" in str(error) or "越出" in str(error):
                    raise
        topics: list[tuple[str, str, list[StudyCard]]] = []
        for entry in sorted(study_dir.iterdir(), key=lambda p: p.name):
            if not entry.name.startswith(".") and entry.suffix.lower() == ".md":
                _assert_safe_target(study_dir, entry)
                title, cards = parse_study_file(entry.read_text(encoding="utf-8"), default_title=entry.stem)
                if cards:
                    topics.append((entry.stem, title or entry.stem, cards))
    all_items = [(slug, title, c) for slug, title, cs in topics for c in cs]
    if not all_items:
        return "学习卡片目录为空（data/workspace/study 暂无卡片），请先使用 save_study_card 记录知识点。"
    if mode == "due":
        due_items = sorted((item for item in all_items if item[2].due <= ref_day), key=lambda x: (x[2].due, x[2].interval_days, x[0], x[2].concept))
        if not due_items:
            return f"今天（{ref_day.isoformat()}）没有到期或逾期的复习卡片（共 {len(all_items)} 张卡片，最近待复习：{min(c.due for _, _, c in all_items).isoformat()}）。"
        rows = [f"{i}. [{_format_card_status(c.due, ref_day, c.summary)[0]}] {t}（{s}）｜{c.concept}｜掌握度: {c.grade} | 复习次数: {c.reps} | 间隔: {c.interval_days} 天 | 到期: {c.due.isoformat()}\n   要点: {_format_card_status(c.due, ref_day, c.summary)[1]}" for i, (s, t, c) in enumerate(due_items, 1)]
        return "\n".join([f"待复习队列（基准日期：{ref_day.isoformat()}，共 {len(due_items)} 张到期/逾期）：", *rows])
    if mode == "all":
        due_count = sum(1 for _, _, c in all_items if c.due <= ref_day)
        dist = " | ".join(f"{g}={sum(1 for _, _, c in all_items if c.grade == g)}" for g in ("again", "hard", "good", "easy"))
        return _render_topic_sections([f"学习卡片全局概览（基准日期：{ref_day.isoformat()}）", f"- 专题数: {len(topics)} | 总卡片数: {len(all_items)} | 今日待复习: {due_count}", f"- 掌握度分布: {dist}"], topics, ref_day, with_preview=False)
    matched = [(s, t, cs) for s, t, cs in topics if probe_slug and s == probe_slug] or [(s, t, cs) for s, t, cs in topics if mode in t.casefold() or mode in s.casefold()]
    return _render_topic_sections([f"专题卡片清单（查询：{raw_scope}，基准日期：{ref_day.isoformat()}）："], matched, ref_day, with_preview=True) if matched else f"未找到与「{raw_scope}」匹配的学习专题或卡片。"
