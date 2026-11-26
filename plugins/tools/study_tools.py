"""零微内核改动学习助手知识卡片与间隔复习插件：纯 Markdown 落盘、原子写入与精简间隔重复。"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from personal_assistant.agent.tools import tool
from personal_assistant.assistant.common import atomic_write_text, parse_iso_date, today_shanghai

VALID_GRADES = frozenset({"again", "hard", "good", "easy"})
_STUDY_LOCK = threading.Lock()
_TOPIC_PREFIX = "# 学习卡片："
_EMPTY_MSG = "学习卡片目录为空（data/workspace/study 暂无卡片），请先使用 save_study_card 记录知识点。"
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


# 解析逻辑当前日期：缺省取北京时间当日，复用 common.parse_iso_date 校验 date 实例与 YYYY-MM-DD 字符串
def _resolve_today(today: date | str | None) -> date:
    if today is None:
        return today_shanghai()
    if not isinstance(today, (date, str)):
        raise ValueError("逻辑日期 today 必须是 date 或 YYYY-MM-DD 字符串。")
    parsed = parse_iso_date(today)
    if parsed is None:
        raise ValueError(f"日期格式无效，应为 YYYY-MM-DD：{today}")
    return parsed


# 规范化多行要点文本：将全部 splitlines() 行分隔符统一转为 \n 并剥离首尾纯空白行
def _normalize_summary(raw: str) -> str:
    if not isinstance(raw, str) or "\x00" in raw:
        raise ValueError("要点总结必须是非空有效文本且不能包含 NUL。")
    lines = raw.splitlines()
    non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
    if not non_empty:
        raise ValueError("要点总结不能为空。")
    return "\n".join(lines[non_empty[0] : non_empty[-1] + 1])


# 校验专题名称并生成安全文件短名 slug（限 120 字节）与规范化单行展示标题
def _slugify_topic(topic: str) -> tuple[str, str]:
    if not isinstance(topic, str) or "\x00" in topic or not (stripped := topic.strip()):
        raise ValueError("专题名称必须是非空有效文本。")
    if set(stripped) <= {"."} or "/" in stripped or "\\" in stripped or ".." in stripped:
        raise ValueError("专题名称不能为纯点号或包含路径分隔符/..。")
    title = " ".join(stripped.split())
    slug = re.sub(r"-{2,}", "-", _DISALLOWED_SLUG_RE.sub("-", title)).strip("-_").lower()
    if not slug or set(slug) <= {"."}:
        raise ValueError("专题名称未包含有效的中文、字母或数字字符。")
    if len(slug.encode("utf-8")) > 120:
        raise ValueError("专题名称过长（UTF-8 编码不能超过 120 字节）。")
    return slug, title


# 校验并解析学习卡片目录 data/workspace/study，强制要求显式 data_root 并拒绝符号链接
def _resolve_study_dir(data_root: Path | None) -> Path:
    if data_root is None:
        raise ValueError("缺少数据目录 data_root，学习卡片工具需要显式注入 data_root。")
    raw_base = Path(data_root)
    if raw_base.is_symlink():
        raise ValueError("安全校验失败：数据目录不能是符号链接。")
    base_data = raw_base.resolve()
    study_dir = base_data / "workspace" / "study"
    for node in (base_data, base_data / "workspace", study_dir):
        if node.is_symlink() or (node.exists() and not node.is_dir()):
            raise ValueError(f"安全校验失败：路径不能包含符号链接或非目录节点（{node.name}）。")
    if not study_dir.resolve(strict=False).is_relative_to(base_data):
        raise ValueError("安全校验失败：学习卡片目录越界。")
    return study_dir


# 校验目标卡片文件不含符号链接、非普通文件且严格位于 study_dir 目录边界内
def _assert_safe_target(study_dir: Path, target_file: Path) -> None:
    if target_file.is_symlink() or (target_file.exists() and not target_file.is_file()):
        raise ValueError(f"安全校验失败：学习卡片文件不能是符号链接或非普通文件（{target_file.name}）。")
    if not target_file.resolve(strict=False).is_relative_to(study_dir.resolve()):
        raise ValueError("安全校验失败：卡片文件路径越出学习目录边界。")


# 校验学习卡片目录与目标文件边界后，复用 common.atomic_write_text 原子落盘
def _atomic_write(study_dir: Path, target_file: Path, content: str) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    if study_dir.is_symlink():
        raise ValueError("安全校验失败：study 目录不能是符号链接。")
    _assert_safe_target(study_dir, target_file)
    atomic_write_text(target_file, content)


# 解析单个专题 Markdown 文件（仅按 \n 切行与渲染对称），精确还原多行要点
def parse_study_file(text: str, *, default_title: str = "") -> tuple[str, list[StudyCard]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
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


# 将专题标题与卡片列表序列化为人类可读且无损往返的 Markdown 文本（消除任何非 \n 分隔符）
def render_study_file(topic_title: str, cards: list[StudyCard]) -> str:
    blocks = [f"{_TOPIC_PREFIX}{topic_title}"]
    for card in cards:
        indented = "\n".join(f"  {ln}" if ln else "" for ln in card.summary.splitlines())
        blocks.append(f"## {card.concept}\n- 掌握度: {card.grade} | 复习次数: {card.reps} | 间隔天数: {card.interval_days} | 下次复习: {card.due.isoformat()} | 最近更新: {card.updated.isoformat()}\n- 要点与易错点:\n{indented}")
    return "\n\n".join(blocks) + "\n"


# 按严格单调递进 Leitner/SM-2 规则计算下一轮复习次数、间隔天数与到期日（again <= hard <= good <= easy）
def _compute_schedule(grade: str, prev_reps: int, prev_interval: int, ref_day: date) -> tuple[int, int, date]:
    if grade == "again":
        return 0, 1, ref_day + timedelta(days=1)
    reps = prev_reps + 1
    hard_iv = 1 if prev_reps == 0 else min(180, max(2, round(prev_interval * 1.2)))
    good_iv = 1 if prev_reps == 0 else min(180, max(3, prev_interval * 2))
    easy_iv = 4 if prev_reps == 0 else min(365, max(good_iv + 1, round(prev_interval * 2.5)))
    interval = hard_iv if grade == "hard" else (good_iv if grade == "good" else easy_iv)
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
def save_study_card(topic: str, concept: str, summary: str, grade: str, *, data_root: Path | None = None, today: date | str | None = None) -> str:
    slug, topic_title = _slugify_topic(topic)
    if not isinstance(concept, str) or "\x00" in concept or not (norm_concept := " ".join(concept.split())):
        raise ValueError("知识点标题必须是非空有效文本。")
    clean_summary = _normalize_summary(summary)
    if (clean_grade := grade.strip().lower() if isinstance(grade, str) else "") not in VALID_GRADES:
        raise ValueError("掌握度 grade 必须是 again、hard、good 或 easy 之一。")
    ref_day = _resolve_today(today)
    with _STUDY_LOCK:
        study_dir = _resolve_study_dir(data_root)
        target_file = study_dir / f"{slug}.md"
        _assert_safe_target(study_dir, target_file)
        cards: list[StudyCard] = []
        if target_file.exists():
            raw_text = target_file.read_text(encoding="utf-8")
            existing_title, cards = parse_study_file(raw_text, default_title=topic_title)
            if raw_text.strip() and not raw_text.startswith(_TOPIC_PREFIX) and not cards:
                raise ValueError(f"目标文件 {target_file.name} 不是有效学习卡片格式，拒绝覆盖非卡片笔记。")
            topic_title = existing_title or topic_title
        idx = next((i for i, c in enumerate(cards) if c.concept.casefold() == norm_concept.casefold()), -1)
        prev = cards[idx] if idx >= 0 else None
        reps, interval, due = _compute_schedule(clean_grade, prev.reps if prev else 0, prev.interval_days if prev else 0, ref_day)
        card = StudyCard(norm_concept, clean_grade, reps, interval, due, ref_day, clean_summary)
        if idx >= 0:
            cards[idx] = card
        else:
            cards.append(card)
        action = "更新" if idx >= 0 else "创建"
        _atomic_write(study_dir, target_file, render_study_file(topic_title, cards))
    return f"已{action}学习卡片：data/workspace/study/{slug}.md\n- 专题: {topic_title} | 知识点: {norm_concept}\n- 掌握度: {clean_grade} | 复习次数: {reps} | 间隔天数: {interval} 天 | 下次复习: {due.isoformat()}"


# 插件工具：查询学习卡片与间隔复习队列（due / all / 指定专题）
@tool(description="查询 data/workspace/study/ 中的学习卡片与间隔复习队列（传 'due' 查看今日及逾期需复习卡片，传 'all' 查看全局学习概览与统计，或传具体专题名筛选该专题卡片），用于主动出题考评与查漏补缺。", parameters={"scope": "查询范围：'due'（今日及逾期需复习）、'all'（全部概览）或具体专题名称"}, risk="read")
def list_due_reviews(scope: str, *, data_root: Path | None = None, today: date | str | None = None) -> str:
    if not isinstance(scope, str) or "\x00" in scope or not (raw_scope := scope.strip()):
        raise ValueError("查询范围 scope 必须是非空字符串。")
    mode = raw_scope.lower()
    if mode not in {"due", "all"} and (set(raw_scope) <= {"."} or "/" in raw_scope or "\\" in raw_scope or ".." in raw_scope):
        raise ValueError("查询专题名称不能为纯点号、路径分隔符或 ..。")
    ref_day = _resolve_today(today)
    probe_slug = ""
    topics: list[tuple[str, str, list[StudyCard]]] = []
    with _STUDY_LOCK:
        study_dir = _resolve_study_dir(data_root)
        if study_dir.exists():
            if mode not in {"due", "all"}:
                try:
                    probe_slug, _ = _slugify_topic(raw_scope)
                    _assert_safe_target(study_dir, study_dir / f"{probe_slug}.md")
                except ValueError as error:
                    if any(k in str(error) for k in ("符号链接", "越出", "非普通文件")):
                        raise
            for entry in sorted(study_dir.iterdir(), key=lambda p: p.name):
                if not entry.name.startswith(".") and entry.suffix.lower() == ".md":
                    _assert_safe_target(study_dir, entry)
                    title, cards = parse_study_file(entry.read_text(encoding="utf-8"), default_title=entry.stem)
                    if cards:
                        topics.append((entry.stem, title or entry.stem, cards))
    if not (all_items := [(slug, title, c) for slug, title, cs in topics for c in cs]):
        return _EMPTY_MSG
    if mode == "due":
        if not (due_items := sorted((item for item in all_items if item[2].due <= ref_day), key=lambda x: (x[2].due, x[2].interval_days, x[0], x[2].concept))):
            return f"今天（{ref_day.isoformat()}）没有到期或逾期的复习卡片（共 {len(all_items)} 张卡片，最近待复习：{min(c.due for _, _, c in all_items).isoformat()}）。"
        rows = [f"{idx}. [{st[0]}] {t}（{s}）｜{c.concept}｜掌握度: {c.grade} | 复习次数: {c.reps} | 间隔: {c.interval_days} 天 | 到期: {c.due.isoformat()}\n   要点: {st[1]}" for idx, (s, t, c) in enumerate(due_items, 1) for st in (_format_card_status(c.due, ref_day, c.summary),)]
        return "\n".join([f"待复习队列（基准日期：{ref_day.isoformat()}，共 {len(due_items)} 张到期/逾期）：", *rows])
    if mode == "all":
        due_count = sum(1 for _, _, c in all_items if c.due <= ref_day)
        dist = " | ".join(f"{g}={sum(1 for _, _, c in all_items if c.grade == g)}" for g in ("again", "hard", "good", "easy"))
        return _render_topic_sections([f"学习卡片全局概览（基准日期：{ref_day.isoformat()}）", f"- 专题数: {len(topics)} | 总卡片数: {len(all_items)} | 今日待复习: {due_count}", f"- 掌握度分布: {dist}"], topics, ref_day, with_preview=False)
    matched = [(s, t, cs) for s, t, cs in topics if probe_slug and s == probe_slug] or [(s, t, cs) for s, t, cs in topics if mode in t.casefold() or mode in s.casefold()]
    return _render_topic_sections([f"专题卡片清单（查询：{raw_scope}，基准日期：{ref_day.isoformat()}）："], matched, ref_day, with_preview=True) if matched else f"未找到与「{raw_scope}」匹配的学习专题或卡片。"
