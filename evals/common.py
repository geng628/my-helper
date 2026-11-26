"""评测模块共用工具：加载 JSONL 案例、校验金标引文、获取 Git SHA 与保存报告。"""
from __future__ import annotations

import json
import subprocess
from datetime import date, datetime
from pathlib import Path

from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT
from personal_assistant.assistant.memory import MemoryStore


# 安全解析可选的 ISO 日期字符串（YYYY-MM-DD）或 date 对象，空值返回 None，非法格式抛 ValueError
def parse_optional_date(raw: object) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    return date.fromisoformat(raw.strip())


# 获取当前仓库短 Git Commit SHA（若有未提交改动则追加 -dirty），用于绑定评测基线
def current_git_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# 从 JSONL 文件逐行读取评测案例，校验案例 id 唯一且包含非空 query
def load_cases(cases_path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_no, raw_line in enumerate(cases_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw_line.strip()
        if not text:
            continue
        item = json.loads(text)
        if not isinstance(item, dict):
            raise ValueError(f"{cases_path.name}:{line_no} 必须是 JSON 对象。")
        case_id = str(item.get("id") or f"case-{line_no}").strip()
        if case_id in seen_ids:
            raise ValueError(f"重复的案例 id：{case_id}")
        seen_ids.add(case_id)
        query = str(item.get("query") or "").strip()
        if not query:
            raise ValueError(f"案例 {case_id} 缺少非空 query。")
        item["id"] = case_id
        item["query"] = query
        cases.append(item)
    if not cases:
        raise ValueError("评测案例文件为空。")
    return cases


# 校验每条期望来源（金标）文件可读且引文确实存在于原文，并返回标准化路径列表
def normalize_expected_sources(
    memory: MemoryStore, raw_sources: object
) -> list[dict[str, str]]:
    if not raw_sources:
        return []
    if not isinstance(raw_sources, list):
        raise ValueError("expected_sources 必须是数组。")
    normalized: list[dict[str, str]] = []
    file_cache: dict[Path, str] = {}
    for entry in raw_sources:
        if not isinstance(entry, dict):
            raise ValueError("expected_sources 项必须是对象。")
        raw_path = str(entry.get("path") or "").strip()
        quote = str(entry.get("quote") or "").strip()
        if not raw_path or not quote:
            raise ValueError("expected_sources 每项都需要非空 path 和 quote。")
        resolved = memory.resolve_path(raw_path, write=False)
        if not resolved.is_file():
            raise ValueError(f"金标来源文件不存在：{raw_path}")
        if resolved not in file_cache:
            file_cache[resolved] = resolved.read_text(encoding="utf-8")
        if quote not in file_cache[resolved]:
            raise ValueError(f"金标 quote 未在原文中找到：{raw_path}")
        normalized.append({"path": memory.display_path(resolved), "quote": quote})
    return normalized


# 解析评测用 tools.json 路径与所属项目根目录（外部独立配置文件自动回退仓库根目录以定位内置插件）
def resolve_eval_paths(config_path: Path | None) -> tuple[Path, Path]:
    cfg_path = (config_path or (REPO_ROOT / "config" / "tools.json")).resolve()
    if not cfg_path.is_file():
        return cfg_path, REPO_ROOT
    candidate_root = cfg_path.parent.parent
    project_root = candidate_root if (candidate_root / "plugins").is_dir() else REPO_ROOT
    return cfg_path, project_root


# 统一解析评测配置：优先复用调用方透传的 ToolsConfig，否则按路径单次加载并标记所有权
def resolve_eval_tools_config(
    config_path: Path | None,
    tools_config: ToolsConfig | None = None,
) -> tuple[Path | None, Path, ToolsConfig | None, bool]:
    if tools_config is not None:
        cfg_path = config_path.resolve() if config_path is not None else tools_config.config_path
        return cfg_path, tools_config.project_root, tools_config, False
    cfg_path, project_root = resolve_eval_paths(config_path)
    if cfg_path.is_file():
        return cfg_path, project_root, ToolsConfig.load(cfg_path, project_root=project_root), True
    return cfg_path, project_root, None, False


# 将评测报告写入指定 JSON 文件，禁止输出到 --data-dir 内部以防污染数据目录
def safe_write_report(output_path: Path, data_dir: Path, report: dict[str, object]) -> None:
    resolved_out = output_path.resolve()
    resolved_data = data_dir.resolve()
    if resolved_out.is_relative_to(resolved_data):
        raise ValueError("评测报告输出路径不能放在 --data-dir 目录内。")
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
