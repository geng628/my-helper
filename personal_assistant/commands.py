"""Allowlisted fixed-argv command loader and subprocess runner."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from personal_assistant.plugins import load_plugin_tools
from personal_assistant.tools import Tool, ToolRegistry

COMMAND_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
PLACEHOLDER_RE = re.compile(r"^\{([a-z][a-z0-9_]*)\}$")
FORBIDDEN_CHARS_RE = re.compile(r"[\x00\r\n]")


def truncate_utf8(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore") + "\n…（输出已截断）"


@dataclass(frozen=True)
class ParamSpec:
    description: str
    enum: tuple[str, ...] = ()
    pattern: str = ""

    def validate(self, value: str) -> str:
        text = value.strip()
        if not text or FORBIDDEN_CHARS_RE.search(text):
            raise ValueError("参数必须是非空单行文本。")
        if self.enum and text not in self.enum:
            raise ValueError(f"参数只能是：{', '.join(self.enum)}")
        if self.pattern and not re.fullmatch(self.pattern, text):
            raise ValueError("参数格式不符合要求。")
        return text


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    argv: tuple[str, ...]
    params: dict[str, ParamSpec]
    cwd: Path
    timeout: float = 10.0
    max_output_bytes: int = 8192

    @property
    def tool_name(self) -> str:
        return f"cmd_{self.name}"

    def build_argv(self, values: dict[str, str]) -> tuple[str, ...]:
        if set(values) != set(self.params):
            raise ValueError(f"参数必须正好是：{', '.join(self.params) or '（无）'}")
        resolved = {name: spec.validate(values[name]) for name, spec in self.params.items()}
        final: list[str] = []
        for part in self.argv:
            match = PLACEHOLDER_RE.fullmatch(part)
            final.append(resolved[match.group(1)] if match else part)
        return tuple(final)

    def run(self, **values: str) -> str:
        try:
            argv = self.build_argv(values)
        except ValueError as error:
            return f"错误：{error}"

        executable = shutil.which(argv[0]) if not Path(argv[0]).is_absolute() else argv[0]
        if not executable:
            return f"错误：找不到可执行程序 {argv[0]}。"

        env = {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HOME": str(self.cwd),
            "TMPDIR": str(self.cwd),
        }
        try:
            done = subprocess.run(
                (executable, *argv[1:]),
                shell=False,
                cwd=self.cwd,
                timeout=self.timeout,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return f"错误：命令执行超时（>{self.timeout:g}s）。"
        except OSError as error:
            return f"错误：命令启动失败（{error}）。"

        out = truncate_utf8((done.stdout or "").strip(), self.max_output_bytes)
        err = truncate_utf8((done.stderr or "").strip(), self.max_output_bytes)
        lines = [f"退出码：{done.returncode}"]
        if out:
            lines.append(out)
        if err:
            lines.append(f"stderr:\n{err}")
        return "\n".join(lines)

    def to_tool(self) -> Tool:
        param_desc = {name: spec.description for name, spec in self.params.items()}
        param_enums = {name: spec.enum for name, spec in self.params.items() if spec.enum}
        cmd_preview = " ".join(self.argv)
        return Tool(
            name=self.tool_name,
            description=f"执行固定白名单命令 `{cmd_preview}`。{self.description}",
            parameters=param_desc,
            handler=self.run,
            param_enums=param_enums or None,
        )


def _parse_command_spec(raw: object, project_root: Path) -> tuple[CommandSpec, bool]:
    if not isinstance(raw, dict):
        raise ValueError("commands 项必须是对象。")
    name = raw.get("name")
    if not isinstance(name, str) or not COMMAND_NAME_RE.fullmatch(name):
        raise ValueError(f"命令名不合法：{name!r}")
    argv = raw.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError(f"{name}: argv 必须是非空字符串数组。")
    if any(FORBIDDEN_CHARS_RE.search(part) for part in argv):
        raise ValueError(f"{name}: argv 不能包含换行或 NUL。")

    raw_params = raw.get("params") or {}
    if not isinstance(raw_params, dict):
        raise ValueError(f"{name}: params 必须是对象。")
    params: dict[str, ParamSpec] = {}
    for pname, pval in raw_params.items():
        if not isinstance(pname, str) or not COMMAND_NAME_RE.fullmatch(pname) or not isinstance(pval, dict):
            raise ValueError(f"{name}: 参数定义不合法：{pname!r}")
        enum_raw = pval.get("enum", [])
        if not isinstance(enum_raw, list) or not all(isinstance(v, str) and v for v in enum_raw):
            raise ValueError(f"{name}.{pname}: enum 必须是非空字符串数组。")
        pattern = pval.get("pattern", "")
        if not isinstance(pattern, str):
            raise ValueError(f"{name}.{pname}: pattern 必须是字符串。")
        params[pname] = ParamSpec(
            description=str(pval.get("description", pname)),
            enum=tuple(enum_raw),
            pattern=pattern,
        )

    placeholders: list[str] = []
    for idx, part in enumerate(argv):
        match = PLACEHOLDER_RE.fullmatch(part)
        if match:
            if idx == 0:
                raise ValueError(f"{name}: 可执行文件位置不能使用占位符。")
            placeholders.append(match.group(1))
    if set(placeholders) != set(params) or len(placeholders) != len(set(placeholders)):
        raise ValueError(f"{name}: argv 占位符必须与 params 一一对应。")

    cwd_raw = raw.get("cwd", ".")
    if not isinstance(cwd_raw, str) or not cwd_raw.strip():
        raise ValueError(f"{name}: cwd 必须是非空字符串。")
    cwd_rel = Path(cwd_raw)
    if cwd_rel.is_absolute() or ".." in cwd_rel.parts:
        raise ValueError(f"{name}: cwd 必须是项目内相对路径：{cwd_raw}")
    cwd = (project_root / cwd_rel).resolve()
    if not cwd.is_relative_to(project_root) or not cwd.is_dir():
        raise ValueError(f"{name}: cwd 不存在或超出项目范围：{cwd_raw}")

    timeout = raw.get("timeout", 10)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not (0 < timeout <= 120):
        raise ValueError(f"{name}: timeout 必须在 (0, 120] 秒之间。")
    max_output = raw.get("max_output_bytes", 8192)
    if isinstance(max_output, bool) or not isinstance(max_output, int) or max_output <= 0:
        raise ValueError(f"{name}: max_output_bytes 必须是正整数。")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{name}: enabled 必须是布尔值。")

    spec = CommandSpec(
        name=name,
        description=str(raw.get("description", "")),
        argv=tuple(argv),
        params=params,
        cwd=cwd,
        timeout=float(timeout),
        max_output_bytes=max_output,
    )
    return spec, enabled


@dataclass(frozen=True)
class ToolsConfig:
    enabled_tools: tuple[str, ...]
    files: dict[str, object]
    commands: tuple[CommandSpec, ...]
    plugins: tuple[str, ...] = ()
    plugin_tools: tuple[Tool, ...] = ()

    @classmethod
    def load(cls, config_path: Path, project_root: Path | None = None) -> "ToolsConfig":
        resolved_cfg = config_path.resolve()
        root = (project_root or resolved_cfg.parent.parent).resolve()
        raw = json.loads(resolved_cfg.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("tools.json 顶层必须是 JSON 对象。")

        enabled_raw = raw.get("enabled_tools", [])
        if not isinstance(enabled_raw, list) or not all(isinstance(x, str) and x for x in enabled_raw):
            raise ValueError("enabled_tools 必须是字符串数组。")

        plugins_raw = raw.get("plugins", [])
        if not isinstance(plugins_raw, list) or not all(isinstance(x, str) and x for x in plugins_raw):
            raise ValueError("plugins 必须是字符串数组。")

        files_raw = raw.get("files", {})
        if not isinstance(files_raw, dict):
            raise ValueError("files 必须是 JSON 对象。")

        commands_raw = raw.get("commands", [])
        if not isinstance(commands_raw, list):
            raise ValueError("commands 必须是数组。")

        commands: list[CommandSpec] = []
        seen: set[str] = set()
        for item in commands_raw:
            spec, enabled = _parse_command_spec(item, root)
            if spec.name in seen:
                raise ValueError(f"重复的命令名：{spec.name}")
            seen.add(spec.name)
            if enabled:
                commands.append(spec)

        loaded_plugins = tuple(load_plugin_tools(plugins_raw, root))

        return cls(
            enabled_tools=tuple(enabled_raw),
            files=dict(files_raw),
            commands=tuple(commands),
            plugins=tuple(plugins_raw),
            plugin_tools=loaded_plugins,
        )

    def command_tools(self) -> list[Tool]:
        return [spec.to_tool() for spec in self.commands]

    def build_registry(
        self,
        *builtin_tools: Tool,
        extra_tools: Iterable[Tool] = (),
    ) -> ToolRegistry:
        """Filter built-in & plugin tools by `enabled_tools`, then mount enabled commands & extras."""
        allowset = set(self.enabled_tools)
        selected = [tool for tool in (*builtin_tools, *self.plugin_tools) if tool.name in allowset]
        return ToolRegistry(*selected, *self.command_tools(), *extra_tools)
