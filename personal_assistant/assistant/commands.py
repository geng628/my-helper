"""固定 argv 白名单命令加载器、子进程执行器与总配置解析。"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from personal_assistant.agent.plugins import ConfigCompiler, RetrieverFactory, load_plugins
from personal_assistant.agent.tools import Tool, ToolRegistry
from personal_assistant.ai.client import (
    LLMClient,
    from_env,
    from_model_profile,
    register_provider,
)
from personal_assistant.ai.types import ModelsConfig
from personal_assistant.assistant.common import truncate_or_offload

COMMAND_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
PLACEHOLDER_RE = re.compile(r"^\{([a-z][a-z0-9_]*)\}$")
FORBIDDEN_CHARS_RE = re.compile(r"[\x00\r\n]")


# 固定命令模板中单个占位符参数的约束规则（描述、枚举白名单、正则格式）
@dataclass(frozen=True)
class ParamSpec:
    description: str
    enum: tuple[str, ...] = ()
    pattern: str = ""

    # 校验并清洗模型传入的参数值，拦截换行/NUL/非法枚举及不匹配正则的输入
    def validate(self, value: str) -> str:
        text = value.strip()
        if not text or FORBIDDEN_CHARS_RE.search(text):
            raise ValueError("参数必须是非空单行文本。")
        if self.enum and text not in self.enum:
            raise ValueError(f"参数只能是：{', '.join(self.enum)}")
        if self.pattern and not re.fullmatch(self.pattern, text):
            raise ValueError("参数格式不符合要求。")
        return text


# 一条白名单外部命令的完整规格（固定 argv 模板、工作目录、超时与输出上限）
@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    argv: tuple[str, ...]
    params: dict[str, ParamSpec]
    cwd: Path
    timeout: float = 10.0
    max_output_bytes: int = 8192
    offload_dir: Path | None = None

    # 返回注册到工具表中的统一名称（带 cmd_ 前缀）
    @property
    def tool_name(self) -> str:
        return f"cmd_{self.name}"

    # 用校验通过的参数值填充 argv 占位符，生成最终可执行参数数组
    def build_argv(self, values: dict[str, str]) -> tuple[str, ...]:
        if set(values) != set(self.params):
            raise ValueError(f"参数必须正好是：{', '.join(self.params) or '（无）'}")
        resolved = {name: spec.validate(values[name]) for name, spec in self.params.items()}
        final: list[str] = []
        for part in self.argv:
            match = PLACEHOLDER_RE.fullmatch(part)
            final.append(resolved[match.group(1)] if match else part)
        return tuple(final)

    # 以 shell=False 和精简环境变量白名单安全执行命令，返回退出码与截断输出
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

        clean_out = (done.stdout or "").strip()
        clean_err = (done.stderr or "").strip()
        sections: list[str] = []
        if clean_out:
            sections.append(clean_out)
        if clean_err:
            sections.append(f"stderr:\n{clean_err}")
        if not sections:
            return f"退出码：{done.returncode}"
        body = truncate_or_offload(
            "\n".join(sections),
            self.max_output_bytes,
            tag=f"cmd-{self.name}",
            mode="head_tail",
            offload_dir=self.offload_dir,
        )
        return f"退出码：{done.returncode}\n{body}"

    # 将当前命令规格转换成可供 ToolRegistry 注册的 Tool 对象
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


# 解析并严格校验 tools.json 中 commands 数组里的单条命令配置
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


# 以 0600 仅属主可读权限原子落盘私密文本文件（如 models.dsl / models.json），防误泄露
def _atomic_write_private_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-models-", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
        os.chmod(target, 0o600)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# 以 0600 仅属主可读权限原子落盘私密 JSON 配置（如编译后的 config/models.json），防误泄露
def _atomic_write_private_json(target: Path, data: dict[str, object]) -> None:
    _atomic_write_private_text(target, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


# config/tools.json 的完整配置对象（管理白名单、插件工具、检索/配置编译插件、文件策略与固定命令）
@dataclass(frozen=True)
class ToolsConfig:
    enabled_tools: tuple[str, ...]
    files: dict[str, object]
    commands: tuple[CommandSpec, ...]
    plugins: tuple[str, ...] = ()
    plugin_tools: tuple[Tool, ...] = ()
    retrieval: dict[str, object] = field(default_factory=dict)
    plugin_retrievers: tuple[tuple[str, RetrieverFactory], ...] = ()
    plugin_compilers: tuple[tuple[str, ConfigCompiler], ...] = ()

    # 从 tools.json 读取并校验白名单配置，同时自动扫描加载 plugins 目录中的全部插件
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

        retrieval_raw = raw.get("retrieval", {})
        if not isinstance(retrieval_raw, dict):
            raise ValueError("retrieval 必须是 JSON 对象。")

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

        loaded_tools, loaded_retrievers, loaded_providers, loaded_compilers = load_plugins(
            plugins_raw, root
        )
        for prov_name, prov_factory in loaded_providers.items():
            register_provider(prov_name, prov_factory)

        return cls(
            enabled_tools=tuple(enabled_raw),
            files=dict(files_raw),
            commands=tuple(commands),
            plugins=tuple(plugins_raw),
            plugin_tools=tuple(loaded_tools),
            retrieval=dict(retrieval_raw),
            plugin_retrievers=tuple(sorted(loaded_retrievers.items())),
            plugin_compilers=tuple(sorted(loaded_compilers.items())),
        )

    # 加载模型配置：优先通过 models_dsl 插件编译 models.dsl（可选原子落盘为 0600 权限的 models.json），否则直接读 models.json
    def load_models_config(
        self,
        config_dir: Path,
        *,
        write_json: bool = False,
    ) -> ModelsConfig | None:
        cfg_dir = config_dir.resolve()
        dsl_path = cfg_dir / "models.dsl"
        json_path = cfg_dir / "models.json"
        if dsl_path.is_file():
            compilers = dict(self.plugin_compilers)
            compiler = compilers.get("models_dsl")
            if compiler is None:
                raise ValueError("发现 config/models.dsl，但未加载 models_dsl 编译插件。")
            compiled = compiler(dsl_path.read_text(encoding="utf-8"), dsl_path.name)
            models_cfg = ModelsConfig.from_dict(compiled)
            if write_json:
                _atomic_write_private_json(json_path, compiled)
            return models_cfg
        if json_path.is_file():
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            return ModelsConfig.from_dict(raw)
        return None

    # 获取 models_dsl 插件暴露的内置厂商预设目录，供终端向导与网页设置表单展示
    def model_presets(self) -> list[dict[str, object]]:
        compiler = dict(self.plugin_compilers).get("models_dsl")
        presets_fn = getattr(compiler, "presets", None)
        if callable(presets_fn):
            return list(presets_fn())
        return []

    # 自动生成极简 config/models.dsl 并编译落盘为 0600 权限的 config/models.json（零手写代码）
    def save_model_setup(
        self,
        config_dir: Path,
        *,
        preset: str,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        name: str = "",
        temperature: float = 0.2,
    ) -> ModelsConfig:
        compiler = dict(self.plugin_compilers).get("models_dsl")
        render_fn = getattr(compiler, "render", None)
        if compiler is None or not callable(render_fn):
            raise ValueError("未加载 models_dsl 配置插件，无法自动生成模型配置。")
        dsl_text = render_fn(
            preset=preset,
            api_key=api_key,
            model=model,
            base_url=base_url,
            name=name,
            temperature=temperature,
        )
        # 先在内存中编译校验，确认无任何语法或语义错误后再原子落盘
        compiled = compiler(dsl_text, "models.dsl")
        models_cfg = ModelsConfig.from_dict(compiled)
        cfg_dir = config_dir.resolve()
        _atomic_write_private_text(cfg_dir / "models.dsl", dsl_text)
        _atomic_write_private_json(cfg_dir / "models.json", compiled)
        return models_cfg

    # 基于 models.dsl / models.json 构造 LLMClient；若均不存在则回退到环境变量 from_env()
    def build_llm(
        self,
        config_dir: Path,
        *,
        model_name: str | None = None,
    ) -> LLMClient:
        models_cfg = self.load_models_config(config_dir)
        if models_cfg is not None:
            return from_model_profile(models_cfg.get_profile(model_name))
        return from_env()

    # 根据 retrieval 配置或显式名称实例化对应的检索插件；未启用或设为 linear 时返回 None
    def build_retriever(self, override_name: str | None = None) -> object | None:
        raw_name = override_name if override_name is not None else self.retrieval.get("plugin")
        plugin_name = str(raw_name or "").strip()
        if not plugin_name or plugin_name == "linear":
            return None
        registry = dict(self.plugin_retrievers)
        if plugin_name not in registry:
            available = "、".join(sorted(registry)) or "（无）"
            raise ValueError(f"未找到检索插件 {plugin_name!r}，当前已加载：{available}")
        return registry[plugin_name](dict(self.retrieval))

    # 将全部已启用的固定命令转换为 Tool 列表
    def command_tools(self) -> list[Tool]:
        return [spec.to_tool() for spec in self.commands]

    # 按 enabled_tools 白名单过滤内置工具与插件工具，再挂载启用的固定命令并生成注册表
    def build_registry(
        self,
        *builtin_tools: Tool,
        extra_tools: Iterable[Tool] = (),
    ) -> ToolRegistry:
        allowset = set(self.enabled_tools)
        selected = [tool for tool in (*builtin_tools, *self.plugin_tools) if tool.name in allowset]
        return ToolRegistry(*selected, *self.command_tools(), *extra_tools)
