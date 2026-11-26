"""固定 argv 白名单命令加载器、子进程执行器与总配置解析。"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from personal_assistant.agent.plugins import ObserverProtocol, PluginManager
from personal_assistant.agent.tools import (
    Tool,
    ToolRegistry,
    ToolsetProvider,
    build_tool,
)
from personal_assistant.ai.client import (
    LLMClient,
    from_env,
    from_model_profile,
    register_provider,
)
from personal_assistant.ai.types import ModelsConfig
from personal_assistant.assistant.common import REPO_ROOT, truncate_or_offload

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
            "TMPDIR": tempfile.gettempdir(),
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

    # 统一通过 build_tool 将当前命令规格包装成 CommandToolWrapper(Tool) 子类实例（支持注入实例专属 offload_dir）
    def to_tool(self, *, offload_dir: Path | None = None) -> "CommandToolWrapper":
        bound_spec = (
            replace(self, offload_dir=offload_dir)
            if offload_dir is not None and self.offload_dir is None
            else self
        )
        param_desc = {name: spec.description for name, spec in bound_spec.params.items()}
        param_enums = {name: spec.enum for name, spec in bound_spec.params.items() if spec.enum}
        cmd_preview = " ".join(bound_spec.argv)
        built = build_tool(
            bound_spec.run,
            name=bound_spec.tool_name,
            description=f"执行固定白名单命令 `{cmd_preview}`。{bound_spec.description}",
            parameters=param_desc,
            param_enums=param_enums or None,
            risk="exec",
            source_type="command",
            source_name="tools.json",
            base_cls=CommandToolWrapper,
            extra_fields={"spec": bound_spec},
        )
        assert isinstance(built, CommandToolWrapper)
        return built


# 固定白名单命令的 Tool 子类包装器（对标 McpToolWrapper / CommandToolWrapper）
@dataclass(frozen=True)
class CommandToolWrapper(Tool):
    spec: CommandSpec | None = field(default=None, kw_only=True)

    # 委派给绑定的 CommandSpec.run 执行固定命令
    def run(self, **values: str) -> str:
        if self.spec is not None:
            return self.spec.run(**values)
        return super().run(**values)


# 固定命令工具集提供方（继承 ToolsetProvider）：按统一上下文将 commands 适配为 CommandToolWrapper(Tool) 子类列表
@dataclass(frozen=True)
class CommandToolsetProvider(ToolsetProvider):
    commands: tuple[CommandSpec, ...]
    source_type: str = "command"
    source_name: str = "tools.json"

    # 返回默认无上下文时的 CommandToolWrapper 列表
    def tools(self) -> list[Tool]:
        return [spec.to_tool() for spec in self.commands]

    # 按运行时上下文（如 offload_dir）通过 build_tool 构建全部 CommandToolWrapper 实例
    def build_tools(self, context: Mapping[str, object] | None = None) -> list[Tool]:
        raw_offload = context.get("offload_dir") if context is not None else None
        offload_dir = Path(str(raw_offload)).resolve() if isinstance(raw_offload, (str, Path)) else None
        return [spec.to_tool(offload_dir=offload_dir) for spec in self.commands]


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


# config/tools.json 的完整配置对象（管理白名单、插件管理器、检索/配置编译/观测插件、文件策略与固定命令）
@dataclass(frozen=True)
class ToolsConfig:
    enabled_tools: tuple[str, ...]
    files: dict[str, object]
    commands: tuple[CommandSpec, ...]
    plugin_manager: PluginManager
    plugins: tuple[str, ...] = ()
    retrieval: dict[str, object] = field(default_factory=dict)
    telemetry: dict[str, object] = field(default_factory=dict)
    project_root: Path = REPO_ROOT
    config_path: Path | None = None
    raw_config: dict[str, object] = field(default_factory=dict)

    # 从 tools.json 读取并校验白名单配置，同时通过 PluginManager 扫描加载 plugins 目录中的全部插件
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

        telemetry_raw = raw.get("telemetry", {})
        if not isinstance(telemetry_raw, dict):
            raise ValueError("telemetry 必须是 JSON 对象。")

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

        manager = PluginManager.from_entries(
            plugins_raw,
            root,
            context={"project_root": root, "config_path": resolved_cfg, "raw_config": dict(raw)},
        )
        for prov_name, prov_factory in manager.providers.items():
            register_provider(prov_name, prov_factory)

        return cls(
            enabled_tools=tuple(enabled_raw),
            files=dict(files_raw),
            commands=tuple(commands),
            plugin_manager=manager,
            plugins=tuple(plugins_raw),
            retrieval=dict(retrieval_raw),
            telemetry=dict(telemetry_raw),
            project_root=root,
            config_path=resolved_cfg,
            raw_config=dict(raw),
        )

    # 按 telemetry 配置实例化全部已注册的观测插件；telemetry.enabled=false 时返回空列表
    def build_observers(self) -> list[ObserverProtocol]:
        if self.telemetry.get("enabled") is False:
            return []
        cfg = dict(self.telemetry)
        instances: list[ObserverProtocol] = []
        for name, factory in sorted(self.plugin_manager.observers.items()):
            obs = factory(cfg)
            if not isinstance(obs, ObserverProtocol):
                raise TypeError(f"观测插件 {name!r} 返回值未实现 ObserverProtocol。")
            instances.append(obs)
        return instances

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
            compiler = self.plugin_manager.compilers.get("models_dsl")
            if compiler is None:
                raise ValueError("发现 config/models.dsl，但未加载 models_dsl 编译插件。")
            compiled = compiler.compile(dsl_path.read_text(encoding="utf-8"), dsl_path.name)
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
        compiler = self.plugin_manager.compilers.get("models_dsl")
        return list(compiler.presets()) if compiler is not None else []

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
        compiler = self.plugin_manager.compilers.get("models_dsl")
        if compiler is None:
            raise ValueError("未加载 models_dsl 配置插件，无法自动生成模型配置。")
        dsl_text = compiler.render(
            preset=preset,
            api_key=api_key,
            model=model,
            base_url=base_url,
            name=name,
            temperature=temperature,
        )
        # 先在内存中编译校验，确认无任何语法或语义错误后再原子落盘
        compiled = compiler.compile(dsl_text, "models.dsl")
        models_cfg = ModelsConfig.from_dict(compiled)
        cfg_dir = config_dir.resolve()
        _atomic_write_private_text(cfg_dir / "models.dsl", dsl_text)
        _atomic_write_private_json(cfg_dir / "models.json", compiled)
        return models_cfg

    # 按预设 ID、别名或默认模型名查找已加载编译插件暴露的厂商预设
    def _match_preset(self, raw_name: str) -> dict[str, object] | None:
        key = raw_name.strip().lower()
        if not key:
            return None
        for preset in self.model_presets():
            raw_aliases = preset.get("aliases")
            aliases = (
                {str(a).strip().lower() for a in raw_aliases if str(a).strip()}
                if isinstance(raw_aliases, (list, tuple))
                else set()
            )
            preset_id = str(preset.get("id") or "").strip().lower()
            default_model = str(preset.get("default_model") or "").strip().lower()
            if key in {preset_id, default_model} - {""} or key in aliases:
                return preset
        return None

    # 基于 models.dsl / models.json、内置预设或环境变量统一构造 LLMClient
    def build_llm(
        self,
        config_dir: Path,
        *,
        model_name: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> LLMClient:
        clean_name = (model_name or "").strip() or None
        source_env = os.environ if env is None else env
        models_cfg = self.load_models_config(config_dir)
        if models_cfg is not None:
            target = clean_name
            if target is not None and target not in models_cfg.models:
                preset = self._match_preset(target)
                preset_id = str(preset.get("id") or "").strip() if preset is not None else ""
                if preset_id and preset_id in models_cfg.models:
                    target = preset_id
            return from_model_profile(models_cfg.get_profile(target), env=source_env)
        if clean_name is None:
            return from_env(source_env)
        merged = dict(source_env)
        preset = self._match_preset(clean_name)
        if preset is not None:
            merged["LLM_PROVIDER"] = str(preset.get("provider") or "openai_compatible")
            merged["LLM_MODEL"] = str(preset.get("default_model") or clean_name)
            if not merged.get("LLM_BASE_URL", "").strip() and preset.get("base_url"):
                merged["LLM_BASE_URL"] = str(preset["base_url"])
            if preset.get("needs_key") is False and not merged.get("LLM_API_KEY", "").strip():
                merged["LLM_API_KEY"] = str(preset.get("id") or "local")
        else:
            merged["LLM_MODEL"] = clean_name
        return from_env(merged)

    # 按「显式参数 > models.dsl/json 的 consolidator 槽位 > 纯环境变量模式下 LLM_CONSOLIDATOR_MODEL > 复用 fallback_llm」构造后台沉淀模型
    def build_consolidator_llm(
        self,
        config_dir: Path,
        *,
        fallback_llm: LLMClient,
        model_name: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> LLMClient:
        clean_name = (model_name or "").strip() or None
        source_env = os.environ if env is None else env
        if clean_name is not None:
            return self.build_llm(config_dir, model_name=clean_name, env=source_env)
        models_cfg = self.load_models_config(config_dir)
        if models_cfg is not None:
            if models_cfg.consolidator:
                return from_model_profile(models_cfg.get_profile(models_cfg.consolidator), env=source_env)
            return fallback_llm
        env_val = str(source_env.get("LLM_CONSOLIDATOR_MODEL") or "").strip()
        if env_val:
            return self.build_llm(config_dir, model_name=env_val, env=source_env)
        return fallback_llm

    # 根据 retrieval 配置或显式名称实例化对应的检索插件；未启用或设为 linear 时返回 None
    def build_retriever(self, override_name: str | None = None) -> object | None:
        raw_name = override_name if override_name is not None else self.retrieval.get("plugin")
        plugin_name = str(raw_name or "").strip()
        if not plugin_name or plugin_name == "linear":
            return None
        registry = self.plugin_manager.retrievers
        if plugin_name not in registry:
            available = "、".join(sorted(registry)) or "（无）"
            raise ValueError(f"未找到检索插件 {plugin_name!r}，当前已加载：{available}")
        return registry[plugin_name](dict(self.retrieval))

    # 统一将所有 ToolsetProvider（内置 / 插件 / 命令）按同一套 build_ctx 汇入单个 ToolRegistry 并按需激活
    def build_registry(
        self,
        *builtin_tools: Tool,
        extra_tools: Iterable[Tool] = (),
        providers: Iterable[ToolsetProvider] = (),
        data_root: Path | None = None,
        tool_context: Mapping[str, object] | None = None,
    ) -> ToolRegistry:
        extras = list(extra_tools)
        allowed = [
            *self.enabled_tools,
            *(spec.tool_name for spec in self.commands),
            *(item.name for item in extras),
        ]
        registry = ToolRegistry(allowed_names=allowed)
        resolved_data = data_root.resolve() if data_root is not None else None
        offload_dir = (resolved_data / "workspace" / "tool_outputs") if resolved_data is not None else None
        build_ctx: dict[str, object] = {
            "project_root": self.project_root,
            "config_path": self.config_path,
            "data_root": resolved_data,
            "offload_dir": offload_dir,
            "raw_config": self.raw_config,
        }
        if tool_context:
            build_ctx["extra_bindings"] = dict(tool_context)

        for provider in providers:
            registry.register_provider(provider, context=build_ctx)
        for item in builtin_tools:
            registry.register(item)
        registry.register_provider(self.plugin_manager, context=build_ctx)
        registry.register_provider(CommandToolsetProvider(self.commands), context=build_ctx)
        for item in extras:
            registry.register(item)

        return registry
