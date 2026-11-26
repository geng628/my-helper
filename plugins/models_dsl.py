"""极简模型配置 DSL 外部编译插件（零第三方依赖）：支持一行预设语法（免写网址/大括号）与完整块语法。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from personal_assistant.agent.plugins import Plugin

COMPILER_NAME = "models_dsl"

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
_ALLOWED_FIELDS = frozenset({"provider", "base_url", "model", "api_key", "key_env", "temp", "temperature"})


# 内置主流大模型厂商预设：普通用户只需选名字 + 填密钥，网址/默认模型/协议由系统自动补全
@dataclass(frozen=True)
class ModelPreset:
    id: str
    label: str
    base_url: str
    default_model: str
    needs_key: bool = True
    aliases: tuple[str, ...] = ()
    provider: str = "openai_compatible"

    # 导出给终端向导与网页下拉框使用的展示字典
    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "needs_key": self.needs_key,
            "provider": self.provider,
            "aliases": list(self.aliases),
        }


MODEL_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset("deepseek", "DeepSeek（推荐）", "https://api.deepseek.com/v1", "deepseek-chat", True, ("ds",)),
    ModelPreset(
        "deepseek_reasoner",
        "DeepSeek Reasoner (R1 深度推理)",
        "https://api.deepseek.com/v1",
        "deepseek-reasoner",
        True,
        ("ds-r1", "reasoner", "r1"),
        "openai_reasoning",
    ),
    ModelPreset(
        "qwen",
        "通义千问 (Qwen)",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
        True,
        ("tongyi", "千问", "通义千问", "dashscope"),
    ),
    ModelPreset("kimi", "Kimi (Moonshot)", "https://api.moonshot.cn/v1", "moonshot-v1-8k", True, ("moonshot",)),
    ModelPreset("glm", "智谱 GLM", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash", True, ("zhipu", "智谱")),
    ModelPreset("openai", "OpenAI", "https://api.openai.com/v1", "gpt-4o-mini", True, ("gpt",)),
    ModelPreset(
        "siliconflow",
        "硅基流动 (SiliconFlow)",
        "https://api.siliconflow.cn/v1",
        "deepseek-ai/DeepSeek-V3",
        True,
        ("硅基流动",),
    ),
    ModelPreset("ollama", "本地 Ollama（免费免密钥）", "http://127.0.0.1:11434/v1", "qwen2.5:7b", False, ("local",)),
)


# 按预设 ID 或中英文别名查找内置厂商预设，未匹配时返回 None
def resolve_preset(raw_name: str) -> ModelPreset | None:
    key = raw_name.strip().lower()
    for preset in MODEL_PRESETS:
        if key == preset.id or key in {a.lower() for a in preset.aliases}:
            return preset
    return None


# 返回所有内置预设的目录列表（供终端向导与网页配置表单直接渲染）
def list_presets() -> list[dict[str, object]]:
    return [preset.to_dict() for preset in MODEL_PRESETS]


# 为 DSL 文本安全格式化取值（含空格或特殊字符时自动加双引号）
def _format_dsl_value(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("配置取值不能包含换行符。")
    clean = value.strip()
    if not clean:
        return '""'
    if any(ch in clean for ch in (" ", "\t", "#", "{", "}", '"', "'")):
        escaped = clean.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return clean


# 根据小白向导/网页表单输入自动生成极简可读的 models.dsl 文本
def render_models_dsl(
    *,
    preset: str,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    name: str = "",
    provider: str = "",
    temperature: float = 0.2,
) -> str:
    header = "# 由 My Helper 自动生成（已默认 .gitignore + 0600 权限 + 沙箱屏蔽保护）\n"
    matched = resolve_preset(preset)
    clean_key = api_key.strip()
    clean_model = model.strip()
    clean_url = base_url.strip()
    clean_provider = provider.strip(" \t").lower()
    final_provider = clean_provider or (matched.provider if matched else "openai_compatible")

    if matched is not None and not clean_url and final_provider == matched.provider and temperature == 0.2:
        if matched.needs_key:
            if not clean_key:
                raise ValueError(f"使用 {matched.label} 需要填写 API Key。")
            if clean_model and clean_model != matched.default_model:
                return f"{header}use {matched.id} {_format_dsl_value(clean_model)} {_format_dsl_value(clean_key)}\n"
            return f"{header}use {matched.id} {_format_dsl_value(clean_key)}\n"
        if clean_model and clean_model != matched.default_model:
            return f"{header}use {matched.id} {_format_dsl_value(clean_model)}\n"
        return f"{header}use {matched.id}\n"

    # 自定义接口或高级覆盖模式：生成清晰的标准 model 块
    profile_name = (name.strip(" \t") or (matched.id if matched else "custom")).lower()
    if not _MODEL_NAME_RE.fullmatch(profile_name):
        raise ValueError(f"模型名称 {profile_name!r} 仅允许字母、数字、下划线、点或连字符。")
    final_url = clean_url or (matched.base_url if matched else "")
    final_model = clean_model or (matched.default_model if matched else "")
    final_key = clean_key or ("" if (matched is None or matched.needs_key) else "ollama")
    if not final_url:
        raise ValueError("自定义模型必须填写接口地址（base_url）。")
    if not final_model:
        raise ValueError("自定义模型必须填写模型名称（model）。")
    lines = [
        header.rstrip("\n"),
        f"default {profile_name}",
        "",
        f"model {profile_name} {{",
    ]
    if final_provider != "openai_compatible":
        lines.append(f"  provider {_format_dsl_value(final_provider)}")
    lines.extend(
        [
            f"  base_url {_format_dsl_value(final_url)}",
            f"  model    {_format_dsl_value(final_model)}",
        ]
    )
    if final_key:
        lines.append(f"  api_key  {_format_dsl_value(final_key)}")
    if temperature != 0.2:
        lines.append(f"  temp     {temperature}")
    lines.append("}")
    return "\n".join(lines) + "\n"


# DSL 编译期异常：继承 ValueError 并附带精确来源文件名、行号与列号
class DSLCompileError(ValueError):
    # 构造带精确位置信息的友好中文报错
    def __init__(self, message: str, *, source: str, line: int, col: int) -> None:
        self.source = source
        self.line = line
        self.col = col
        super().__init__(f"{source} 第 {line} 行第 {col} 列：{message}")


# 词法单元（类型、字面文本、起始行号与列号）
@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    col: int


# 单个 model 块的 AST 节点（记录模型名、字段名到 (值, 行, 列) 的映射及声明位置）
@dataclass(frozen=True)
class ModelDecl:
    name: str
    fields: dict[str, tuple[str, int, int]]
    line: int
    col: int


# 顶层 DSL 程序 AST 节点（可选 default 声明 + 全部 model 块）
@dataclass(frozen=True)
class ModelsDSLProgram:
    default_model: tuple[str, int, int] | None
    models: tuple[ModelDecl, ...]


# 词法分析器：支持 # 注释、单双引号字符串、免引号裸值（URL/密钥/数字）与花括号块
def tokenize_dsl(text: str, *, source: str = "models.dsl") -> list[Token]:
    tokens: list[Token] = []
    line = 1
    col = 1
    idx = 0
    length = len(text)

    while idx < length:
        ch = text[idx]
        if ch == "\r":
            idx += 1
            continue
        if ch == "\n":
            tokens.append(Token("NEWLINE", "\n", line, col))
            idx += 1
            line += 1
            col = 1
            continue
        if ch in {" ", "\t"}:
            idx += 1
            col += 1
            continue
        if ch == "#":
            while idx < length and text[idx] != "\n":
                idx += 1
            continue
        if ch == "{":
            tokens.append(Token("LBRACE", "{", line, col))
            idx += 1
            col += 1
            continue
        if ch == "}":
            tokens.append(Token("RBRACE", "}", line, col))
            idx += 1
            col += 1
            continue
        if ch in {'"', "'"}:
            quote = ch
            start_line, start_col = line, col
            idx += 1
            col += 1
            buf: list[str] = []
            closed = False
            while idx < length:
                cur = text[idx]
                if cur == "\n":
                    raise DSLCompileError("字符串字面量未在行内闭合。", source=source, line=start_line, col=start_col)
                if cur == "\\" and idx + 1 < length:
                    nxt = text[idx + 1]
                    buf.append(nxt)
                    idx += 2
                    col += 2
                    continue
                if cur == quote:
                    idx += 1
                    col += 1
                    closed = True
                    break
                buf.append(cur)
                idx += 1
                col += 1
            if not closed:
                raise DSLCompileError("字符串字面量缺少结束引号。", source=source, line=start_line, col=start_col)
            tokens.append(Token("WORD", "".join(buf), start_line, start_col))
            continue

        start_col = col
        buf = []
        while idx < length and text[idx] not in {" ", "\t", "\r", "\n", "{", "}", "#"}:
            buf.append(text[idx])
            idx += 1
            col += 1
        tokens.append(Token("WORD", "".join(buf), line, start_col))

    tokens.append(Token("EOF", "", line, col))
    return tokens


# 递归下降语法分析器：支持极简一行预设语句（use/add）与完整 model { ... } 代码块
class _DSLParser:
    # 初始化 Token 流与游标
    def __init__(self, tokens: list[Token], *, source: str) -> None:
        self._tokens = tokens
        self._pos = 0
        self._source = source

    # 查看当前游标所指 Token（不前进）
    def _peek(self) -> Token:
        return self._tokens[self._pos]

    # 消费并返回当前 Token
    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    # 跳过连续空行
    def _skip_newlines(self) -> None:
        while self._peek().kind == "NEWLINE":
            self._advance()

    # 断言并消费指定类型的 Token，不匹配时抛出精确行列号的语法错误
    def _expect(self, kind: str, hint: str) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise DSLCompileError(hint, source=self._source, line=tok.line, col=tok.col)
        return self._advance()

    # 校验语句末尾必须是换行或文件结尾，防止同一行拼错多余字段
    def _expect_line_end(self) -> None:
        tok = self._peek()
        if tok.kind == "NEWLINE":
            self._advance()
            return
        if tok.kind in {"EOF", "RBRACE"}:
            return
        raise DSLCompileError(
            f"行尾出现多余内容 {tok.value!r}（每行请只写一条配置项）。",
            source=self._source,
            line=tok.line,
            col=tok.col,
        )

    # 读取同一行剩余的全部 WORD 参数，供一行预设语法 `use/add` 使用
    def _read_line_words(self) -> list[Token]:
        words: list[Token] = []
        while self._peek().kind == "WORD":
            words.append(self._advance())
        self._expect_line_end()
        return words

    # 解析整篇 DSL 文件，产出 ModelsDSLProgram AST
    def parse(self) -> ModelsDSLProgram:
        default_decl: tuple[str, int, int] | None = None
        models: list[ModelDecl] = []

        self._skip_newlines()
        while self._peek().kind != "EOF":
            tok = self._peek()
            if tok.kind != "WORD":
                raise DSLCompileError(
                    f"顶层期望 use、add、default 或 model 声明，实际遇到 {tok.value!r}。",
                    source=self._source,
                    line=tok.line,
                    col=tok.col,
                )
            if tok.value == "default":
                self._advance()
                target = self._expect("WORD", "default 后必须指定默认模型名称。")
                if default_decl is not None:
                    raise DSLCompileError("不允许重复声明 default。", source=self._source, line=tok.line, col=tok.col)
                default_decl = (target.value, target.line, target.col)
                self._expect_line_end()
            elif tok.value in {"use", "add"}:
                decl, is_default = self._parse_preset_line()
                models.append(decl)
                if is_default:
                    if default_decl is not None:
                        raise DSLCompileError(
                            "已设置默认模型，请勿重复使用 default 或多个 use（备用模型请改用 add）。",
                            source=self._source,
                            line=tok.line,
                            col=tok.col,
                        )
                    default_decl = (decl.name, decl.line, decl.col)
            elif tok.value == "model":
                models.append(self._parse_model_block())
            else:
                raise DSLCompileError(
                    f"未知顶层指令 {tok.value!r}（支持 use、add、default 与 model）。",
                    source=self._source,
                    line=tok.line,
                    col=tok.col,
                )
            self._skip_newlines()

        return ModelsDSLProgram(default_model=default_decl, models=tuple(models))

    # 解析极简一行预设语句：`use/add <厂商> [模型名] [api_key]`，自动展开为完整 ModelDecl AST
    def _parse_preset_line(self) -> tuple[ModelDecl, bool]:
        kw = self._advance()
        is_default = kw.value == "use"
        preset_tok = self._expect("WORD", f"{kw.value} 后请填写厂商名，例如：{kw.value} deepseek sk-你的密钥")
        preset = resolve_preset(preset_tok.value)
        if preset is None:
            names = "、".join(p.id for p in MODEL_PRESETS)
            raise DSLCompileError(
                f"未知内置厂商预设 {preset_tok.value!r}（支持：{names}；自定义网址请用 model 块）。",
                source=self._source,
                line=preset_tok.line,
                col=preset_tok.col,
            )

        args = self._read_line_words()
        if len(args) > 2:
            raise DSLCompileError(
                f"{kw.value} {preset.id} 参数过多（最多传 [模型名] [API Key] 两个参数）。",
                source=self._source,
                line=args[2].line,
                col=args[2].col,
            )

        model_val = preset.default_model
        model_loc = (preset_tok.line, preset_tok.col)
        api_key_val = ""
        key_loc = (preset_tok.line, preset_tok.col)

        if preset.needs_key:
            if not args:
                raise DSLCompileError(
                    f"使用 {preset.id} 预设需要提供 API Key，例如：{kw.value} {preset.id} sk-你的密钥",
                    source=self._source,
                    line=preset_tok.line,
                    col=preset_tok.col,
                )
            if len(args) == 1:
                api_key_val = args[0].value
                key_loc = (args[0].line, args[0].col)
            else:
                model_val = args[0].value
                model_loc = (args[0].line, args[0].col)
                api_key_val = args[1].value
                key_loc = (args[1].line, args[1].col)
        else:
            api_key_val = "ollama"
            if len(args) == 1:
                model_val = args[0].value
                model_loc = (args[0].line, args[0].col)
            elif len(args) == 2:
                model_val = args[0].value
                model_loc = (args[0].line, args[0].col)
                api_key_val = args[1].value
                key_loc = (args[1].line, args[1].col)

        fields: dict[str, tuple[str, int, int]] = {
            "provider": (preset.provider, preset_tok.line, preset_tok.col),
            "base_url": (preset.base_url, preset_tok.line, preset_tok.col),
            "model": (model_val, model_loc[0], model_loc[1]),
            "api_key": (api_key_val, key_loc[0], key_loc[1]),
        }
        return ModelDecl(name=preset.id, fields=fields, line=preset_tok.line, col=preset_tok.col), is_default

    # 解析单个 `model <name> { ... }` 高级代码块
    def _parse_model_block(self) -> ModelDecl:
        kw = self._advance()
        name_tok = self._expect("WORD", "model 后必须跟模型名称，例如 `model deepseek {`。")
        self._expect("LBRACE", f"模型 {name_tok.value!r} 声明后缺少左花括号 `{{`。")
        self._skip_newlines()

        fields: dict[str, tuple[str, int, int]] = {}
        while self._peek().kind != "RBRACE":
            if self._peek().kind == "EOF":
                raise DSLCompileError(
                    f"模型 {name_tok.value!r} 代码块缺少闭合右花括号 `}}`。",
                    source=self._source,
                    line=kw.line,
                    col=kw.col,
                )
            key_tok = self._expect("WORD", "模型块内每行必须以字段名开头。")
            val_tok = self._expect("WORD", f"字段 {key_tok.value!r} 缺少取值。")
            canonical_key = "temperature" if key_tok.value == "temp" else key_tok.value
            if key_tok.value not in _ALLOWED_FIELDS:
                allowed = "、".join(sorted(_ALLOWED_FIELDS))
                raise DSLCompileError(
                    f"未知字段 {key_tok.value!r}（允许字段：{allowed}）。",
                    source=self._source,
                    line=key_tok.line,
                    col=key_tok.col,
                )
            if canonical_key in fields:
                raise DSLCompileError(
                    f"模型 {name_tok.value!r} 中重复设置了字段 {key_tok.value!r}。",
                    source=self._source,
                    line=key_tok.line,
                    col=key_tok.col,
                )
            fields[canonical_key] = (val_tok.value, val_tok.line, val_tok.col)
            self._expect_line_end()
            self._skip_newlines()

        self._expect("RBRACE", "缺少右花括号 `}`。")
        self._expect_line_end()
        return ModelDecl(name=name_tok.value, fields=fields, line=name_tok.line, col=name_tok.col)


# 解析并编译极简模型配置 DSL 文本为标准 models.json 字典（支持一行预设与完整 model 块）
def compile_models_dsl(text: str, source: str = "models.dsl") -> dict[str, object]:
    tokens = tokenize_dsl(text, source=source)
    program = _DSLParser(tokens, source=source).parse()
    if not program.models:
        raise DSLCompileError("至少需要定义一个模型（例如：use deepseek sk-你的密钥）。", source=source, line=1, col=1)

    emitted_models: dict[str, dict[str, object]] = {}
    for decl in program.models:
        if not _MODEL_NAME_RE.fullmatch(decl.name):
            raise DSLCompileError(
                f"模型名 {decl.name!r} 仅允许字母、数字、下划线、点或连字符。",
                source=source,
                line=decl.line,
                col=decl.col,
            )
        if decl.name in emitted_models:
            raise DSLCompileError(
                f"模型名称重复：{decl.name!r}。",
                source=source,
                line=decl.line,
                col=decl.col,
            )

        if "base_url" not in decl.fields:
            raise DSLCompileError(f"模型 {decl.name!r} 缺少必填字段 base_url。", source=source, line=decl.line, col=decl.col)
        if "model" not in decl.fields:
            raise DSLCompileError(f"模型 {decl.name!r} 缺少必填字段 model。", source=source, line=decl.line, col=decl.col)

        base_url, url_line, url_col = decl.fields["base_url"]
        if not base_url.startswith(("http://", "https://")):
            raise DSLCompileError(
                f"base_url 必须以 http:// 或 https:// 开头：{base_url!r}",
                source=source,
                line=url_line,
                col=url_col,
            )

        model_id = decl.fields["model"][0]
        provider = decl.fields["provider"][0].lower() if "provider" in decl.fields else "openai_compatible"

        temperature = 0.2
        if "temperature" in decl.fields:
            raw_temp, t_line, t_col = decl.fields["temperature"]
            try:
                temperature = float(raw_temp)
            except ValueError as error:
                raise DSLCompileError(
                    f"temp/temperature 必须是数字：{raw_temp!r}",
                    source=source,
                    line=t_line,
                    col=t_col,
                ) from error
            if not (0.0 <= temperature <= 2.0):
                raise DSLCompileError(
                    f"temp/temperature 必须在 [0, 2] 之间：{temperature}",
                    source=source,
                    line=t_line,
                    col=t_col,
                )

        entry: dict[str, object] = {
            "provider": provider,
            "base_url": base_url,
            "model": model_id,
            "temperature": temperature,
        }
        if "api_key" in decl.fields:
            entry["api_key"] = decl.fields["api_key"][0]
        if "key_env" in decl.fields:
            entry["key_env"] = decl.fields["key_env"][0]
        emitted_models[decl.name] = entry

    if program.default_model is not None:
        default_name, def_line, def_col = program.default_model
        if default_name not in emitted_models:
            raise DSLCompileError(
                f"default 指向未定义的模型：{default_name!r}。",
                source=source,
                line=def_line,
                col=def_col,
            )
    elif len(emitted_models) == 1:
        default_name = next(iter(emitted_models))
    else:
        raise DSLCompileError(
            "定义了多个模型时请用 `use <厂商>` 或在顶部用 `default <名称>` 指定默认模型。",
            source=source,
            line=1,
            col=1,
        )

    return {
        "default": default_name,
        "models": emitted_models,
    }


# 实现内层 ConfigCompilerProtocol 协议的强类型配置编译器（支持 compile / presets / render）
class ModelsDSLCompiler:
    name = COMPILER_NAME

    # 编译 DSL 文本为标准 models.json 字典
    def compile(self, text: str, source: str = "models.dsl") -> dict[str, object]:
        return compile_models_dsl(text, source=source)

    # 返回内置模型厂商预设列表
    def presets(self) -> list[dict[str, object]]:
        return list_presets()

    # 按表单参数自动生成极简一行或块状 DSL 文本
    def render(self, **kwargs: object) -> str:
        return render_models_dsl(
            preset=str(kwargs.get("preset") or ""),
            api_key=str(kwargs.get("api_key") or ""),
            model=str(kwargs.get("model") or ""),
            base_url=str(kwargs.get("base_url") or ""),
            name=str(kwargs.get("name") or ""),
            provider=str(kwargs.get("provider") or ""),
            temperature=float(kwargs.get("temperature", 0.2)),  # type: ignore[arg-type]
        )


# 极简模型配置 DSL 主板插件（外层继承 Plugin ABC，内层向 compilers 插槽注册 ConfigCompilerProtocol 实现）
class ModelsDSLPlugin(Plugin):
    plugin_id = COMPILER_NAME

    # 统一初始化生命周期：向 self.context.register_compiler 插槽登记 ModelsDSLCompiler
    def initialize(self) -> None:
        self.context.register_compiler(ModelsDSLCompiler())
