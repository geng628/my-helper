"""极简模型配置 DSL 外部编译插件（零第三方依赖）：词法分析 -> 递归下降 AST -> 语义校验 -> 标准 JSON。"""
from __future__ import annotations

import re
from dataclasses import dataclass

COMPILER_NAME = "models_dsl"

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
_ALLOWED_FIELDS = frozenset({"provider", "base_url", "model", "api_key", "key_env", "temp", "temperature"})


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


# 递归下降语法分析器：将 Token 流解析为结构化 ModelsDSLProgram AST
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

    # 解析整篇 DSL 文件，产出 ModelsDSLProgram AST
    def parse(self) -> ModelsDSLProgram:
        default_decl: tuple[str, int, int] | None = None
        models: list[ModelDecl] = []

        self._skip_newlines()
        while self._peek().kind != "EOF":
            tok = self._peek()
            if tok.kind != "WORD":
                raise DSLCompileError(
                    f"顶层期望 default 或 model 声明，实际遇到 {tok.value!r}。",
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
            elif tok.value == "model":
                models.append(self._parse_model_block())
            else:
                raise DSLCompileError(
                    f"未知顶层指令 {tok.value!r}（仅支持 default 与 model）。",
                    source=self._source,
                    line=tok.line,
                    col=tok.col,
                )
            self._skip_newlines()

        return ModelsDSLProgram(default_model=default_decl, models=tuple(models))

    # 解析单个 `model <name> { ... }` 代码块
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


# 解析并编译极简模型配置 DSL 文本为标准 models.json 字典（支持直接写 api_key 或 key_env）
def compile_models_dsl(text: str, source: str = "models.dsl") -> dict[str, object]:
    tokens = tokenize_dsl(text, source=source)
    program = _DSLParser(tokens, source=source).parse()
    if not program.models:
        raise DSLCompileError("至少需要定义一个 model 代码块。", source=source, line=1, col=1)

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
            "定义了多个模型时必须在文件顶部通过 `default <名称>` 指定默认模型。",
            source=source,
            line=1,
            col=1,
        )

    return {
        "default": default_name,
        "models": emitted_models,
    }


CONFIG_COMPILERS = {
    COMPILER_NAME: compile_models_dsl,
}
