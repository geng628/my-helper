"""极简模型配置 DSL 编译插件与系统级防泄漏保护单元测试（TDD）。"""
from __future__ import annotations

import json
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from personal_assistant.agent.plugins import PluginManager
from personal_assistant.ai import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ModelClient,
    ModelsConfig,
    StreamDone,
    StreamOptions,
    TokenUsage,
)
from personal_assistant.assistant.app import build_app
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.files_policy import FilesPolicy
from personal_assistant.assistant.memory import MemoryStore

REPO_ROOT = Path(__file__).resolve().parent.parent


# 测试插件发现器自动加载 plugins/models_dsl.py 中的 models_dsl 编译器并编译多模型配置（含 reasoning/max_tokens/cache_retention）
def test_models_dsl_plugin_compiles_plaintext_key_and_key_env() -> None:
    compilers = PluginManager.from_entries(["plugins"], REPO_ROOT).compilers
    assert "models_dsl" in compilers
    compiler = compilers["models_dsl"]

    dsl = """
    # 极简模型配置：统一使用单一路由键 api，直接填 api_key 即可用
    default deepseek

    model deepseek {
      api      openai-completions
      base_url https://api.deepseek.com/v1
      model    deepseek-chat
      api_key  sk-live-secret-9876543210
      temp     0.2
    }

    model claude {
      api             anthropic-messages
      base_url        https://api.anthropic.com
      model           claude-sonnet-4-5
      api_key         sk-ant-demo
      max_tokens      8192
      reasoning       medium
      cache_retention long
    }

    model local_qwen {
      base_url http://127.0.0.1:11434/v1
      model    qwen2.5:7b
      key_env  OLLAMA_API_KEY
      temp     0.1
    }
    """
    compiled = compiler.compile(dsl, "models.dsl")
    assert compiled["default"] == "deepseek"
    assert set(compiled["models"].keys()) == {"deepseek", "claude", "local_qwen"}  # type: ignore[union-attr]
    assert compiled["models"]["deepseek"]["api_key"] == "sk-live-secret-9876543210"  # type: ignore[index]
    assert compiled["models"]["deepseek"]["temperature"] == 0.2  # type: ignore[index]
    assert compiled["models"]["claude"]["api"] == "anthropic-messages"  # type: ignore[index]
    assert compiled["models"]["claude"]["max_tokens"] == 8192  # type: ignore[index]
    assert compiled["models"]["claude"]["reasoning"] == "medium"  # type: ignore[index]
    assert compiled["models"]["claude"]["cache_retention"] == "long"  # type: ignore[index]
    assert compiled["models"]["local_qwen"]["api"] == "openai-completions"  # type: ignore[index]
    assert compiled["models"]["local_qwen"]["key_env"] == "OLLAMA_API_KEY"  # type: ignore[index]

    # 单模型省略 default 时自动设为默认模型
    single = compiler.compile(
        "model solo {\n  base_url https://api.example.com/v1\n  model demo-1\n  api_key sk-demo\n}\n",
        "single.dsl",
    )
    assert single["default"] == "solo"

    # ModelsConfig 脱敏导出绝不暴露明文 api_key
    cfg = ModelsConfig.from_dict(compiled)
    redacted = cfg.redacted_dict()
    dumped = json.dumps(redacted, ensure_ascii=False)
    assert "sk-live-secret-9876543210" not in dumped
    assert "sk-ant-demo" not in dumped
    assert redacted["models"]["deepseek"]["api_key_configured"] is True  # type: ignore[index]
    assert redacted["models"]["claude"]["reasoning"] == "medium"  # type: ignore[index]


# 测试 DSL 编译器对语法错误、未知字段（含已废弃 provider）、重复定义、无效 URL/温度与悬空 default 给出精确行列报错
def test_models_dsl_reports_precise_syntax_and_semantic_errors() -> None:
    compiler = PluginManager.from_entries(["plugins"], REPO_ROOT).compilers["models_dsl"]

    # 未知字段（含已废弃的旧 provider 关键字）
    with pytest.raises(ValueError, match="第 3 行"):
        compiler.compile(
            "model a {\n  base_url https://x.com/v1\n  bogus 123\n  model m\n}\n",
            "bad.dsl",
        )
    with pytest.raises(ValueError, match="未知字段 'provider'"):
        compiler.compile(
            "model a {\n  provider openai-completions\n  base_url https://x.com/v1\n  model m\n}\n",
            "bad_old.dsl",
        )

    # default 指向不存在的模型
    with pytest.raises(ValueError, match="ghost"):
        compiler.compile(
            "default ghost\nmodel a {\n  base_url https://x.com/v1\n  model m\n}\n",
            "bad.dsl",
        )

    # 重复模型名
    with pytest.raises(ValueError, match="重复"):
        compiler.compile(
            "model a {\n  base_url https://x.com/v1\n  model m1\n}\n"
            "model a {\n  base_url https://x.com/v1\n  model m2\n}\n",
            "bad.dsl",
        )

    # 非法 base_url、越界 temperature、非法 reasoning / max_tokens / cache_retention
    with pytest.raises(ValueError, match="http"):
        compiler.compile("model a {\n  base_url ftp://x.com\n  model m\n}\n", "bad.dsl")
    with pytest.raises(ValueError, match="0"):
        compiler.compile("model a {\n  base_url https://x.com/v1\n  model m\n  temp 3.5\n}\n", "bad.dsl")
    with pytest.raises(ValueError, match="reasoning"):
        compiler.compile("model a {\n  base_url https://x.com/v1\n  model m\n  reasoning turbo\n}\n", "bad.dsl")
    with pytest.raises(ValueError, match="max_tokens"):
        compiler.compile("model a {\n  base_url https://x.com/v1\n  model m\n  max_tokens 0\n}\n", "bad.dsl")
    with pytest.raises(ValueError, match="cache_retention"):
        compiler.compile("model a {\n  base_url https://x.com/v1\n  model m\n  cache_retention forever\n}\n", "bad.dsl")

    # 缺少右花括号
    with pytest.raises(ValueError, match="}"):
        compiler.compile("model a {\n  base_url https://x.com/v1\n  model m\n", "bad.dsl")


# 测试直写明文密钥端到端零环境变量运行、落盘 JSON 自动 chmod 600 与回退 models.json
def test_tools_config_compiles_models_dsl_chmod_600_and_builds_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 清空环境变量，验证用户完全不配 .env 也能靠 models.dsl 直接运行
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    tools_json = cfg_dir / "tools.json"
    tools_json.write_text(
        json.dumps(
            {
                "enabled_tools": ["list_files"],
                "plugins": ["plugins"],
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    (cfg_dir / "models.dsl").write_text(
        "default fast\n"
        "model fast {\n"
        "  base_url https://api.deepseek.com/v1\n"
        "  model deepseek-chat\n"
        "  api_key sk-user-direct-key-12345\n"
        "  temp 0.3\n"
        "}\n"
        "model backup {\n"
        "  base_url http://127.0.0.1:11434/v1\n"
        "  model qwen2.5:7b\n"
        "  api_key local-ollama\n"
        "}\n",
        encoding="utf-8",
    )

    tcfg = ToolsConfig.load(tools_json, project_root=REPO_ROOT)
    models_cfg = tcfg.load_models_config(cfg_dir, write_json=True)
    assert models_cfg is not None
    assert models_cfg.default == "fast"

    # 落盘的 models.json 权限必须自动收紧为 0600（仅当前用户可读）
    out_json = cfg_dir / "models.json"
    assert out_json.is_file()
    mode = stat.S_IMODE(out_json.stat().st_mode)
    assert mode == 0o600

    # 按默认与显式 --model 名称构造客户端，零环境变量即可生效
    llm_default = tcfg.build_llm(cfg_dir)
    assert isinstance(llm_default, ModelClient)
    assert llm_default.profile.api == "openai-completions"
    assert llm_default.profile.api_key == "sk-user-direct-key-12345"
    assert llm_default.model == "deepseek-chat"
    assert llm_default.profile.temperature == 0.3

    llm_backup = tcfg.build_llm(cfg_dir, model_name="backup")
    assert isinstance(llm_backup, ModelClient)
    assert llm_backup.model == "qwen2.5:7b"

    # 删掉 models.dsl 后，核心不靠插件也能直接加载已编译的 models.json
    (cfg_dir / "models.dsl").unlink()
    llm_from_json = tcfg.build_llm(cfg_dir)
    assert isinstance(llm_from_json, ModelClient)
    assert llm_from_json.profile.api_key == "sk-user-direct-key-12345"

    # build_app 也能自动接管并避免 status() 暴露明文密钥
    app = build_app(data_root=tmp_path / "data", config_root=cfg_dir, project_root=REPO_ROOT)
    try:
        status = app.status()
        assert status["llm_configured"] is True
        assert "sk-user-direct-key-12345" not in json.dumps(status, ensure_ascii=False)
    finally:
        app.close()


# 测试系统级自动防泄漏：.gitignore 屏蔽 models.dsl/json，且文件工具与 Shell 插件默认拦截读取
def test_system_owned_leak_prevention_gitignore_and_tool_sandboxes(tmp_path: Path) -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "config/models.dsl" in gitignore
    assert "config/models.json" in gitignore
    assert (REPO_ROOT / "config" / "models.example.dsl").is_file()

    # MemoryStore 必须拒绝读取 models.dsl 与 models.json
    data_dir = tmp_path / "data"
    policy = FilesPolicy(
        project_root=tmp_path,
        read_roots=(tmp_path / "config", data_dir / "memory"),
        write_root=data_dir / "workspace",
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.dsl").write_text("api_key sk-secret\n", encoding="utf-8")
    (tmp_path / "config" / "models.json").write_text('{"api_key":"sk-secret"}', encoding="utf-8")
    mem = MemoryStore(data_dir, policy=policy)
    with pytest.raises(ValueError, match="禁止访问受保护路径"):
        mem.resolve_path("config/models.dsl", write=False)
    with pytest.raises(ValueError, match="禁止访问受保护路径"):
        mem.resolve_path("config/models.json", write=False)

    # Shell 插件必须拒绝 cat config/models.dsl 与 config/models.json
    tcfg = ToolsConfig.load(REPO_ROOT / "config" / "tools.json", project_root=REPO_ROOT)
    reg = tcfg.build_registry()
    res_dsl = reg.execute("run_command", '{"command": "cat config/models.dsl"}')
    res_json = reg.execute("run_command", '{"command": "cat config/models.json"}')
    assert "敏感" in res_dsl or "拒绝" in res_dsl or "禁止" in res_dsl
    assert "敏感" in res_json or "拒绝" in res_json or "禁止" in res_json


# 测试极简一行预设语法（含 anthropic/gemini/reasoner/中英文别名/免密钥 ollama）与自动生成配置
def test_models_dsl_one_line_preset_sugar_and_auto_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    compiler = PluginManager.from_entries(["plugins"], REPO_ROOT).compilers["models_dsl"]
    dsl = """
    # 非程序员极简写法：一行自动补全网址、默认模型、api 协议与 temperature
    use deepseek sk-ds-one-line
    add reasoner sk-r1-one-line
    add claude sk-ant-one-line
    add gemini sk-gem-one-line
    add 通义千问 sk-qwen-one-line
    add ollama
    add openai gpt-4o sk-oa-override
    """
    compiled = compiler.compile(dsl, "simple.dsl")
    models = compiled["models"]
    assert isinstance(models, dict)
    assert compiled["default"] == "deepseek"
    assert models["deepseek"]["api"] == "openai-completions"
    assert models["deepseek"]["base_url"] == "https://api.deepseek.com/v1"
    assert models["deepseek"]["model"] == "deepseek-chat"
    assert models["deepseek"]["api_key"] == "sk-ds-one-line"
    assert models["deepseek-reasoner"]["api"] == "openai-reasoning"
    assert models["deepseek-reasoner"]["base_url"] == "https://api.deepseek.com/v1"
    assert models["deepseek-reasoner"]["model"] == "deepseek-reasoner"
    assert models["deepseek-reasoner"]["api_key"] == "sk-r1-one-line"
    assert models["anthropic"]["api"] == "anthropic-messages"
    assert models["anthropic"]["base_url"] == "https://api.anthropic.com"
    assert models["anthropic"]["model"] == "claude-sonnet-4-5"
    assert models["anthropic"]["api_key"] == "sk-ant-one-line"
    assert models["gemini"]["api"] == "google-generative-ai"
    assert models["gemini"]["base_url"] == "https://generativelanguage.googleapis.com"
    assert models["gemini"]["model"] == "gemini-2.5-flash"
    assert models["gemini"]["api_key"] == "sk-gem-one-line"
    assert models["qwen"]["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert models["qwen"]["model"] == "qwen-plus"
    assert models["ollama"]["base_url"] == "http://127.0.0.1:11434/v1"
    assert models["ollama"]["api_key"] == "ollama"
    assert models["openai"]["model"] == "gpt-4o"
    assert models["openai"]["api_key"] == "sk-oa-override"

    # 自定义接口覆写 reasoner 预设或显式传入 api / reasoning / max_tokens / cache_retention 时均须准确保留并防注入
    custom_r1 = compiler.render(
        preset="reasoner",
        api_key="sk-r1-proxy",
        base_url="https://proxy.example.com/v1",
    )
    assert "api openai-reasoning" in custom_r1
    assert compiler.compile(custom_r1, "custom_r1.dsl")["models"]["deepseek-reasoner"]["api"] == "openai-reasoning"  # type: ignore[index]
    custom_api_dsl = compiler.render(
        preset="custom",
        api="anthropic-messages",
        api_key="sk-ant-custom",
        base_url="https://proxy.example.com",
        model="claude-sonnet-4-5",
        name="my_claude",
        reasoning="high",
        max_tokens=12000,
        cache_retention="long",
    )
    compiled_custom = compiler.compile(custom_api_dsl, "custom_api.dsl")
    assert compiled_custom["models"]["my_claude"]["api"] == "anthropic-messages"  # type: ignore[index]
    assert compiled_custom["models"]["my_claude"]["reasoning"] == "high"  # type: ignore[index]
    assert compiled_custom["models"]["my_claude"]["max_tokens"] == 12000  # type: ignore[index]
    assert compiled_custom["models"]["my_claude"]["cache_retention"] == "long"  # type: ignore[index]
    with pytest.raises(ValueError, match="换行"):
        compiler.render(preset="deepseek", api_key="sk-ok\nuse ollama")
    with pytest.raises(ValueError, match="模型名称"):
        compiler.render(preset="custom", api_key="sk-ok", base_url="https://x.com/v1", model="m", name="bad\nuse ollama")
    with pytest.raises(ValueError, match="模型名称"):
        compiler.render(preset="custom", api_key="sk-ok", base_url="https://x.com/v1", model="m", name="good\n")
    with pytest.raises(ValueError, match="模型名称"):
        compiler.render(preset="custom", api_key="sk-ok", base_url="https://x.com/v1", model="m", name="bad name!")

    # 测试自动生成向导底层 save_model_setup：自动生成 0600 权限的 models.dsl + models.json，并支持 app.reload_llm 热切换
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    tools_json = cfg_dir / "tools.json"
    tools_json.write_text(
        json.dumps(
            {
                "enabled_tools": ["list_files"],
                "plugins": ["plugins"],
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    app = build_app(data_root=tmp_path / "data", config_root=cfg_dir, project_root=REPO_ROOT)
    try:
        assert app.status()["llm_configured"] is False
        saved = app.configure_model(preset="deepseek", api_key="sk-auto-generated-key")
        assert saved["default"] == "deepseek"
        assert "sk-auto-generated-key" not in json.dumps(saved, ensure_ascii=False)
        assert app.status()["llm_configured"] is True
        assert isinstance(app.llm, ModelClient)
        assert app.llm.profile.api_key == "sk-auto-generated-key"
        assert app.consolidator.llm is app.llm
        assert stat.S_IMODE((cfg_dir / "models.dsl").stat().st_mode) == 0o600
        assert stat.S_IMODE((cfg_dir / "models.json").stat().st_mode) == 0o600
        assert "use deepseek" in (cfg_dir / "models.dsl").read_text(encoding="utf-8")
    finally:
        app.close()


# 测试终端一键向导 run_setup_wizard：纯小白选编号 + 粘贴密钥即可自动生成配置
def test_cli_setup_wizard_generates_config_for_non_programmers(tmp_path: Path) -> None:
    from personal_assistant.assistant.cli import run_setup_wizard

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "tools.json").write_text(
        json.dumps(
            {
                "enabled_tools": ["list_files"],
                "plugins": ["plugins"],
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    answers = iter(["1", "sk-wizard-key-999", ""])
    messages: list[str] = []
    cfg = run_setup_wizard(
        cfg_dir,
        project_root=REPO_ROOT,
        input_fn=lambda _: next(answers),
        secret_fn=lambda _: next(answers),
        print_fn=messages.append,
    )
    assert cfg is not None
    assert cfg.default == "deepseek"
    assert (cfg_dir / "models.dsl").is_file()
    assert (cfg_dir / "models.json").is_file()
    # 输出文案绝不回显明文密钥
    assert "sk-wizard-key-999" not in "\n".join(messages)


# 测试 ModelsConfig 与 models_dsl 编译/渲染对 consolidator 双模型分槽与别名解析的支持
def test_models_config_and_dsl_support_consolidator_slot() -> None:
    compiler = PluginManager.from_entries(["plugins"], REPO_ROOT).compilers["models_dsl"]

    # 1. DSL 顶层 consolidator 支持预设别名（如 ds -> deepseek）与显式 default 别名解析
    dsl = """
    use reasoner sk-r1-secret
    add deepseek sk-ds-secret
    consolidator ds
    """
    compiled = compiler.compile(dsl, "dual.dsl")
    assert compiled["default"] == "deepseek-reasoner"
    assert compiled["roles"] == {"consolidator": "deepseek"}

    cfg = ModelsConfig.from_dict(compiled)
    assert cfg.roles == {"consolidator": "deepseek"}
    redacted = cfg.redacted_dict()
    assert redacted["roles"] == {"consolidator": "deepseek"}
    assert "sk-r1-secret" not in json.dumps(redacted, ensure_ascii=False)
    assert "sk-ds-secret" not in json.dumps(redacted, ensure_ascii=False)

    # 显式 default + add 别名匹配
    compiled_alias_def = compiler.compile(
        "default reasoner\nconsolidator ds\nadd reasoner sk-r1\nadd deepseek sk-ds\n",
        "alias.dsl",
    )
    assert compiled_alias_def["default"] == "deepseek-reasoner"
    assert compiled_alias_def["roles"] == {"consolidator": "deepseek"}

    # 未配置命名角色时 ModelsConfig.roles 为空字典
    single_cfg = ModelsConfig.from_dict(compiler.compile("use deepseek sk-only", "single.dsl"))
    assert single_cfg.roles == {}
    assert single_cfg.redacted_dict()["roles"] == {}

    # 2. 非法校验：ModelsConfig 悬空 role、DSL 重复声明/悬空 consolidator 报精确错
    with pytest.raises(ValueError, match="ghost"):
        ModelsConfig.from_dict(
            {
                "default": "a",
                "roles": {"consolidator": "ghost"},
                "models": {"a": {"base_url": "https://x.com/v1", "model": "m", "api_key": "sk-1"}},
            }
        )
    with pytest.raises(ValueError, match="重复"):
        compiler.compile("use deepseek sk-1\nconsolidator deepseek\nconsolidator deepseek\n", "dup.dsl")
    with pytest.raises(ValueError, match="第 2 行"):
        compiler.compile("use deepseek sk-1\nconsolidator missing_slot\n", "dangling.dsl")

    # 3. render_models_dsl / compiler.render 支持可选 consolidator 并防注入校验
    rendered = compiler.render(preset="deepseek", api_key="sk-render", consolidator="deepseek")
    assert "consolidator deepseek" in rendered
    assert compiler.compile(rendered, "rendered.dsl")["roles"] == {"consolidator": "deepseek"}

    rendered_custom = compiler.render(
        preset="custom",
        name="r1",
        base_url="https://api.example.com/v1",
        model="r1-model",
        api_key="sk-c",
        consolidator="r1",
    )
    assert "consolidator r1" in rendered_custom
    assert compiler.compile(rendered_custom, "rendered_custom.dsl")["roles"] == {"consolidator": "r1"}

    with pytest.raises(ValueError, match="模型名称"):
        compiler.render(preset="deepseek", api_key="sk-1", consolidator="bad\nuse ollama")

    # 4. 示例配置文件包含推荐双模型分槽注释
    example_text = (REPO_ROOT / "config" / "models.example.dsl").read_text(encoding="utf-8")
    assert "use reasoner" in example_text
    assert "consolidator deepseek" in example_text


# 测试 ToolsConfig.build_consolidator_llm、AssistantApp 热重载与 CLI/Web --consolidator-model 优先级全链路
def test_tools_config_and_app_dual_model_consolidator_slot_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from personal_assistant.assistant import cli as cli_mod
    from personal_assistant.assistant.app import AssistantApp
    from scripts import web_chat as web_mod

    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_CONSOLIDATOR_MODEL", raising=False)

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    tools_json = cfg_dir / "tools.json"
    tools_json.write_text(
        json.dumps(
            {
                "enabled_tools": ["list_files"],
                "plugins": ["plugins"],
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    dsl_path = cfg_dir / "models.dsl"
    dsl_path.write_text(
        "use deepseek sk-main-1\n",
        encoding="utf-8",
    )

    tcfg = ToolsConfig.load(tools_json, project_root=REPO_ROOT)

    # 1. 未配置任何双模型槽位或环境变量时，必须原样复用主模型实例（app.consolidator.llm is app.llm）
    app_single = build_app(data_root=tmp_path / "d1", config_root=cfg_dir, project_root=REPO_ROOT)
    try:
        assert app_single.consolidator.llm is app_single.llm
        fallback = app_single.llm
        assert tcfg.build_consolidator_llm(cfg_dir, fallback_llm=fallback) is fallback
    finally:
        app_single.close()

    # 2. 存在单模型 models.dsl（未配置 consolidator）时，绝不被残留环境变量 LLM_CONSOLIDATOR_MODEL 炸崩
    dsl_path.write_text(
        "use reasoner sk-r1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_CONSOLIDATOR_MODEL", "nonexistent_env_slot")
    app_isolated = build_app(data_root=tmp_path / "d2a", config_root=cfg_dir, project_root=REPO_ROOT)
    try:
        assert isinstance(app_isolated.llm, ModelClient)
        assert app_isolated.llm.api == "openai-reasoning"
        assert app_isolated.consolidator.llm is app_isolated.llm
    finally:
        app_isolated.close()

    # 无 models.dsl / models.json 的纯环境变量模式下，LLM_CONSOLIDATOR_MODEL 正常生效
    env_only_cfg_dir = tmp_path / "env_only_config"
    env_only_cfg_dir.mkdir()
    (env_only_cfg_dir / "tools.json").write_text(tools_json.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("LLM_MODE", "on")
    monkeypatch.setenv("LLM_API_KEY", "sk-env-shared")
    monkeypatch.setenv("LLM_API", "openai-reasoning")
    monkeypatch.setenv("LLM_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("LLM_CONSOLIDATOR_MODEL", "qwen")
    app_env = build_app(data_root=tmp_path / "d2b", config_root=env_only_cfg_dir, project_root=REPO_ROOT)
    try:
        assert isinstance(app_env.llm, ModelClient)
        assert app_env.llm.api == "openai-reasoning"
        assert isinstance(app_env.consolidator.llm, ModelClient)
        assert app_env.consolidator.llm.api == "openai-completions"
        assert app_env.consolidator.llm.model == "qwen-plus"
    finally:
        app_env.close()
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_CONSOLIDATOR_MODEL", raising=False)

    # 3. models.dsl 显式声明 consolidator 优先级高于环境变量 LLM_CONSOLIDATOR_MODEL
    dsl_path.write_text(
        "use reasoner sk-r1\n"
        "add deepseek sk-ds\n"
        "add qwen sk-qwen\n"
        "consolidator deepseek\n",
        encoding="utf-8",
    )
    app_dsl = build_app(data_root=tmp_path / "d3", config_root=cfg_dir, project_root=REPO_ROOT)
    try:
        assert isinstance(app_dsl.llm, ModelClient)
        assert app_dsl.llm.model == "deepseek-reasoner"
        assert app_dsl.consolidator.llm is not app_dsl.llm
        assert isinstance(app_dsl.consolidator.llm, ModelClient)
        assert app_dsl.consolidator.llm.api == "openai-completions"
        assert app_dsl.consolidator.llm.model == "deepseek-chat"

        # 4. reload_llm 热重载时同步刷新主模型与后台沉淀模型，且失败时保持原运行实例零污染
        dsl_path.write_text(
            "use reasoner sk-r1\n"
            "add deepseek sk-ds\n"
            "add qwen sk-qwen\n"
            "consolidator qwen\n",
            encoding="utf-8",
        )
        app_dsl.reload_llm()
        assert isinstance(app_dsl.consolidator.llm, ModelClient)
        assert app_dsl.consolidator.llm.model == "qwen-plus"

        prev_llm = app_dsl.llm
        prev_cons_llm = app_dsl.consolidator.llm
        with pytest.raises(ValueError, match="missing_bad_cons"):
            app_dsl.reload_llm(model_name="deepseek", consolidator_model="missing_bad_cons")
        assert app_dsl.llm is prev_llm
        assert app_dsl.consolidator.llm is prev_cons_llm
        assert app_dsl._consolidator_model is None

        app_dsl.reload_llm(model_name="deepseek", consolidator_model="qwen")
        assert isinstance(app_dsl.llm, ModelClient) and app_dsl.llm.model == "deepseek-chat"
        assert isinstance(app_dsl.consolidator.llm, ModelClient) and app_dsl.consolidator.llm.model == "qwen-plus"

        # reload_llm(consolidator_model="") 可清空显式覆盖；configure_model() 切单模型时同样必须清空残留槽位
        app_dsl.reload_llm(consolidator_model="")
        assert app_dsl._consolidator_model is None
        app_dsl.reload_llm(consolidator_model="qwen")
        assert isinstance(app_dsl.consolidator.llm, ModelClient) and app_dsl.consolidator.llm.model == "qwen-plus"
        app_dsl.configure_model(preset="deepseek", api_key="sk-single-reset")
        assert app_dsl._consolidator_model is None
        assert app_dsl.consolidator.llm is app_dsl.llm
    finally:
        app_dsl.close()

    # 恢复 dsl_path 为双模型，继续测试显式参数覆盖与 Mock llm= 隔离
    dsl_path.write_text(
        "use reasoner sk-r1\n"
        "add deepseek sk-ds\n"
        "add qwen sk-qwen\n"
        "consolidator qwen\n",
        encoding="utf-8",
    )

    # 5. 显式传入 consolidator_model 参数优先级最高（高于 models.dsl 与环境变量）
    app_override = build_app(
        data_root=tmp_path / "d4",
        config_root=cfg_dir,
        project_root=REPO_ROOT,
        consolidator_model="deepseek",
    )
    try:
        assert isinstance(app_override.consolidator.llm, ModelClient)
        assert app_override.consolidator.llm.model == "deepseek-chat"
    finally:
        app_override.close()

    # 6. 显式注入 llm= Mock（且未传 consolidator_model）时，绝不穿透读取宿主 models.dsl 或 LLM_CONSOLIDATOR_MODEL
    class _FixedLLM:
        last_usage: TokenUsage | None = None

        # 流式返回固定应答
        def stream(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> Iterator[AssistantMessageEvent]:
            del context, options
            yield StreamDone(reason="stop", message=AssistantMessage.from_text("fixed", stop_reason="stop"))

        # 同步返回固定应答
        def complete(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessage:
            del context, options
            return AssistantMessage.from_text("fixed", stop_reason="stop")

    fixed_main = _FixedLLM()
    monkeypatch.setenv("LLM_CONSOLIDATOR_MODEL", "deepseek")
    app_mock_iso = AssistantApp(
        data_root=tmp_path / "d5a",
        config_path=tools_json,
        project_root=REPO_ROOT,
        llm=fixed_main,
    )
    try:
        assert app_mock_iso.llm is fixed_main
        assert app_mock_iso.consolidator.llm is fixed_main
    finally:
        app_mock_iso.close()
    monkeypatch.delenv("LLM_CONSOLIDATOR_MODEL", raising=False)

    # 显式注入 consolidator_llm 对象时，reload_llm 绝不覆写自定义沉淀模型
    fixed_cons = _FixedLLM()
    app_fixed = AssistantApp(
        data_root=tmp_path / "d5b",
        config_path=tools_json,
        project_root=REPO_ROOT,
        consolidator_llm=fixed_cons,
    )
    try:
        app_fixed.reload_llm()
        assert app_fixed.consolidator.llm is fixed_cons
    finally:
        app_fixed.close()

    # 7. 无 tools_config 时仍支持通过 LLM_CONSOLIDATOR_MODEL 环境变量装配后台沉淀模型
    monkeypatch.setenv("LLM_MODE", "on")
    monkeypatch.setenv("LLM_API_KEY", "sk-env-only")
    monkeypatch.setenv("LLM_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("LLM_CONSOLIDATOR_MODEL", "deepseek-chat")
    app_no_cfg = AssistantApp(data_root=tmp_path / "d6")
    try:
        assert isinstance(app_no_cfg.llm, ModelClient) and app_no_cfg.llm.model == "deepseek-reasoner"
        assert app_no_cfg.consolidator.llm is not app_no_cfg.llm
        assert isinstance(app_no_cfg.consolidator.llm, ModelClient) and app_no_cfg.consolidator.llm.model == "deepseek-chat"
    finally:
        app_no_cfg.close()

    # 8. CLI 与 Web 入口 --consolidator-model 参数正确透传给 build_app
    captured_cli: dict[str, object] = {}

    # 捕获 CLI 调用 build_app 的参数
    def fake_cli_build_app(**kwargs):
        captured_cli.update(kwargs)
        return app_override

    monkeypatch.setattr(cli_mod, "build_app", fake_cli_build_app)
    monkeypatch.setattr(cli_mod, "run_repl", lambda _app: None)
    cli_mod.main(["--config", str(cfg_dir), "--model", "reasoner", "--consolidator-model", "deepseek", "chat"])
    assert captured_cli["model_name"] == "reasoner"
    assert captured_cli["consolidator_model"] == "deepseek"

    captured_web: dict[str, object] = {}

    # 构造空转 Web 服务桩，避免真实阻塞监听
    class _DummyServer:
        server_port = 8765

        # 空转退出
        def serve_forever(self):
            return None

        # 空关闭
        def server_close(self):
            return None

    # 捕获 Web 启动调用 build_app 的参数
    def fake_web_build_app(**kwargs):
        captured_web.update(kwargs)

        # 仅含 close 空方法的轻量应用替身
        class _DummyApp:
            # 清理占位
            def close(self):
                return None

        return _DummyApp()

    monkeypatch.setattr(web_mod, "build_app", fake_web_build_app)
    monkeypatch.setattr(
        web_mod,
        "create_web_server",
        lambda _app, host="127.0.0.1", port=8765, *, auth_token=None: _DummyServer(),
    )
    web_mod.main(["--model", "reasoner", "--consolidator-model", "qwen"])
    assert captured_web["model_name"] == "reasoner"
    assert captured_web["consolidator_model"] == "qwen"

    # 9. 确认 ConfigCompilerProtocol.render 保持零业务名泄漏且统一使用 api，ModelsDSLCompiler.render 与底层对齐
    import inspect
    from personal_assistant.agent.plugins import ConfigCompilerProtocol
    from plugins.models_dsl import ModelsDSLCompiler, render_models_dsl

    proto_sig = inspect.signature(ConfigCompilerProtocol.render)
    impl_sig = inspect.signature(ModelsDSLCompiler.render)
    proto_params = list(proto_sig.parameters)[1:]
    impl_params = list(impl_sig.parameters)[1:]
    fn_params = list(inspect.signature(render_models_dsl).parameters)
    assert "api" in proto_params
    assert "provider" not in proto_params
    assert "consolidator" not in proto_params
    assert impl_params[: len(proto_params)] == proto_params
    assert impl_params == fn_params
    for extra_name in impl_params[len(proto_params) :]:
        assert impl_sig.parameters[extra_name].default is not inspect.Parameter.empty
