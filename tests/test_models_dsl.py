"""极简模型配置 DSL 编译插件与系统级防泄漏保护单元测试（TDD）。"""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from personal_assistant.agent.plugins import PluginManager
from personal_assistant.ai import (
    LLMClient,
    ModelProfile,
    ModelsConfig,
    OpenAICompatibleLLM,
    from_models_dict,
)
from personal_assistant.assistant.app import build_app
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.memory import FilesPolicy, MemoryStore

REPO_ROOT = Path(__file__).resolve().parent.parent


# 测试插件发现器自动加载 plugins/models_dsl.py 中的 models_dsl 编译器并编译多模型配置
def test_models_dsl_plugin_compiles_plaintext_key_and_key_env() -> None:
    compilers = PluginManager.from_entries(["plugins"], REPO_ROOT).compilers
    assert "models_dsl" in compilers
    compiler = compilers["models_dsl"]

    dsl = """
    # 极简模型配置：直接填 api_key 即可用，无需用户折腾环境变量
    default deepseek

    model deepseek {
      provider openai_compatible
      base_url https://api.deepseek.com/v1
      model    deepseek-chat
      api_key  sk-live-secret-9876543210
      temp     0.2
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
    assert set(compiled["models"].keys()) == {"deepseek", "local_qwen"}
    assert compiled["models"]["deepseek"]["api_key"] == "sk-live-secret-9876543210"
    assert compiled["models"]["deepseek"]["temperature"] == 0.2
    assert compiled["models"]["local_qwen"]["provider"] == "openai_compatible"
    assert compiled["models"]["local_qwen"]["key_env"] == "OLLAMA_API_KEY"

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
    assert redacted["models"]["deepseek"]["api_key_configured"] is True


# 测试 DSL 编译器对语法错误、未知字段、重复定义、无效 URL/温度与悬空 default 给出精确行列报错
def test_models_dsl_reports_precise_syntax_and_semantic_errors() -> None:
    compiler = PluginManager.from_entries(["plugins"], REPO_ROOT).compilers["models_dsl"]

    # 未知字段
    with pytest.raises(ValueError, match="第 3 行"):
        compiler.compile(
            "model a {\n  base_url https://x.com/v1\n  bogus 123\n  model m\n}\n",
            "bad.dsl",
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

    # 非法 base_url 与越界 temperature
    with pytest.raises(ValueError, match="http"):
        compiler.compile("model a {\n  base_url ftp://x.com\n  model m\n}\n", "bad.dsl")
    with pytest.raises(ValueError, match="0"):
        compiler.compile("model a {\n  base_url https://x.com/v1\n  model m\n  temp 3.5\n}\n", "bad.dsl")

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
    assert isinstance(llm_default, OpenAICompatibleLLM)
    assert llm_default.api_key == "sk-user-direct-key-12345"
    assert llm_default.model == "deepseek-chat"
    assert llm_default.temperature == 0.3

    llm_backup = tcfg.build_llm(cfg_dir, model_name="backup")
    assert isinstance(llm_backup, OpenAICompatibleLLM)
    assert llm_backup.model == "qwen2.5:7b"

    # 删掉 models.dsl 后，核心不靠插件也能直接加载已编译的 models.json
    (cfg_dir / "models.dsl").unlink()
    llm_from_json = tcfg.build_llm(cfg_dir)
    assert isinstance(llm_from_json, OpenAICompatibleLLM)
    assert llm_from_json.api_key == "sk-user-direct-key-12345"

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


# 测试极简一行预设语法（免写大括号、网址、provider、temp），支持中英文厂商别名与本地免密钥 Ollama
def test_models_dsl_one_line_preset_sugar_and_auto_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    compiler = PluginManager.from_entries(["plugins"], REPO_ROOT).compilers["models_dsl"]
    dsl = """
    # 非程序员极简写法：一行自动补全网址、默认模型、provider 与 temperature
    use deepseek sk-ds-one-line
    add reasoner sk-r1-one-line
    add 通义千问 sk-qwen-one-line
    add ollama
    add openai gpt-4o sk-oa-override
    """
    compiled = compiler.compile(dsl, "simple.dsl")
    assert compiled["default"] == "deepseek"
    assert compiled["models"]["deepseek"]["provider"] == "openai_compatible"
    assert compiled["models"]["deepseek"]["base_url"] == "https://api.deepseek.com/v1"
    assert compiled["models"]["deepseek"]["model"] == "deepseek-chat"
    assert compiled["models"]["deepseek"]["api_key"] == "sk-ds-one-line"
    assert compiled["models"]["deepseek_reasoner"]["provider"] == "openai_reasoning"
    assert compiled["models"]["deepseek_reasoner"]["base_url"] == "https://api.deepseek.com/v1"
    assert compiled["models"]["deepseek_reasoner"]["model"] == "deepseek-reasoner"
    assert compiled["models"]["deepseek_reasoner"]["api_key"] == "sk-r1-one-line"
    assert compiled["models"]["qwen"]["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert compiled["models"]["qwen"]["model"] == "qwen-plus"
    assert compiled["models"]["ollama"]["base_url"] == "http://127.0.0.1:11434/v1"
    assert compiled["models"]["ollama"]["api_key"] == "ollama"
    assert compiled["models"]["openai"]["model"] == "gpt-4o"
    assert compiled["models"]["openai"]["api_key"] == "sk-oa-override"

    # 自定义接口覆写 reasoner 预设时，生成的 model 块必须保留 provider openai_reasoning 且拒绝换行注入
    custom_r1 = compiler.render(
        preset="reasoner",
        api_key="sk-r1-proxy",
        base_url="https://proxy.example.com/v1",
    )
    assert "provider openai_reasoning" in custom_r1
    assert compiler.compile(custom_r1, "custom_r1.dsl")["models"]["deepseek_reasoner"]["provider"] == "openai_reasoning"
    with pytest.raises(ValueError, match="换行"):
        compiler.render(preset="deepseek", api_key="sk-ok\nuse ollama")
    with pytest.raises(ValueError, match="换行"):
        compiler.render(preset="custom", api_key="sk-ok", base_url="https://x.com/v1", model="m", name="bad\nuse ollama")
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
        assert isinstance(app.llm, OpenAICompatibleLLM)
        assert app.llm.api_key == "sk-auto-generated-key"
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
