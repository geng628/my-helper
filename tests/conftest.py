"""pytest 全局测试夹具：隔离宿主机环境变量、仓库根目录 .env / models.dsl 与全局默认协议注册表。"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from personal_assistant.ai import default_registry
from personal_assistant.assistant import commands as commands_mod
from personal_assistant.assistant import common as common_mod

_HOST_ENV_VARS = (
    "LLM_MODE",
    "LLM_API",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_REASONING",
    "LLM_CACHE_RETENTION",
    "LLM_CONSOLIDATOR_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "MY_HELPER_DATA",
    "MY_HELPER_CONFIG",
    "MY_HELPER_WEB_TOKEN",
)

_ORIG_LOAD_DOTENV = common_mod.load_dotenv
_ORIG_LOAD_MODELS_CONFIG = commands_mod.ToolsConfig.load_models_config
_REAL_REPO_ROOT = common_mod.REPO_ROOT.resolve()
_REPO_DOTENV = (_REAL_REPO_ROOT / ".env").resolve()
_REPO_CONFIG_DIR = (_REAL_REPO_ROOT / "config").resolve()


# 统一在每个用例前后隔离宿主机环境变量、仓库根目录 .env / models.dsl 与全局默认 ApiProviderRegistry
@pytest.fixture(autouse=True)
def isolate_host_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    reg = default_registry()
    saved_providers = reg.snapshot()

    for var in _HOST_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    # 跳过仓库根目录真实 .env 防止宿主配置污染测试；测试临时目录中的 .env 正常执行原加载逻辑
    def _guarded_load_dotenv(dotenv_path: Path | None = None) -> None:
        target = (dotenv_path if dotenv_path is not None else (common_mod.REPO_ROOT / ".env")).resolve()
        if target == _REPO_DOTENV:
            return
        _ORIG_LOAD_DOTENV(target)

    monkeypatch.setattr(common_mod, "load_dotenv", _guarded_load_dotenv)
    for mod in list(sys.modules.values()):
        if mod is None or mod is common_mod:
            continue
        mod_name = getattr(mod, "__name__", "")
        if not mod_name.startswith(("personal_assistant.", "evals.", "scripts.")):
            continue
        current_fn = getattr(mod, "load_dotenv", None)
        if current_fn is _ORIG_LOAD_DOTENV or getattr(current_fn, "__name__", "") == "_guarded_load_dotenv":
            monkeypatch.setattr(mod, "load_dotenv", _guarded_load_dotenv)

    # 拦截读取仓库真实 config/models.dsl 或 models.json；测试构造的临时配置目录正常放行
    def _guarded_load_models_config(
        self: commands_mod.ToolsConfig,
        config_dir: Path | None = None,
        *,
        write_json: bool = False,
    ) -> object:
        resolved_dir = (
            Path(config_dir).resolve()
            if config_dir is not None
            else (self.config_path.parent.resolve() if self.config_path is not None else None)
        )
        if resolved_dir is None or resolved_dir == _REPO_CONFIG_DIR:
            return None
        return _ORIG_LOAD_MODELS_CONFIG(self, resolved_dir, write_json=write_json)

    monkeypatch.setattr(commands_mod.ToolsConfig, "load_models_config", _guarded_load_models_config)

    try:
        yield
    finally:
        for var in _HOST_ENV_VARS:
            os.environ.pop(var, None)
        reg.restore(saved_providers)
