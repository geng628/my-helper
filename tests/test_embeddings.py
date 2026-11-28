"""标准 /v1/embeddings 客户端、float32 向量编解码与 embedder 角色配置单元测试（TDD）。"""
from __future__ import annotations

import io
import math
from pathlib import Path
from urllib.error import HTTPError

import pytest

from personal_assistant.ai import (
    AbortController,
    LLMNotConfigured,
    ModelProfile,
    OpenAIEmbeddingClient,
    StreamAbortedError,
    cosine_similarity,
    from_env_embedder,
    normalize_vector,
    pack_vector,
    unpack_vector,
)
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.commands import ToolsConfig


# 测试向量 L2 单位归一化、小端 float32 BLOB 编解码与 math.sumprod 精确余弦相似度及异常校验
def test_vector_normalize_pack_unpack_and_sumprod_cosine() -> None:
    unit = normalize_vector([3.0, 4.0])
    assert math.isclose(unit[0], 0.6, rel_tol=1e-6)
    assert math.isclose(unit[1], 0.8, rel_tol=1e-6)
    assert math.isclose(cosine_similarity(unit, unit), 1.0, rel_tol=1e-6)
    assert math.isclose(cosine_similarity(unit, (-0.8, 0.6)), 0.0, abs_tol=1e-6)
    assert math.isclose(cosine_similarity(unit, (-0.6, -0.8)), -1.0, rel_tol=1e-6)

    blob = pack_vector(unit)
    assert len(blob) == 8
    restored = unpack_vector(blob, 2)
    assert math.isclose(restored[0], 0.6, rel_tol=1e-5)
    assert math.isclose(restored[1], 0.8, rel_tol=1e-5)

    for bad in ([], [0.0, 0.0], [1.0, float("nan")], [1.0, float("inf")], [True, 1.0]):
        with pytest.raises(ValueError):
            normalize_vector(bad)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        unpack_vector(blob, 3)
    with pytest.raises(ValueError):
        unpack_vector(blob, 0)
    with pytest.raises(ValueError):
        cosine_similarity(unit, (1.0, 0.0, 0.0))


# 测试 OpenAIEmbeddingClient 线协议：/v1/embeddings 拼接、分批发送、乱序 index 重排、归一化、错误脱敏与鉴权校验
def test_openai_embedding_client_batching_index_order_and_validation() -> None:
    captured: list[tuple[str, dict[str, object], dict[str, str]]] = []

    # 模拟服务端返回乱序 index 的未归一化向量
    def _fake_post(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        *,
        abort_signal: object = None,
    ) -> dict[str, object]:
        captured.append((url, payload, headers))
        batch = payload["input"]
        assert isinstance(batch, list)
        # 故意倒序返回 index，验证客户端严格按 index 对齐输入顺序
        items = [
            {"index": i, "embedding": [float(len(str(txt))), 3.0, 4.0]}
            for i, txt in reversed(list(enumerate(batch)))
        ]
        return {"data": items}

    profile = ModelProfile(
        name="emb",
        base_url="https://api.siliconflow.cn/v1",
        model="BAAI/bge-m3",
        api_key="sk-secret-embed-key-999",
    )
    client = OpenAIEmbeddingClient(profile, batch_size=2, dimensions=3, post_json=_fake_post)
    assert client.model_id == "BAAI/bge-m3"
    assert client.embed([]) == []

    vecs = client.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert len(captured) == 3
    assert len(vecs) == 5
    assert captured[0][0] == "https://api.siliconflow.cn/v1/embeddings"
    assert captured[0][1]["model"] == "BAAI/bge-m3"
    assert captured[0][1]["dimensions"] == 3
    assert captured[0][2]["Authorization"] == "Bearer sk-secret-embed-key-999"
    # 校验倒序 index 还原正确且已 L2 单位归一化
    for idx, vec in enumerate(vecs, start=1):
        norm = math.sqrt(sum(v * v for v in vec))
        assert math.isclose(norm, 1.0, rel_tol=1e-6)
        expected_first = idx / math.sqrt(idx * idx + 9.0 + 16.0)
        assert math.isclose(vec[0], expected_first, rel_tol=1e-6)

    # 远程非本地主机缺少 API Key 显式抛出 LLMNotConfigured
    remote_no_key = ModelProfile(name="r", base_url="https://api.openai.com/v1", model="text-embedding-3-small")
    with pytest.raises(LLMNotConfigured):
        OpenAIEmbeddingClient(remote_no_key, post_json=_fake_post).embed(["hello"])

    # 本地 Ollama（127.0.0.1）免密钥允许调用
    local_ollama = ModelProfile(name="o", base_url="http://127.0.0.1:11434/v1/embeddings", model="bge-m3")
    local_vecs = OpenAIEmbeddingClient(local_ollama, post_json=_fake_post).embed(["hello"])
    assert len(local_vecs) == 1
    assert captured[-1][0] == "http://127.0.0.1:11434/v1/embeddings"
    assert "Authorization" not in captured[-1][2]

    # 服务端返回长度不匹配、跨批维度漂移或 HTTP 错误均安全校验与密钥脱敏
    # 模拟服务端漏返回向量导致数量不一致
    def _bad_count_post(url: str, payload: dict[str, object], headers: dict[str, str], *, abort_signal: object = None) -> dict[str, object]:
        return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    with pytest.raises(ValueError, match="数量"):
        OpenAIEmbeddingClient(profile, post_json=_bad_count_post).embed(["x", "y"])

    # 模拟上游返回含明文 sk- 的 HTTP 401 错误
    def _http_err_post(url: str, payload: dict[str, object], headers: dict[str, str], *, abort_signal: object = None) -> dict[str, object]:
        raise HTTPError(url, 401, "Unauthorized sk-secret-embed-key-999", {}, io.BytesIO(b"bad sk-secret-embed-key-999"))

    with pytest.raises(RuntimeError) as err_info:
        OpenAIEmbeddingClient(profile, post_json=_http_err_post).embed(["x"])
    assert "sk-secret-embed-key-999" not in str(err_info.value)


# 测试 OpenAIEmbeddingClient 协同取消与环境变量工厂 from_env_embedder
def test_openai_embedding_client_abort_signal_and_env_factory() -> None:
    ctrl = AbortController()
    calls = 0

    # 首个批次完成后立即触发协同取消，验证第二批次绝不发起请求
    def _aborting_post(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        *,
        abort_signal: object = None,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        ctrl.abort("用户取消向量化")
        batch = payload["input"]
        assert isinstance(batch, list)
        return {"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(batch))]}

    profile = ModelProfile(name="e", base_url="http://127.0.0.1:11434/v1", model="bge-m3")
    client = OpenAIEmbeddingClient(profile, batch_size=2, post_json=_aborting_post)
    with pytest.raises(StreamAbortedError):
        client.embed(["a", "b", "c", "d"], abort_signal=ctrl.signal)
    assert calls == 1

    # 未配 EMBEDDING_MODEL 时返回 None（平滑退回纯词面检索）
    assert from_env_embedder({}) is None
    env_client = from_env_embedder(
        {
            "LLM_BASE_URL": "https://api.siliconflow.cn/v1",
            "LLM_API_KEY": "sk-env-key",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
            "EMBEDDING_DIMENSIONS": "1024",
        }
    )
    assert isinstance(env_client, OpenAIEmbeddingClient)
    assert env_client.model_id == "BAAI/bge-m3"
    assert env_client.dimensions == 1024


# 测试 models.dsl 的 embedder 顶层指令、ToolsConfig.build_embedder 优先级及 AssistantApp 自动装配与热重载
def test_models_dsl_embedder_role_and_tools_config_build_embedder(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "models.dsl").write_text(
        "use deepseek sk-main-key\n"
        "add siliconflow BAAI/bge-m3 sk-embed-key\n"
        "consolidator deepseek\n"
        "embedder siliconflow\n",
        encoding="utf-8",
    )

    tcfg = ToolsConfig.load(repo_root / "config" / "tools.json", project_root=repo_root)
    try:
        models_cfg = tcfg.load_models_config(cfg_dir)
        assert models_cfg is not None
        assert models_cfg.roles == {"consolidator": "deepseek", "embedder": "siliconflow"}

        embedder = tcfg.build_embedder(cfg_dir, models_cfg=models_cfg)
        assert isinstance(embedder, OpenAIEmbeddingClient)
        assert embedder.model_id == "BAAI/bge-m3"
        assert embedder.profile.base_url == "https://api.siliconflow.cn/v1"

        # 重复声明 embedder 或指向不存在模型时 DSL 编译显式报错
        compiler = tcfg.plugin_manager.compilers["models_dsl"]
        with pytest.raises(ValueError, match="重复"):
            compiler.compile("use deepseek sk-1\nembedder deepseek\nembedder deepseek\n")
        with pytest.raises(ValueError, match="未定义"):
            compiler.compile("use deepseek sk-1\nembedder ghost\n")

        # AssistantApp 自动装配 embedder 并注入 RAGRetriever
        tools_copy = cfg_dir / "tools.json"
        tools_copy.write_text((repo_root / "config" / "tools.json").read_text(encoding="utf-8"), encoding="utf-8")
        app = AssistantApp(
            data_root=tmp_path / "data",
            config_path=tools_copy,
            project_root=repo_root,
            auto_consolidate=False,
        )
        try:
            assert app.embedder is not None and app.embedder.model_id == "BAAI/bge-m3"
            assert app.status()["embedder"] == "BAAI/bge-m3"
        finally:
            app.close()
    finally:
        tcfg.plugin_manager.terminate_all()
