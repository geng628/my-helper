# 极简模型配置示例（不懂代码可直接运行 `uv run python -m personal_assistant setup` 自动生成，或在网页端点击「模型设置」）
# 若手动编辑：复制为 config/models.dsl，只写下面一行即可，网址/协议/默认模型/温度全部自动补全！
# 系统已默认将 config/models.dsl 与 config/models.json 加入 .gitignore、0600 权限与 Agent 工具沙箱屏蔽。

use deepseek sk-替换为你的真实密钥

# 可选：用 add 追加更多备用模型（内置支持 deepseek / qwen(通义千问) / kimi / glm(智谱) / openai / siliconflow / ollama）
# add 通义千问 sk-你的通义千问密钥
# add ollama

# 学习助手推荐双模型分槽配置（前台用推理模型深度推导讲题，后台用低成本对话模型整理记忆笔记）：
# use reasoner sk-替换为你的真实密钥
# add deepseek sk-替换为你的真实密钥
# consolidator deepseek
#
# 可选：接入标准 /v1/embeddings 向量模型开启 RAG 语义+词面混合检索（支持硅基流动 / 通义 / OpenAI / 本地 Ollama）：
# add siliconflow BAAI/bge-m3 sk-你的硅基流动密钥
# embedder siliconflow
# 或使用本地免密钥 Ollama：
# add ollama bge-m3
# embedder ollama
#
# 可选：接入标准 /v1/rerank Cross-Encoder 重排模型开启粗排 Top-50 -> 精排 Top-K（支持硅基流动 BGE-Reranker / Cohere / Jina / vLLM）：
# model bge_rerank {
#   base_url https://api.siliconflow.cn/v1
#   model    BAAI/bge-reranker-v2-m3
#   api_key  sk-你的硅基流动密钥
# }
# reranker bge_rerank
