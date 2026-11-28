# My Helper v2

一个零第三方运行依赖的纯 Python 个人与学习 AI Agent。项目以借鉴 Pi 洋葱模型的三层单向依赖微内核（`ai ← agent ← assistant`）为底座，并在 Layer 1 `ai/` 内部完整落地 **Pi ch04 三层模型调用翻译架构**（统一前台 ← 12 事件中立协议 ← 多厂商原生翻译器），在外层通过插件协议叠加纯 Markdown 长期记忆生命周期、可折叠流式推导视图、间隔重复学习卡片与前后台双模型分槽。生产代码仅使用 Python 3.12+ 标准库，`pytest` 仅作开发测试依赖。

> 系统设计与工程权衡详见 [docs/mvp-design.md](docs/mvp-design.md)；当前架构基线、踩坑根因、去泄漏评测总账与迭代记录详见 [docs/dev-log.md](docs/dev-log.md) 与 [AGENTS.md](AGENTS.md)。

---

## 架构概览

1. **三层单向依赖漏斗（`ai ← agent ← assistant`）与 Layer 1 三层模型翻译架构**：
   - **Layer 1 `personal_assistant/ai/`**（零向上依赖，内部遵循 Pi ch04 三层解耦）：
     - **第二层·事件与契约协议层**（[personal_assistant/ai/protocol.py](personal_assistant/ai/protocol.py)）：零内部依赖，定义厂商中立的 `Context` / `Message`（`UserMessage`、`AssistantMessage`、`ToolResultMessage`）/ `ToolSchema` / `ToolCall`、12 种统一流式事件（`StreamStart`、`TextStart/Delta/End`、`ThinkingStart/Delta/End`、`ToolCallStart/Delta/End`、`StreamDone`、`StreamError`）、协同取消原语（只读 `AbortSignal`、触发端 `AbortController` 与 `StreamAbortedError`）、语义思考级别 `ThinkingLevel` 与缓存保留策略 `CacheRetention`、单一路由键 `api` 的 `ModelProfile`、通用命名角色表 `ModelsConfig.roles`、翻译器入职宪法 `ApiProvider` 与前台句柄协议 `ModelInvoker`。
     - **第一层·统一前台与通讯录**（[personal_assistant/ai/registry.py](personal_assistant/ai/registry.py)）：仅依赖 `protocol.py` 零感知具体翻译器，提供 `stream()` / `stream_simple()` / `complete()` 统一入口、线程安全可隔离拷贝的 `ApiProviderRegistry` 通讯录、`ModelClient` / `NoopModelClient` 与直通工厂 `from_model_profile()` / `from_env()`。
     - **第三层·原生协议翻译器**（[personal_assistant/ai/providers/](personal_assistant/ai/providers/)）：共享标准库可中断连接/重试/脱敏/错误收口与通用 SSE 底座 `http.py`（依赖 `protocol.py`，支持 `AbortSignal` 毫秒级打断 TTFB/SSE 套接字与退避睡眠），内置四大原生翻译器——`openai_completions.py`（`openai-completions`，覆盖 OpenAI / DeepSeek Chat / 通义千问 / Kimi / GLM / Ollama 并兼容 `prompt_cache_hit_tokens`）、`openai_reasoning.py`（`openai-reasoning`，覆盖 DeepSeek Reasoner / R1 等 `reasoning_content` 推理方言，支持轮内续传与跨轮剥离）、`anthropic_messages.py`（`anthropic-messages`，支持 Prompt Caching 断点与带签名 Extended Thinking）、`google_genai.py`（`google-generative-ai`，支持 `systemInstruction`/`functionDeclarations`/`thinkingConfig`）；接入任意全新模型厂商或私有协议只需编写 **1 个 `ApiProvider` 翻译器 + 注册到通讯录（或 Slot 4 插件）+ 在 `models.dsl` 配 `api`**，上层零改动。
     - **中立 KV Cache 观测与标准向量客户端**（[personal_assistant/ai/cache.py](personal_assistant/ai/cache.py)、[personal_assistant/ai/embeddings.py](personal_assistant/ai/embeddings.py)）：基于中立 `Context` 序列化统计服务端真实命中率与失配段归因；并内置零第三方依赖的标准 `/v1/embeddings` 客户端（`OpenAIEmbeddingClient` / `EmbeddingInvoker`，兼容 OpenAI / 硅基流动 / 通义千问 / 本地 Ollama，支持分批、`index` 对齐、L2 归一化、小端 `float32` BLOB 编解码与 Python 3.12 `math.sumprod` 精确余弦）。
   - **Layer 2 `personal_assistant/agent/`**：仅依赖 `ai/`，彻底清除 OpenAI 字典渗透（无 `to_openai_schema` / `to_openai_dict` / `extract_tool_calls`）；包含中立 `Context` 驱动的单层 ReAct 循环、纯值对象递进扩展（`ToolSchema → Tool` + `bind_tool_context`，`ToolRegistry.schemas()` 仅导出中立签名）与零业务字段 `PluginContext` 统一插件主板（Slot 4 直接复刻 Layer 1 `ApiProvider` 协议）。
   - **Layer 3 `personal_assistant/assistant/`**：个人与学习助手业务层（零内部依赖底座 `common.py`/`text_search.py`/`files_policy.py`、Markdown 记忆与内置 `LinearRetriever` 兜底、SQLite 对话历史、外层回合预算、后台压缩沉淀与 CLI/Web 接入），全程面向中立 `Context` / `Message` / `ModelInvoker` 编排。
   - 顶层评测包 `evals/` 与 `plugins/`、`tests/` 平级；[tests/test_architecture.py](tests/test_architecture.py) 通过 AST 依赖图（含 `ai/` 内部三层解耦边界）、业务字面量与零 OpenAI 方言符号扫描、子进程 `sys.modules` 隔离测试与四棵目录树（`personal_assistant/`、`plugins/`、`scripts/`、`evals/`）零 `# type: ignore` 静态硬守卫锁死架构纯度。
2. **极简单层 ReAct 内核 + 并行只读工具 + 外层安全预算**：
   - 面向个人伴读与学习辅导场景，[personal_assistant/agent/react.py](personal_assistant/agent/react.py) 保持为极简单层 `while True` 循环（`iter_react()` / `run_react()`），仅以 `done.reason != "toolUse" or not assistant_msg.tool_calls` 作为自然终止信号，遇 `StreamError` 透传退出（`run_react()` 对取消抛 `StreamAbortedError`、其余抛 `RuntimeError`），并支持单步 `convert_to_llm: ContextTransformer` 非破坏性上下文投影、执行期 `tool_context` 注入、`StreamOptions`（含 `abort_signal` 全链路协同取消与 `ChainMap` 只读透传至慢 I/O 工具）全链路透传，以及基于 `ToolMetadata.parallel`（默认 `False` 安全串行）的**同轮只读工具分批并发 + 写/执行工具串行路障 + 按模型发射原序提交**（默认上限 `max_parallel_tools=4`，守护 KV Cache 前缀确定性）。
   - 防死循环、成本控制与运行时取消由 Layer 3 `AssistantApp` 统一实施：`_GuardedTurnLLM` 实施默认单轮上限 `24` 步、`250,000` Token、`180s` 墙钟、连续 `3` 次等价工具调用熔断及 `StreamDone.reason == "length"` 截断熔断（取消态优先）；`AssistantApp.abort_turn(chat_id)` 联动 CLI `SIGINT` 与 Web `POST /api/chat/abort`「停止」按钮，毫秒级打断在途 LLM 套接字、子进程组与联网抓取，自动回滚 `(turn#)` 暂存事实、零写入 SQLite 并即刻释放会话锁。
3. **学习助手叠加能力（零微内核侵入）**：
   - **流式推导视图与跨协议热切换**：通过 12 种规范流事件（含 `ThinkingDelta`、`TextDelta`）与 `ToolStart` 事件，在 CLI 与本地网页端实时展示可折叠的推理过程与慢工具执行状态；同一条中立对话轨迹支持跨 `anthropic-messages` / `openai-completions` / `google-generative-ai` / `openai-reasoning` 无缝热切换，落库历史始终仅含纯净问答正文。
   - **前后台双模型分槽**：支持在 `config/models.dsl` 中配置 `consolidator <模型名>`（编译至通用角色映射 `ModelsConfig.roles["consolidator"]`），前台使用推理或旗舰模型（如 `deepseek-reasoner`、`anthropic`、`gemini`）讲题推导，后台使用轻量对话模型（如 `deepseek-chat`）执行结构化记忆沉淀。
   - **纯 Markdown 间隔复习卡片**：通过外围插件 [plugins/tools/study_tools.py](plugins/tools/study_tools.py) 提供 `save_study_card` 与 `list_due_reviews`，在 `data/workspace/study/` 下维护人类可读的复习卡片与调度计划。

---

## 核心工程特性

### 1. 五大插件主板插槽与单次初始化生命周期
- 外层基类 `Plugin(ABC)` 在启动期执行一次 `initialize()` 向零业务字段的 `PluginContext` 注册五类强类型子协议（业务配置与运行时绑定统一由 Layer 3 `_build_plugin_bindings()` 注入 `context.bindings`），关闭时经 `AssistantApp.close()` 触发 `terminate_all()`：
  1. **工具插槽 `Tool(ToolSchema)`**：[plugins/tools/plugin.py](plugins/tools/plugin.py) 自动扫描 `plugins/tools/*.py` 命名空间子包中的 `@tool` 函数，受 `config/tools.json` 白名单管控；`shell_tools.py`/`web_tools.py`/`study_tools.py` 单轨消费主板注入的配置与 `data_root`，无旧配置文件回退或隐式 CWD 落盘。
  2. **检索插槽 `RetrieverProtocol`**：[plugins/rag_retriever.py](plugins/rag_retriever.py)（`rag`）提供两层父子结构化分块（[personal_assistant/assistant/chunker.py](personal_assistant/assistant/chunker.py)：中英文句界切分、超长表格续片保表头与可配整句 Overlap）、基于同文档 Prompt Caching 前缀的离线上下文摘要器（[personal_assistant/assistant/rag_contextualizer.py](personal_assistant/assistant/rag_contextualizer.py)）、独立持久化派生库 `data/state/rag_index.db`（`parent_chunks` + `child_chunks` + `child_chunks_fts` + 跨重建保留的 `chunk_contexts` 与 `chunk_embeddings`）、基于 Python 3.12 `math.sumprod` 的精确余弦向量召回与 **BM25 + Dense 加权 RRF 混合检索**（CLI `/rag status`、`/rag enrich`、`/rag embed`；未配置 `embedder` 时自动零开销退回纯词面 BM25），未启用插件时由 `MemoryStore` 内置零索引 `LinearRetriever` 自动兜底；与 `history_tools.py` 正交共享零内部依赖底座 [personal_assistant/assistant/text_search.py](personal_assistant/assistant/text_search.py) 的双语分词与检索阈值校验。
  3. **配置编译插槽 `ConfigCompilerProtocol`**：[plugins/models_dsl.py](plugins/models_dsl.py)，手写词法与递归下降解析器，将精简 DSL（支持 `api`、`reasoning`、`max_tokens`、`cache_retention` 及 `anthropic`/`gemini`/`deepseek-reasoner` 等内置预设）编译为含通用 `roles` 映射的 `ModelsConfig` 标准字典，`ConfigCompilerProtocol.render` 与 `ModelsDSLCompiler.render` 均采用显式强类型具名参数（插件按需以带默认值参数扩展 `consolidator`/`reasoning`/`max_tokens`/`cache_retention`，不向 Layer 2 泄漏业务角色名）。
  4. **协议翻译器插槽 `ApiProvider`**：直接复用 Layer 1 `ApiProvider` 宪法，支持外部/私有协议插件通过 `PluginContext.register_provider()` 零核心改动接入新厂商（`ToolsConfig.build_provider_registry()` 将插件翻译器与内建四大翻译器合并为应用级隔离副本，不污染进程全局注册表）。
  5. **运行观测插槽 `ObserverProtocol`**：[plugins/telemetry.py](plugins/telemetry.py)，以显式字段累加与有界 MRU 会话账本（`max_sessions`）统计分阶段耗时（`total_ms` / `ttft_ms` / `llm_ms` / `tool_ms`）与后台沉淀事件，遇 `StreamError` 或 `turn_aborted` 自动清理在途状态防止跨轮污染。
- `AssistantApp` 在启动期构建一次常驻 `_tool_registry`，每轮通过 `bind_tool_context` / `tool_context` 注入动态回合上下文；`build_tool()` 自动从 Python 默认参数推断精简 `ToolSchema.required`。

### 2. 全链路 KV Cache 友好的 Markdown 记忆、按块裁剪与非破坏性投影
- **动静分离与空闲 TTL 刷新**：由独立上下文装配引擎 [personal_assistant/assistant/context.py](personal_assistant/assistant/context.py)（`ContextAssembler`）统一收口：`system_prompt` 仅包含静态指令、白名单工具、规则摘要与 `MEMORY.md` 索引（前 100 行 / 4000 字符），不含逐日日期与滚动前情摘要；当前逻辑日期通过 `format_user_envelope()` 下沉至每条用户消息头（库内仍存干净原话，回放按 `created_at` 逐字节还原），前情摘要经 `format_compaction_summary()` 包裹 `<system-reminder>` 信源隔离围栏后置于 `messages` 头部。热聊期内（默认 5 分钟 `prompt_cache_ttl_seconds=300`）死锁 `system` 前缀，跨自然日若无过期记忆归档亦零击穿缓存，空闲超时后零代价刷新最新索引。
- **记忆生命周期与轮内暂存**：`remember_fact` 轮内先写入 `(turn#<tag>)` 暂存行，回合成功落库后升级为 `(msg#<id>)`，流中断或 `StreamError` 时自动回滚；后台沉淀支持 `add` / `merge` / `replace` / `remove`，被替换或删除的旧事实软退休至 `data/memory/archive/YYYY-MM.md`（热检索默认隔离归档目录）。
- **成对原子落库、粘性按块裁剪与轮内高水位投影**：`StateStore.append_turn()` 单事务成对写入问答消息，`get_context_snapshot()` 单锁原子读取压缩摘要与活跃历史二元组；超限时通过 `select_chunked_history()` 按会话粘性 `floor_id` 整块切至目标水位（避免逐条滑窗每轮击穿历史缓存）；单轮长工具链经 `TurnContextProjector` 接入 `convert_to_llm`，低水位（`<= 48,000` 字符）零改写吃满轮内缓存，超水位一次性折叠前半批大 `ToolResultMessage`（保留 `read_markdown` 回读指针与完整 `TurnEnd.messages` 原文）并按滞回水位锁定折叠下标使后续步骤恢复纯追加。
- **已知工程取舍**：跨轮对话历史仅持久化 `user` 与 `assistant` 文本，剥离轮内 `tool_calls` 与原始 `tool` 观察以控制 Token 与保护缓存前缀（若上轮工具细节未写入回复正文，追问时需重新调工具读取）；回合中断仅回滚未提交问答对与 `turn_tag` 暂存事实，已落盘的 `write_file` / `save_study_card` 依赖 `data/workspace/.backups/` 备份恢复。

### 3. 纵深安全边界
- **APFS Unicode 折叠与 Inode 文件隔离**：零内部依赖的 [personal_assistant/assistant/files_policy.py](personal_assistant/assistant/files_policy.py) 剥离零宽格式符并执行 `NFKC + casefold()`，配合受保护路径 `(st_dev, st_ino)` 硬链接/软链接校验（统一覆盖文件读写、`list_files` 列举与 `LinearRetriever`/`RAGRetriever` 搜索遍历），阻断 macOS APFS 字符折叠绕过（如 `ſ -> s`），严格限制只读根目录与唯一写目录 `data/workspace/`。
- **语法级只读 Shell 沙箱**：[plugins/tools/shell_tools.py](plugins/tools/shell_tools.py) 以 `shell=False`、隔离空 `HOME`（禁用系统/全局 `.gitconfig`）与独立进程组运行白名单命令：`find` 使用独立只读表达式白名单（绝不因 `--` 放过 `-exec`/`-delete` 并禁 `-L`/`-H`）；`ls`/`wc`/`rg` 封堵解引用与大小写绕过；`pytest` 强制限定 `tests/` 并忽略 `data/`；`git` 执行语法级逐子命令只读白名单（拦截 `branch -D/-m/--edit-description`、`log -p` 及冒号引用/解引用/裸对象哈希，无显式文件时自动补 `--stat`，并为 `git diff` 注入敏感路径负向排除）；超时或超 1 MB 管道输出时经 `os.killpg` 整组回收子孙进程。
- **直连 `http.client` + TLS/DNS Pinning 防 SSRF 与间接注入防护**：[plugins/tools/web_tools.py](plugins/tools/web_tools.py) 基于标准库 `http.client` + `socket.create_connection` + TLS SNI/证书校验直接连接已校验公网 IP（天然不走系统环境代理、消除二次 DNS 解析 Rebinding 窗口，自动处理非 ASCII URL 百分号与 IDNA 编码，每次阻塞读前刷新 socket 剩余截止超时），默认关闭 Fake-IP 放行（`allow_fake_ip=False`），拦截私网及 NAT64/6to4/Teredo 隧道内嵌私网地址；同轮一旦调用网络类工具，执行期上下文立即锁死 `save_rule` 写入，防范网页内容持久化篡改全局系统提示词。
- **本地 Web 服务防护**：[scripts/web_chat.py](scripts/web_chat.py) 校验 `Host`、`Origin`、`Sec-Fetch-Site` 与单一 `Content-Type: application/json`，下发 CSP / `nosniff` / `DENY` 响应头，支持非回环绑定下静态首页壳与 `/api/*` 常量时间 Bearer Token 鉴权；密钥配置文件纳入 `.gitignore` 并以 `0600` 权限原子写入。

---

## 快速开始

环境要求：**Python 3.12+** 与 [uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装虚拟环境（仅含开发测试依赖 pytest）
uv sync

# 2. 运行确定性单元测试与架构/安全/跨协议守卫（256 passed，自动隔离宿主机 .env 与模型配置，零 API 调用）
uv run pytest -v

# 3. 运行离线检索基准（支持 --fail-under 门禁）
uv run python -m evals --mode search --retriever rag --fail-under 0.90

# 4. 下载并转换公开评测集（存入 gitignore 的 evals/public/）
uv run python -m evals --mode download --dataset all --longmemeval-limit 70 --locomo-memory-limit 100 --locomo-max-convs 1
uv run python -m evals --mode search \
  --data-dir evals/public/locomo_search_data \
  --cases evals/public/locomo_search_cases.jsonl

# 5. 运行零配置离线 ReAct 演示
uv run python scripts/demo.py --show-messages
```

### 配置模型并启动对话（CLI / Web UI / DSL）

```bash
# 方式 A：终端交互向导生成配置（0600 权限）并启动命令行对话
uv run python -m personal_assistant setup
uv run python -m personal_assistant chat

# 方式 B：启动本地流式网页端（默认监听 127.0.0.1:8765；跨网卡监听可加 --auth-token）
uv run python scripts/web_chat.py --open
```

也可以直接复制 [config/models.example.dsl](config/models.example.dsl) 为 `config/models.dsl`，支持一行多厂商预设与高级 `model { api ... }` 块语法：

```text
# 一行预设（内置 deepseek / deepseek-reasoner / anthropic / gemini / qwen / kimi / glm / openai / siliconflow / ollama）
use deepseek-reasoner sk-你的DeepSeek密钥
add deepseek sk-你的DeepSeek密钥
add anthropic sk-ant-你的Anthropic密钥
add gemini AIza你的Gemini密钥
consolidator deepseek

# 高级块语法：显式声明 api 协议、思考级别 reasoning、缓存保留 cache_retention 与 max_tokens
model claude_deep {
  api             anthropic-messages
  base_url        https://api.anthropic.com
  model           claude-sonnet-4-5
  api_key         sk-ant-你的Anthropic密钥
  max_tokens      8192
  reasoning       medium
  cache_retention long
}
```

### 运行实模评测

```bash
# 单轮隔离 Agent 评测（逐题基于 AssistantApp(auto_consolidate=False) 沙箱并透传逻辑日期 today）
uv run python -m evals --mode agent --model deepseek --output /tmp/eval-agent.json

# 16 场景多轮记忆生命周期评测（中立 v2-neutral Judge）
uv run python -m evals --mode memory --model deepseek --workers 3 --output /tmp/eval-memory.json

# LongMemEval_oracle 抽样评测（按上文下载配置为 70 题；不传 limit 默认 35 题）
uv run python -m evals --mode memory \
  --cases evals/public/longmemeval_scenarios.jsonl \
  --model reasoner \
  --judge-model deepseek \
  --workers 4 \
  --output /tmp/eval-lme.json

# LoCoMo conv-26（单段 19 会话长对话，100 道 QA 共享一次沉淀后隔离问答）
uv run python -m evals --mode memory \
  --cases evals/public/locomo_memory_scenarios.jsonl \
  --model deepseek \
  --workers 4 \
  --output /tmp/eval-locomo-mem.json
```

---

## 评测方法、局限说明与实测基线

为保证可复核性，阅读下方指标时请注意以下口径边界（详细逐轮记录见 [docs/dev-log.md](docs/dev-log.md)）：

1. **开发集与外部集的区分**：[evals/sample_cases.jsonl](evals/sample_cases.jsonl)（40 题）与 [evals/memory_scenarios.jsonl](evals/memory_scenarios.jsonl)（16 场景）属于**仓库内开发与回归集**，分词与提示词迭代过程中参考过其失败样本；外部泛化表现应重点参考未参与调参的公开数据集。
2. **LongMemEval 口径说明**：`public_benchmarks.py` 拉取的官方文件为 `longmemeval_oracle.json`（**仅包含证据会话的 Oracle 子集**，不含大量无关干扰会话），按 7 个子类别固定随机种子分层抽样（`70 / 500` 题，每类 10 题；早期基线为 `35 / 500` 题）。因此该分数衡量的是**给定相关会话后的多轮记忆提炼、冲突更新与时序问答准确率（QA Accuracy）**，**并非全库大规模噪声检索召回率**；小样本下请结合置信区间解读，不宜将几题波动视为统计显著差异。
3. **LoCoMo Memory 口径说明**：默认下载配置（`--locomo-max-convs 1 --locomo-memory-limit 100`）导出的 100 道题**全部来自同一段包含 19 个时间会话的长对话 `conv-26`**（通过 `shared_setup` 对该对话沉淀一次后逐题隔离提问），而非 19 段独立对话；跨多段对话的纯词面检索能力由 `LoCoMo Search`（200 题，覆盖多段对话共 272 篇会话 `.md`）单独评估。
4. **Judge 去泄漏版本化**：自 `c761c81` / `1b421d2` 起，`evals/memory_eval.py` 移除早期评分提示词中针对公开集失败题补充的具名示例，统一采用零测试集实体的中立提示词（`JUDGE_PROMPT_VERSION = "v2-neutral"`，`prompt_sha256 = "08e7c0adcb56"`），并在输出报告中记录 `self_judged` 标记。

### 1. 离线检索基准（确定性，零 API 调用）

| 数据集 | 性质与规模 | `linear` 基线（`552863d`） | 旧 `fts5_bm25`（`8b19d97`） | `rag` Step 1（`d96e25c`） | `rag` Step 1.5 父子索引 + 整句 Overlap（本版） | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| [evals/sample_cases.jsonl](evals/sample_cases.jsonl) | 仓库内回归集（5 类，40 题） | 通过 `80.0%`（32/40），MRR `0.908`，负例干净度 `63.6%` | 通过 `95.0%`（38/40），Recall `94.8%`，MRR `0.948`，负例干净度 `100%` | 通过 `95.0%`（38/40），Recall `94.8%`，**MRR `0.966`**，负例干净度 `100%` | 通过 `95.0%`（38/40），Recall `94.8%`，**MRR `0.966`**，负例干净度 `100%` | 纯词面离线基准零退步；配置 `embedder` 后自动激活 BM25+Dense RRF 混合检索 |
| LoCoMo Search（ACL 2024） | 外部公开多对话检索（200 题，272 篇 `.md`） | 通过 `28.5%`（57/200），Recall `42.8%`，MRR `0.359` | 通过 `52.5%`（105/200），Recall `68.5%`，MRR `0.622`，Quote `60.2%` | 通过 `52.0%`（104/200），Recall `67.1%`，MRR `0.651`，Quote `60.1%` | **通过 `53.0%`（106/200）**，Recall `67.2%`，MRR `0.631`，**Quote `60.7%`** | 默认 `parent=1000 / child=320 / overlap=64`（20% 重叠）；配 `embedder` 或传 `--embedder` 可开启向量混合检索 |

### 2. 真实模型多轮记忆基准（中立 `v2-neutral` Judge 复测于 `1b421d2`）

| 数据集 | 规模与评测设置 | `1b421d2` 实测结果（95% Wilson CI） | 历史参考对照（旧 `v1` Judge） | 误差与工程观察 |
| --- | --- | --- | --- | --- |
| [evals/memory_scenarios.jsonl](evals/memory_scenarios.jsonl) | 仓库内回归集（6 类，16 场景，短词边界校验 + `deepseek-chat` 自评） | `15/16`（`93.8%`，CI `[71.7%, 98.9%]`），服务端 KV 命中 `81.3%`，前缀复用 `92.9%` | `f039de1`：`15/16`（`93.8%`），KV 命中 `82.4%` | 5/6 类全过；唯一未过项 `chained-three-step-update` 磁盘文件检查残留旧词 `PostgreSQL` |
| LoCoMo Memory（ACL 2024，`conv-26`） | 单段 19 会话长对话 `shared_setup`（100 道 QA，`deepseek-chat` 自评） | `80/100`（`80.0%`，CI `[71.1%, 86.7%]`），`adversarial` 与 `single-hop` 各 `19/22`（`86.4%`） | `dba72a3`（含具名提示泄漏）：`84/100`（`84.0%`，CI `[75.6%, 89.9%]`） | 移除测试集人物示例后下降 `4` 题，未过项集中于 `open-domain`（`9/13`）与跨会话细节合并 |
| `LongMemEval_oracle`（ICLR 2025） | 官方证据会话子集分层抽样（`70/500` 题，每类 10 题，`deepseek-chat` 自评） | `63/70`（`90.0%`，CI `[80.8%, 95.1%]`），`abstention` / `single-session-user` / `assistant` 均 `10/10` | `16ebd78`（`35/500` 子集，固定 `deepseek-chat` Judge）：`deepseek-chat` `30/35`（`85.7%`），`deepseek-reasoner` `34/35`（`97.1%`，输出 Token 约 `4.2×`） | 仅含证据会话（不含噪声检索）；7 例失败中 3 例来自时序先后与日期换算（`temporal-reasoning` `7/10`）、2 例来自多金额/地点更新（`knowledge-update` `8/10`） |

---

## 项目结构

```text
my_helper_v2/
├── config/
│   ├── tools.json                     # 工具白名单、插件路径与各模块安全策略配置
│   └── models.example.dsl             # 模型配置 DSL 示例（支持 use / add / default / consolidator / embedder / api）
├── plugins/
│   ├── tools/                         # Tool(ToolSchema) 工具子包（由 ToolsPlugin 扫描加载）
│   │   ├── plugin.py                  # ToolsPlugin(Plugin)
│   │   ├── custom_tools.py            # get_current_time / add_numbers
│   │   ├── shell_tools.py             # 受控白名单 Shell 沙箱（run_command）
│   │   ├── web_tools.py               # 联网搜索与 DNS Pinning 抓网页（web_search / fetch_webpage）
│   │   ├── history_tools.py           # 只读增量 SQLite FTS5 跨会话历史检索（search_chat_history）
│   │   └── study_tools.py             # 学习卡片与间隔复习插件（save_study_card / list_due_reviews）
│   ├── rag_retriever.py               # RetrieverProtocol：两层父子混合索引 + 向量/摘要持久化缓存 + BM25/Dense RRF 融合
│   ├── models_dsl.py                  # ConfigCompilerProtocol：模型配置 DSL 编译器（支持 api/consolidator/embedder/reasoning）
│   └── telemetry.py                   # ObserverProtocol：有界 MRU 会话耗时拆解与后台任务观测
├── personal_assistant/                # 三层单向依赖产品包（ai ← agent ← assistant）
│   ├── ai/                            # Layer 1（Pi ch04 三层翻译架构 + 缓存观测 + 向量客户端）：
│   │   ├── protocol.py                #   第二层·事件与契约协议（Context/Message/12 事件/ApiProvider）
│   │   ├── registry.py                #   第一层·统一前台（stream/stream_simple/complete）与通讯录注册表
│   │   ├── providers/                 #   第三层·原生协议翻译器：
│   │   │   ├── http.py                #     共享标准库重试/脱敏/通用 SSE 传输底座
│   │   │   ├── openai_completions.py  #     openai-completions 翻译器（OpenAI/DeepSeek Chat 等）
│   │   │   ├── openai_reasoning.py    #     openai-reasoning 翻译器（DeepSeek Reasoner/R1 等）
│   │   │   ├── anthropic_messages.py  #     anthropic-messages 原生翻译器（缓存断点/扩展思考）
│   │   │   └── google_genai.py        #     google-generative-ai 原生翻译器（思考/函数调用）
│   │   ├── cache.py                   #   中立 Context 有界 KV Cache 观测（CacheMonitor/TrackingModelClient）
│   │   └── embeddings.py              #   标准 /v1/embeddings 纯标准库客户端、float32 编解码与 math.sumprod 余弦
│   ├── agent/                         # Layer 2：tools.py / react.py / plugins.py（零 OpenAI 方言渗透）
│   └── assistant/                     # Layer 3：common.py / text_search.py / files_policy.py / chunker.py /
│                                      #          rag_contextualizer.py / prompts.py / context.py / memory.py /
│                                      #          builtin_tools.py / memory_lifecycle.py / db.py / consolidator.py /
│                                      #          commands.py / app.py / cli.py
├── scripts/
│   ├── demo.py                        # 零配置离线 ReAct 演示脚本
│   ├── web_chat.py                    # 本地 HTTP/SSE 服务
│   └── web_ui.html                    # 原生 HTML/JS 流式教学界面
├── evals/                             # 顶层评测包（search / agent / memory / download）
├── tests/                             # 确定性单测、conftest.py 宿主隔离、跨协议契约测试与架构/红队守卫
├── docs/
│   ├── mvp-design.md                  # 架构设计、工程权衡与安全边界文档
│   └── dev-log.md                     # 当前基线、ADRs、排坑记录与演进日志
├── AGENTS.md                          # 协作规范与研发流程约定
└── pyproject.toml
```

## License

[MIT](LICENSE)
