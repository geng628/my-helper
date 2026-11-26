# My Helper v2

一个零第三方运行依赖的纯 Python 个人与学习 AI Agent。项目以借鉴 Pi 洋葱模型的三层单向依赖微内核（`ai ← agent ← assistant`）为底座，在外层通过插件协议叠加纯 Markdown 长期记忆生命周期、可折叠流式推导视图、间隔重复学习卡片与前后台双模型分槽。生产代码仅使用 Python 3.12+ 标准库，`pytest` 仅作开发测试依赖。

> 系统设计与工程权衡详见 [docs/mvp-design.md](docs/mvp-design.md)；当前架构基线、踩坑根因、去泄漏评测总账与迭代记录详见 [docs/dev-log.md](docs/dev-log.md) 与 [AGENTS.md](AGENTS.md)。

---

## 架构概览

1. **三层单向依赖漏斗（`ai ← agent ← assistant`）**：
   - **Layer 1 `personal_assistant/ai/`**：零内部依赖的 LLM 传输客户端、SSE 流式解析、模型配置档案与按会话隔离的有界 KV Cache 观测。
   - **Layer 2 `personal_assistant/agent/`**：仅依赖 `ai/` 的通用单层 ReAct 循环、工具类型递进扩展（`ToolSchema → Tool`）与统一插件主板，不含业务路径或具体插件名硬编码。
   - **Layer 3 `personal_assistant/assistant/`**：个人与学习助手业务层（Markdown 记忆、安全策略、SQLite 对话历史、外层回合预算、后台压缩沉淀与 CLI/Web 接入）。
   - 顶层评测包 `evals/` 与 `plugins/`、`tests/` 平级；[tests/test_architecture.py](tests/test_architecture.py) 通过 AST 依赖图、业务字面量扫描与子进程 `sys.modules` 隔离测试守卫分层边界。
2. **极简单层 ReAct 内核 + 外层安全预算**：
   - 面向个人伴读与学习辅导场景，[personal_assistant/agent/react.py](personal_assistant/agent/react.py) 保持为 30 余行单层 `while True` 循环（`iter_react()` / `run_react()`），仅以 `extract_tool_calls(assistant_msg)` 为空作为自然终止信号，并提供 `transform_context -> convert_to_llm` 边界转换钩子与执行期 `tool_context` 注入。
   - 防死循环与成本控制由 Layer 3 `AssistantApp` 的 `_GuardedTurnLLM` 在外层实施（默认单轮上限 `24` 步、`250,000` Token、`180s` 墙钟、连续 `3` 次等价工具调用熔断及 `finish_reason="length"` 截断工具调用熔断），兼顾内核简洁性与运行期有界性。
3. **学习助手叠加能力（零微内核侵入）**：
   - **流式推导视图**：通过 `ThinkingDelta` 与 `ToolStart` 事件，在 CLI 与本地网页端实时展示可折叠的推理过程与慢工具执行状态，同时保持落库历史仅含纯净问答正文。
   - **前后台双模型分槽**：支持在 `config/models.dsl` 中配置 `consolidator <模型名>`，前台使用推理模型（如 `deepseek-reasoner`）讲题推导，后台使用轻量对话模型（如 `deepseek-chat`）执行结构化记忆沉淀。
   - **纯 Markdown 间隔复习卡片**：通过外围插件 [plugins/tools/study_tools.py](plugins/tools/study_tools.py) 提供 `save_study_card` 与 `list_due_reviews`，在 `data/workspace/study/` 下维护人类可读的复习卡片与调度计划。

---

## 核心工程特性

### 1. 五大插件主板插槽与单次初始化生命周期
- 外层基类 `Plugin(ABC)` 在启动期执行一次 `initialize()` 向 `PluginContext` 注册五类子协议，关闭时经 `AssistantApp.close()` 触发 `terminate_all()`：
  1. **工具插槽 `Tool(ToolSchema)`**：[plugins/tools/plugin.py](plugins/tools/plugin.py) 自动扫描 `plugins/tools/*.py` 中的 `@tool` 函数，受 `config/tools.json` 白名单管控。
  2. **检索插槽 `RetrieverProtocol`**：[plugins/fts5_retriever.py](plugins/fts5_retriever.py)（`fts5_bm25`），结合英文词形还原与中文虚词切分 + Bigram，共享零依赖分词底座 [personal_assistant/assistant/text_search.py](personal_assistant/assistant/text_search.py)。
  3. **配置编译插槽 `ConfigCompilerProtocol`**：[plugins/models_dsl.py](plugins/models_dsl.py)，手写词法与递归下降解析器，将精简 DSL 编译为 `ModelsConfig` 标准字典。
  4. **模型协议插槽 `LLMProviderProtocol`**：[plugins/reasoning_provider.py](plugins/reasoning_provider.py)（`openai_reasoning`），处理轮内 `reasoning_content` 续传与跨轮剥离。
  5. **运行观测插槽 `ObserverProtocol`**：[plugins/telemetry.py](plugins/telemetry.py)，以有界环形窗口统计分阶段耗时（`total_ms` / `ttft_ms` / `llm_ms` / `tool_ms`）与后台沉淀事件。
- `AssistantApp` 在启动期构建一次常驻 `_tool_registry`，每轮通过 `tool_context` 注入动态回合上下文；`build_tool()` 自动从 Python 默认参数推断精简 `ToolSchema.required`。

### 2. KV Cache 友好的 Markdown 记忆、原子会话快照与沉淀韧性
- **前缀冻结**：`data/memory/MEMORY.md` 限制为前 100 行 / 4000 字符索引，与 `rules/` 摘要按 `(today_str, chat_id)` 在会话内冻结于 `system` 前缀，跨自然日自动刷新。
- **记忆生命周期与轮内暂存**：`remember_fact` 轮内先写入 `(turn#<tag>)` 暂存行，回合成功落库后升级为 `(msg#<id>)`，流中断时自动回滚；后台沉淀支持 `add` / `merge` / `replace` / `remove`，被替换或删除的旧事实软退休至 `data/memory/archive/YYYY-MM.md`（热检索默认隔离归档目录）。
- **成对原子落库与自适应滑窗**：`StateStore.append_turn()` 单事务成对写入问答消息，`get_context_snapshot()` 单锁原子读取压缩摘要与活跃历史二元组；当后台沉淀繁忙或处于指数退避（默认 `2s` 起步至 `256s`，上限 `300s`）导致未压缩消息超过 `20` 条时，活跃窗口在 `60` 条 / `24,000` 字符上限内自动拓宽保留未压缩上下文，单次沉淀批次上限 `24` 条并在写锁内保护前台在途 `(turn#...)` 事实不被覆盖。
- **已知工程取舍**：跨轮对话历史仅持久化 `user` 与 `assistant` 文本，剥离轮内 `tool_calls` 与原始 `tool` 观察以控制 Token 与保护缓存前缀（若上轮工具细节未写入回复正文，追问时需重新调工具读取）；回合中断仅回滚未提交问答对与 `turn_tag` 暂存事实，已落盘的 `write_file` / `save_study_card` 依赖 `data/workspace/.backups/` 备份恢复。

### 3. 纵深安全边界
- **APFS Unicode 折叠与 Inode 文件隔离**：[personal_assistant/assistant/files_policy.py](personal_assistant/assistant/files_policy.py) 剥离零宽格式符并执行 `NFKC + casefold()`，配合受保护路径 `(st_dev, st_ino)` 硬链接/软链接校验，阻断 macOS APFS 字符折叠绕过（如 `ſ -> s`），严格限制只读根目录与唯一写目录 `data/workspace/`。
- **逐子命令只读 Shell 沙箱**：[plugins/tools/shell_tools.py](plugins/tools/shell_tools.py) 以 `shell=False`、隔离空 `HOME`（禁用系统/全局 `.gitconfig`）与独立进程组运行白名单命令：`pytest` 强制限定 `tests/` 并忽略 `data/`；`git` 执行逐子命令只读白名单（拦截 `branch -D/-m/--edit-description`、`log -p` 及裸 `<blob>`/`<tree>` `show`/`diff`，并为 `git diff` 注入敏感路径负向排除）；超时或超 1 MB 管道输出时经 `os.killpg` 整组回收子孙进程。
- **DNS Pinning 防 SSRF 与间接注入防护**：[plugins/tools/web_tools.py](plugins/tools/web_tools.py) 禁用环境代理、默认关闭 Fake-IP 放行（`allow_fake_ip=False`），拦截私网及 NAT64/6to4/Teredo 内嵌私网地址，并通过自定义 `_PinnedHTTPConnection` / `_PinnedHTTPSConnection` 直连已校验 IP 消除 DNS Rebinding 窗口；同轮一旦调用网络类工具，执行期上下文立即锁死 `save_rule` 写入，防范网页内容持久化篡改全局系统提示词。
- **本地 Web 服务防护**：[scripts/web_chat.py](scripts/web_chat.py) 校验 `Host`、`Origin`、`Sec-Fetch-Site` 与单一 `Content-Type: application/json`，下发 CSP / `nosniff` / `DENY` 响应头，支持非回环绑定下的常量时间 Bearer Token 鉴权；密钥配置文件纳入 `.gitignore` 并以 `0600` 权限原子写入。

---

## 快速开始

环境要求：**Python 3.12+** 与 [uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装虚拟环境（仅含开发测试依赖 pytest）
uv sync

# 2. 运行确定性单元测试与架构/安全守卫（自动隔离宿主机 .env 与模型配置，零 API 调用）
uv run pytest -v

# 3. 运行离线检索基准（支持 --fail-under 门禁）
uv run python -m evals --mode search --retriever fts5_bm25 --fail-under 0.90

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

也可以直接复制 [config/models.example.dsl](config/models.example.dsl) 为 `config/models.dsl` 手写双模型配置：

```text
use reasoner sk-你的DeepSeek密钥
add deepseek sk-你的DeepSeek密钥
consolidator deepseek
```

### 运行实模评测

```bash
# 单轮隔离 Agent 评测（逐题重建独立 MemoryStore 沙箱）
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

### 1. 离线检索基准（确定性，零 API 调用，验证于 `1b421d2`）

| 数据集 | 性质与规模 | `linear` 基线（`552863d`） | `fts5_bm25`（`1b421d2`） | 说明 |
| --- | --- | --- | --- | --- |
| [evals/sample_cases.jsonl](evals/sample_cases.jsonl) | 仓库内回归集（5 类，40 题） | 通过 `80.0%`（32/40），MRR `0.908`，负例干净度 `63.6%` | 通过 `95.0%`（38/40），Recall `94.8%`，MRR `0.948`，负例干净度 `100%` | 剩余 2 题失败为纯同义零共享词改写与跨文档分散引用 |
| LoCoMo Search（ACL 2024） | 外部公开多对话检索（200 题，272 篇 `.md`） | 通过 `28.5%`（57/200），Recall `42.8%`，MRR `0.359` | 通过 `52.5%`（105/200），Recall `68.5%`，MRR `0.622` | `single-hop` Recall `94.0%`、`temporal` `92.0%`；但 `multi-hop`（`43.9%`）与 `open-domain`（`44.2%`）受限于纯词面匹配 |

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
│   └── models.example.dsl             # 模型配置 DSL 示例（use / add / default / consolidator）
├── plugins/
│   ├── tools/                         # Tool(ToolSchema) 工具子包（由 ToolsPlugin 扫描加载）
│   │   ├── plugin.py                  # ToolsPlugin(Plugin)
│   │   ├── custom_tools.py            # get_current_time / add_numbers
│   │   ├── shell_tools.py             # 受控白名单 Shell 沙箱（run_command）
│   │   ├── web_tools.py               # 联网搜索与 DNS Pinning 抓网页（web_search / fetch_webpage）
│   │   ├── history_tools.py           # 只读增量 SQLite FTS5 跨会话历史检索（search_chat_history）
│   │   └── study_tools.py             # 学习卡片与间隔复习插件（save_study_card / list_due_reviews）
│   ├── fts5_retriever.py              # RetrieverProtocol：SQLite FTS5 + BM25 双语检索
│   ├── models_dsl.py                  # ConfigCompilerProtocol：模型配置 DSL 编译器
│   ├── reasoning_provider.py          # LLMProviderProtocol：reasoning_content 推理模型协议
│   └── telemetry.py                   # ObserverProtocol：有界窗口耗时拆解与后台任务观测
├── personal_assistant/                # 三层单向依赖产品包（ai ← agent ← assistant）
│   ├── ai/                            # Layer 1：types.py / client.py / cache.py
│   ├── agent/                         # Layer 2：tools.py / react.py / plugins.py
│   └── assistant/                     # Layer 3：common.py / text_search.py / files_policy.py / prompts.py /
│                                      #          memory.py / builtin_tools.py / memory_lifecycle.py /
│                                      #          db.py / consolidator.py / commands.py / app.py / cli.py
├── scripts/
│   ├── demo.py                        # 零配置离线 ReAct 演示脚本
│   ├── web_chat.py                    # 本地 HTTP/SSE 服务
│   └── web_ui.html                    # 原生 HTML/JS 流式教学界面
├── evals/                             # 顶层评测包（search / agent / memory / download）
├── tests/                             # 确定性单测、conftest.py 宿主隔离与架构/红队守卫
├── docs/
│   ├── mvp-design.md                  # 架构设计、工程权衡与安全边界文档
│   └── dev-log.md                     # 当前基线、ADRs、排坑记录与演进日志
├── AGENTS.md                          # 协作规范与研发流程约定
└── pyproject.toml
```

## License

[MIT](LICENSE)
