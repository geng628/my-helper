# My Helper v2

一个**零第三方运行依赖**的纯 Python 个人与学习 AI Agent：以**借鉴 Pi 洋葱模型的极简 ReAct 微内核**为底座，向上正交叠加长期记忆生命周期、可折叠流式推导视图、间隔重复知识卡片与前后台双模型分槽。生产代码仅使用 Python 3.12+ 标准库（`urllib`、`sqlite3`、`threading`、`http.server`、`shlex` 等），`pytest` 仅作开发测试依赖。

> 系统架构设计见 [docs/mvp-design.md](docs/mvp-design.md)；完整的工程演进、踩坑根因、同口径 Benchmark 总账与跨模型交接记录见 [docs/dev-log.md](docs/dev-log.md) 与 [AGENTS.md](AGENTS.md)。

---

## 架构哲学：为什么选择「Pi 洋葱微内核 + 学习助手叠加层」

1. **三层单向依赖漏斗（`ai ← agent ← assistant`）**：
   - **Layer 1 `personal_assistant/ai/`**：零内部依赖的 LLM 传输协议、流式事件、模型配置档案与有界 KV Cache 观测底座。
   - **Layer 2 `personal_assistant/agent/`**：仅依赖 `ai/` 的通用 ReAct 引擎、工具类型递进扩展（`ToolSchema → Tool`）与统一插件主板。
   - **Layer 3 `personal_assistant/assistant/`**：个人与学习助手业务编排层（纯 Markdown 记忆、SQLite 对话状态、后台沉淀压缩与 CLI/Web 接入）。
   - 顶层评测包 `evals/` 与 `plugins/`、`tests/` 平级；[tests/test_architecture.py](tests/test_architecture.py) 通过 **AST 静态依赖图 + 环检测 + 子进程 `sys.modules` 断层隔离测试** 强制守卫“剥掉任何上层，里层仍可独立运行”。
2. **为什么不照搬 `coding-agent` 的双层插队队列**：
   - 本项目面向个人伴读与学习辅导场景，交互模式为“单轮深度推导 + 按需工具查证 + 后台自动整理笔记”，不需要多并发外部输入抢占执行中的工具队列。
   - 因此刻意保持 [personal_assistant/agent/react.py](personal_assistant/agent/react.py) 的 `iter_react()` / `run_react()` 为 30 余行一眼看穿的**纯粹单层 `while True` 循环**：移除人为 `MAX_TOOL_ROUNDS` 截断与 `tools=None` 强行总结兜底，循环前一次性固定 `tool_defs`，全程只追加消息不篡改，仅以强类型 `ToolCall` 列表为空（`not extract_tool_calls(assistant_msg)`）作为唯一自然终止信号。
   - 同时在通用循环边界提供 `transform_context -> convert_to_llm` 两阶段纯函数转换钩子（缺省 `None` 零拷贝直通；启用时自动字典级浅拷贝隔离），让业务层摘要注入与协议兼容转换不污染通用内核。
3. **个人学习助手零微内核侵入叠加**：
   - **流式推导教学视图**：通过 Layer 1/2 原生 `ThinkingDelta` 与 `ToolStart` 事件，在 `AssistantApp.stream_chat` / `cli.py` / `scripts/web_ui.html` 全链路支持可折叠流式思维链（`thinking_delta`）与慢工具启动指示（`tool_start`）。
   - **前后台双模型性价比分槽**：`config/models.dsl` 支持顶层 `consolidator <模型名>` 指令，实现“前台用 `deepseek-reasoner` 深度讲题推导、后台 `MemoryConsolidator` 用轻量 `deepseek-chat` 低成本整理笔记”。
   - **纯 Markdown 间隔重复卡片**：通过 `plugins/tools/study_tools.py` 扩展 `save_study_card` 与 `list_due_reviews`，把学习知识卡片与 Leitner/SM-2 复习计划作为可插拔工具挂在工作区上。

---

## 五大核心工程亮点

### 1. 纯净微内核与五大主板插槽（`Plugin ABC` + `PluginContext`）
- 外层统一基类 `Plugin(ABC)`（`__init_subclass__` 自动发现、`initialize()` / `terminate()` 生命周期）向主板 `PluginContext` 注册五大强类型子协议（核心运行时只认子协议，不感知实现是否来自插件）：
  1. **工具插槽 `Tool(ToolSchema)`**：由 [plugins/tools/plugin.py](plugins/tools/plugin.py)（`ToolsPlugin`）扫描 `plugins/tools/*.py` 中的 `@tool` 函数；在 [plugins/tools/custom_tools.py](plugins/tools/custom_tools.py) 声明并入 `config/tools.json` 白名单即可热插拔。
  2. **检索插槽 `RetrieverProtocol`**：[plugins/fts5_retriever.py](plugins/fts5_retriever.py)（`fts5_bm25`），内置英文不规则词形还原/词干、中文虚词切分 + Bigram、SQL 级过期过滤、增量 mtime/size 同步与 `os.scandir` 遍历优化。
  3. **配置编译插槽 `ConfigCompilerProtocol`**：[plugins/models_dsl.py](plugins/models_dsl.py)（`models_dsl`），手写 Lexer + 递归下降 AST + 精确行列报错，支持 8 大厂商预设、`use`/`add` 一行语法、`model { ... }` 块语法与 `consolidator` 双模型分槽指令。
  4. **模型协议插槽 `LLMProviderProtocol`**：[plugins/reasoning_provider.py](plugins/reasoning_provider.py)（`openai_reasoning`），复用 Layer 1 `OpenAIChatClient` 传输底座，声明 `reasoning_content` 流式 `ThinkingDelta`、轮内工具续传保留与跨轮安全剥离。
  5. **运行观测插槽 `ObserverProtocol`**：[plugins/telemetry.py](plugins/telemetry.py)（`telemetry`），无锁外侵收集 `total_ms` / `ttft_ms` / `llm_ms` / `tool_ms` / `overhead_ms` 分阶段耗时拆解及后台 `TaskEvent` 账本，配合有界环形窗口与 MRU 会话淘汰。
- **单次初始化解耦**：`PluginManager.from_entries()` 仅在装配期执行一次 `Plugin.initialize()` 并缓存工具规格；每轮 `build_tools(context)` 只按函数签名重绑运行时上下文并克隆自定义 `Tool` 子类，不再重跑初始化。
- **有界观测与传输加固**：`CacheMonitor(max_records=2000, max_sessions=256)` 配合单调 `snapshot_index()` 与 $O(1)$ 累计账本；`default_post_json` / `default_post_stream` 支持瞬时网络/TLS 异常（`IncompleteRead`、`SSLError`、`429`/`5xx`）指数退避重试；大输出通过 [personal_assistant/assistant/common.py](personal_assistant/assistant/common.py) `truncate_or_offload()` 分级卸载至 `data/workspace/tool_outputs/`（`head` / `head_tail` 预览 + `read_markdown` 续读）。

### 2. KV Cache 优先的纯 Markdown 记忆与成对原子会话存储
- **前缀冻结**：`data/memory/MEMORY.md` 严控为限长索引（前 100 行 / 4000 字符），与 `rules/` 摘要一并按 `(today_str, chat_id)` 在会话内冻结在 `system` 前缀，跨自然日自动刷新；历史对话采用标准追加式 `[system, user_1, assistant_1, ...]`，最大化服务端 Prefix Caching。
- **轮内暂存与提交绑定**：`remember_fact` 将用户原话写入 `data/memory/facts.md`（不改写 `MEMORY.md`），流式回合内先带 `(turn#<tag>)` 标记使同轮检索即时可见，回合成功落库后绑定正式 `(msg#<id>)`，流中断异常自动回滚清理。
- **生命周期与冷账本**：[personal_assistant/assistant/memory_lifecycle.py](personal_assistant/assistant/memory_lifecycle.py) 支持结构化 `operations`（`add` / `merge` / `replace` / `remove`）：`merge` 无损拼接互补维度；`replace` / `remove` 将退场旧值软退休至 `data/memory/archive/YYYY-MM.md`（`superseded_by` 替代链），默认热检索隔离归档目录；`(until: YYYY-MM-DD)` 到期自动按月归档并清理 `facts.md` 中的已吸收幽灵行。
- **成对原子落库与只读跨会话回溯**：`StateStore.append_turn()` 在单个 SQLite WAL 事务内成对原子写入 `user` + `assistant`（守住 `messages` 100% 只追不删与问答成对不变量）；`BackgroundConsolidator` 达阈值（16 条或 8000 字符）在后台线程沉淀记忆并单调推进 `session_compactions.compacted_until_id`；[plugins/tools/history_tools.py](plugins/tools/history_tools.py) 提供 `search_chat_history`，以 `?mode=ro` 只读 URI 增量同步内存 FTS5 索引并按 `before_msg_id` 严格隔离当前未完成回合。

### 3. 个人学习助手专属叠加（零微内核侵入）
- **实时流式思维链折叠面板**（同波次特性）：`AssistantApp.stream_chat`、`cli.py` 与 `scripts/web_ui.html` 全链路透传并渲染 `thinking_delta` 与 `tool_start`，在推导数学、算法或源码问题时可按需展开/折叠思考过程，主对话历史保持纯净。
- **前后台双模型分槽**（同波次特性）：在 `config/models.dsl` 中通过 `consolidator <模型名>` 为后台 `MemoryConsolidator` 指定独立轻量模型，兼顾 `deepseek-reasoner` 的复杂推导精度与 `deepseek-chat` 的低成本结构化沉淀。
- **纯 Markdown 知识卡片与间隔复习**（同波次特性，PR `#64`）：`plugins/tools/study_tools.py` 提供 `save_study_card`（在 `data/workspace/study/<slug>.md` 创建或幂等更新结构化学习卡片，并按 Leitner/SM-2 精简调度计算下次复习日）与 `list_due_reviews`（查询到期复习队列与掌握进度）。

### 4. 纵深防御安全边界
- **文件读写隔离**：`FilesPolicy` 严格限制 `read_roots`（`data/memory`、`data/workspace`、`docs`）与唯一 `write_root = data/workspace`（覆盖前自动备份到 `.backups/`），逐段拒绝 `..`、符号链接与敏感路径段（`.env`、`.git`、`.ssh`、`state`、`uploads`、`models.dsl`、`models.json`）；禁止 LLM 无人工审批修改 `personal_assistant/` 与 `plugins/`。
- **Shell 参数级防逃逸**：[plugins/tools/shell_tools.py](plugins/tools/shell_tools.py) 经 `shlex` 拦截管道/重定向/`$()`/反引号并以 `shell=False` + 脱敏环境执行白名单命令：封堵 `write_file + pytest` 两步 RCE（`pytest` 强制限制在 `tests/` 并注入 `--ignore=data`）、`git` 全局选项走私与 pathspec 魔法、`grep`/`rg` 后置 `-e` 参数顺序绕过与高优先级排除项、以及大小写不敏感敏感段与 `.env.*` 变体（精确放行 `.env.example`）。
- **Web SSRF 与本地 HTTP CSRF 防护**：[plugins/tools/web_tools.py](plugins/tools/web_tools.py) 对初始请求及每一跳重定向前调用 `validate_public_url` 阻断私网/回环/云元数据 SSRF（同时对非字面 IP 域名解析兼容 macOS TUN `198.18.0.0/15` Fake-IP）；[scripts/web_chat.py](scripts/web_chat.py) 全路由校验 `Host`（含 IPv6 RFC 5952 规范化）、对 `/api/*` 与 `POST` 校验同源 `Origin` 与 `Sec-Fetch-Site in {"same-origin", "none"}`、强制单一非逗号折叠 `Content-Type: application/json`（`415` 拦截 Simple Request CSRF）。
- **密钥零心智防泄漏**：`config/models.dsl` 与编译产物 `config/models.json` 自动纳入 `.gitignore`、以 `0600` 权限原子落盘、`redacted_dict()` 全链路脱敏并受文件与 Shell 双沙箱屏蔽。

### 5. 四层可复现评测体系
- 所有评测引擎与数据集统一收拢在顶层 [evals/](evals/)（`uv run python -m evals`），与产品代码双向解耦：
  - **Layer 1 确定性单测与架构守卫**：`tests/`（覆盖 AST 漏斗、子进程隔离、流式 SSE、安全对抗、生命周期与并发边界）；
  - **Layer 2 离线检索评测**：`--mode search`（40 题自建 `evals/sample_cases.jsonl` + 200 题公开 LoCoMo，支持 `linear` / `fts5_bm25` A/B）；
  - **Layer 3 单轮隔离 Agent 评测**：`--mode agent`（在临时隔离目录校验工具调用链与端到端 KV 缓存）；
  - **Layer 4 多轮记忆生命周期评测**：`--mode memory`（16 场景 `evals/memory_scenarios.jsonl` + 公开 LongMemEval ICLR 2025 / LoCoMo ACL 2024 `shared_setup` 共享沉淀，支持 `--workers` 保序并发、`--model` 与独立 `--judge-model` 同口径对比）。

---

## 快速开始

需要 **Python 3.12+** 与 [uv](https://docs.astral.sh/uv/)。

```bash
# 1. 初始化虚拟环境（仅安装开发测试依赖 pytest）
uv sync

# 2. 运行全部确定性单元测试与三层架构守卫（零 API Token）
uv run pytest -v

# 3. 运行离线检索评测（40 题内置样本，linear vs fts5_bm25 同口径对比）
uv run python -m evals --mode search --retriever linear
uv run python -m evals --mode search --retriever fts5_bm25

# 4. 下载并运行公开学术基准（自动存入 gitignore 的 evals/public/）
uv run python -m evals --mode download --dataset all --longmemeval-limit 35 --locomo-memory-limit 100 --locomo-max-convs 1
uv run python -m evals --mode search \
  --data-dir evals/public/locomo_search_data \
  --cases evals/public/locomo_search_cases.jsonl

# 5. 零配置离线演示 ReAct 循环与错误自愈轨迹
uv run python scripts/demo.py --show-messages
```

### 配置模型并启动对话（CLI / Web UI / DSL）

支持三种配置入口，均默认启用 `.gitignore`、`0600` 权限与双沙箱屏蔽：

```bash
# 方式 A：终端交互向导（选厂商 + 贴密钥，自动生成 config/models.dsl 与 models.json）
uv run python -m personal_assistant setup
uv run python -m personal_assistant chat

# 方式 B：零依赖本地网页端（支持流式推导思维链折叠与右上角「模型设置」热切换）
uv run python scripts/web_chat.py --open
```

**方式 C：极简手写 `config/models.dsl`（推荐双模型学习助手配置）**

复制 [config/models.example.dsl](config/models.example.dsl) 为 `config/models.dsl`，将深度推理模型设为主对话模型，并用 `consolidator` 指令将后台笔记整理分流给轻量对话模型（工具扩展见可自由编辑的 [plugins/tools/custom_tools.py](plugins/tools/custom_tools.py)）：

```text
use reasoner sk-你的DeepSeek密钥
add deepseek sk-你的DeepSeek密钥
consolidator deepseek
```

### 运行端到端实模评测（支持 `--model` 与独立 `--judge-model`）

```bash
# 单轮 Agent 评测
uv run python -m evals --mode agent --model deepseek --output /tmp/eval-agent.json

# 16 场景多轮记忆生命周期评测（并发 workers=4）
uv run python -m evals --mode memory --model deepseek --workers 4 --output /tmp/eval-memory.json

# LongMemEval 35 题分层抽样：测试 deepseek-reasoner，固定 deepseek-chat 作为独立 Judge
uv run python -m evals --mode memory \
  --cases evals/public/longmemeval_scenarios.jsonl \
  --model reasoner \
  --judge-model deepseek \
  --workers 5 \
  --output /tmp/eval-lme-reasoner.json

# LoCoMo 100 题共享长记忆沉淀（shared_setup 只沉淀一次，逐 QA 隔离复用）
uv run python -m evals --mode memory \
  --cases evals/public/locomo_memory_scenarios.jsonl \
  --model deepseek \
  --workers 5 \
  --output /tmp/eval-locomo-mem.json
```

---

## 评测体系与当前基线

以下基线全部取自 [docs/dev-log.md](docs/dev-log.md) 已记录且绑定 Git SHA 的实测结果（开放问答采用与 Agent KV Cache 统计严格隔离的 `LLM-as-a-Judge` 评分，未删除或挑拣任何失败题）：

### 1. 离线检索基准（零 Token）

| 评测集 | 类型与规模 | `linear` 基线（`552863d`） | 插件 `fts5_bm25`（`d2e20e3` ~ `7e64db0`） | 关键变化 |
| --- | --- | --- | --- | --- |
| [evals/sample_cases.jsonl](evals/sample_cases.jsonl) | 离线检索（5 分类，40 题） | 通过 **80.0%**（32/40），MRR **0.908**，负例干净度 **63.6%**，`0.72 ms` | 通过 **95.0%**（38/40），MRR **0.948**，Quote **94.8%**，负例干净度 **100%**，`0.53 ms` | **+15.0 pp**（+6 / 0 退步）；仅剩纯语义同义改写待可选向量/RRF |
| LoCoMo Search（ACL 2024） | 公开多会话离线检索（200 题抽样，272 篇 `.md`） | 通过 **28.5%**（57/200），Recall **42.8%**，MRR **0.359**，`29.0 ms` | 通过 **52.5%**（105/200），Recall **68.5%**（`single-hop` 94% / `temporal` 92%），MRR **0.622**，`9.5 ms` | Recall **+25.7 pp**，MRR **+0.263**，延迟提速 **~3×**（+51 / -3） |

### 2. 真实模型多轮记忆与端到端基准

| 评测集 | 规模与协议 | 对比基线（`deepseek-chat`） | 最新同口径表现 | 备注与工程结论 |
| --- | --- | --- | --- | --- |
| [evals/memory_scenarios.jsonl](evals/memory_scenarios.jsonl) | 6 分类多轮记忆生命周期（16 场景） | `cffbf26`：**81.25%**（13/16），KV 命中 60.4% / 前缀复用 90.4% | 移除 `MAX_TOOL_ROUNDS` 回归纯 ReAct（`b7b14b5`）：**100%**（**16/16**），KV 命中 **75.2%** / 前缀复用 **91.4%** | 6 大分类全 1.0，零死循环；`deepseek-reasoner`（`16ebd78`）为 **93.75%**（15/16，KV 69.3%，`temp-fact-expiry` 多调 `save_rule` 残留 1 条但最终答对） |
| LongMemEval（ICLR 2025） | 公开 7 分类分层抽样（35 题，固定 `deepseek-chat` Judge） | `deepseek-chat`（`0a6cf70` / `16ebd78`）：**85.71%**（30/35），KV 命中 55.1% | `deepseek-reasoner`（`16ebd78`, `workers=5`）：**97.14%**（**34/35**，**6/7 分类 100%**，仅 `lme-9ea5eabc` 1 题未过） | **+11.43 pp**（+4 / 0 退步）；`completion_tokens` 增至 **4.2×**（36,730 → 154,655），直接催生前后台 `consolidator` 双模型分槽 |
| LoCoMo Memory（ACL 2024） | 全会话共享沉淀 `shared_setup`（100 题，19–32 会话） | — | `deepseek-chat`（`dba72a3`, `workers=5`）：**84.00%**（**84/100**，`temporal` **19/21**，墙钟 `300.65s`） | 前 50 题 `43/50`、后 50 题 `41/50`，不放宽防幻觉拒答红线 |

> 该实测同时验证：当前 `MEMORY.md` 索引 + `fts5_bm25` + `search_chat_history` 架构在 LongMemEval 35 题的召回上限已达 **97.14%**，因此主线保持**零向量库依赖**，把工程重心放在推导可读性与双模型性价比优化。

---

## 项目结构

```text
my_helper_v2/
├── config/
│   ├── tools.json                     # 唯一工具白名单、插件路径、检索/遥测/Web/Shell 策略与固定命令配置
│   └── models.example.dsl             # 极简模型配置 DSL 示例（支持 use / add / consolidator）
├── plugins/
│   ├── tools/                         # Tool(ToolSchema) 工具子包（由 ToolsPlugin 统一扫描加载）
│   │   ├── __init__.py
│   │   ├── plugin.py                  # ToolsPlugin(Plugin)：自动扫描本目录 @tool 模块并注册
│   │   ├── custom_tools.py            # 可自由编辑的工具模块（get_current_time / add_numbers）
│   │   ├── shell_tools.py             # 受控白名单 Shell 插件（run_command + 防两步 RCE/选项走私沙箱）
│   │   ├── web_tools.py               # 联网搜索与防 SSRF 抓网页插件（web_search / fetch_webpage）
│   │   ├── history_tools.py           # 只读增量 SQLite FTS5 跨会话历史检索（search_chat_history）
│   │   └── study_tools.py             # 学习助手卡片与间隔复习插件（save_study_card / list_due_reviews，PR #64）
│   ├── fts5_retriever.py              # RetrieverProtocol 插件：SQLite FTS5 + BM25 双语检索
│   ├── models_dsl.py                  # ConfigCompilerProtocol 插件：手写 Lexer + 递归下降 AST DSL 编译器
│   ├── reasoning_provider.py          # LLMProviderProtocol 插件：DeepSeek Reasoner / R1 等 reasoning_content 协议
│   └── telemetry.py                   # ObserverProtocol 插件：有界窗口、分阶段耗时拆解与后台任务观测
├── personal_assistant/                # 纯净三层单向依赖漏斗产品包（ai ← agent ← assistant，绝不依赖 evals/）
│   ├── __init__.py
│   ├── __main__.py                    # `python -m personal_assistant` 入口
│   ├── ai/                            # Layer 1（零内部依赖）：LLM 传输底座、事件契约与缓存观测
│   │   ├── __init__.py
│   │   ├── types.py                   # ToolSchema、ToolCall、TextDelta、ThinkingDelta、StreamDone、ModelProfile、ModelsConfig
│   │   ├── client.py                  # LLMClient、OpenAIChatClient、OpenAICompatibleLLM、重试/SSE 与 Provider 注册中心
│   │   └── cache.py                   # CacheMonitor 有界滑动窗口（max_records=2000, max_sessions=256）与 TrackingLLMClient
│   ├── agent/                         # Layer 2（仅依赖 ai）：通用 ReAct 微内核与五大插件主板插槽
│   │   ├── __init__.py
│   │   ├── tools.py                   # Tool(ToolSchema)、ToolMetadata、ToolDocument、build_tool、ToolsetProvider、ToolRegistry
│   │   ├── react.py                   # 30 余行单层 while True ReAct 循环、transform_context -> convert_to_llm 与强类型事件
│   │   └── plugins.py                 # Plugin(ABC)、PluginContext（5 插槽）与 PluginManager（单次初始化 + 签名重绑）
│   └── assistant/                     # Layer 3（个人与学习助手业务层）：记忆、SQLite 历史、沉淀、编排与终端
│       ├── __init__.py
│       ├── common.py                  # 无内部依赖底座：REPO_ROOT、load_dotenv、星期格式与 TOAST 卸载原语
│       ├── prompts.py                 # 个人助手系统提示词与前缀分段拼装（从通用 react.py 剥离）
│       ├── memory.py                  # FilesPolicy、MemoryStore（写锁/turn_tag 暂存/原子写盘）与 MemoryToolsetProvider
│       ├── memory_lifecycle.py        # add/merge/replace/remove 冲突替换、superseded_by 软退休与 (until:) 过期归档
│       ├── db.py                      # 无内部依赖线程安全 SQLite（messages 成对原子 append_turn + session_compactions）
│       ├── consolidator.py            # 阈值触发后台记忆沉淀与滚动压缩（支持独立 consolidator_llm 分槽）
│       ├── commands.py                # tools.json 解析、CommandToolWrapper 与五大插槽/双模型装配
│       ├── app.py                     # AssistantApp 核心编排（stream_chat 单一主流程、会话串行锁与前缀冻结）
│       └── cli.py                     # 终端 REPL（/status /memory /search /refresh、setup、compile-config、流式思维链展示）
├── scripts/
│   ├── demo.py                        # 零配置离线 ReAct 演示与 --ask 单句实测脚本
│   ├── web_chat.py                    # 零依赖本地 HTTP/SSE 服务（ThreadingHTTPServer + Host/Origin/Content-Type 防护）
│   └── web_ui.html                    # 纯 HTML/JS 流式教学界面（可折叠思维链、XSS 安全 Markdown、遥测与模型配置弹窗）
├── evals/                             # 顶层统一评测包（与 tests/、plugins/ 平级，统一 `python -m evals`）
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                         # --mode search / agent / memory / download 与 --model / --judge-model / --workers
│   ├── common.py                      # JSONL 加载、金标引文校验、Git SHA 与路径隔离写盘
│   ├── search_eval.py                 # 离线检索评测器（Recall / MRR / Quote / 负例干净度）
│   ├── agent_eval.py                  # 单轮隔离 Agent 评测器
│   ├── memory_eval.py                 # 多轮记忆生命周期、shared_setup 并发复用与 LLM-as-a-Judge 评测器
│   ├── public_benchmarks.py           # LongMemEval / LoCoMo 公开学术基准下载与分层抽样适配器
│   ├── sample_cases.jsonl             # 40 题 5 分类离线检索与单轮问答案例集
│   ├── memory_scenarios.jsonl         # 16 场景 6 分类多轮记忆生命周期与 KV 缓存场景集
│   ├── sample_data/                   # 8 篇主题 Markdown 与规则样例语料
│   └── public/                        # （gitignore）下载转换的公开学术数据集
├── tests/                             # 确定性单测与架构守卫（CI 每次必跑）
│   ├── test_architecture.py           # 三层依赖漏斗 AST 守卫、类型递进与子进程断层隔离实测
│   ├── test_step1_react.py            # 单层 ReAct 循环、无 MAX_TOOL_ROUNDS 自然终止、两阶段转换边界与重试单测
│   ├── test_step2_tools.py            # 工具白名单、build_tool 与固定命令执行器单测
│   ├── test_plugins.py                # Plugin(ABC) 主板、单次初始化解耦、5 大插槽与自定义 Tool 子类单测
│   ├── test_models_dsl.py             # 模型配置 DSL 编译插件、直写密钥、0600 权限与防泄漏单测
│   ├── test_reasoning_provider.py     # Slot-4 推理模型协议插件与跨轮思维链剥离单测
│   ├── test_fts5_plugin.py            # SQLite FTS5 + BM25 双语检索插件单测
│   ├── test_history_plugin.py         # 只读增量 FTS5 跨会话历史检索插件单测
│   ├── test_telemetry_plugin.py       # Slot-5 运行期遥测插件、有界窗口与并发单测
│   ├── test_shell_plugin.py           # 受控白名单 Shell 插件与 38 条红队绕过向量单测
│   ├── test_web_plugin.py             # 联网搜索与防 SSRF 抓网页插件单测
│   ├── test_study_plugin.py           # 学习助手知识卡片与间隔重复调度插件单测（PR #64）
│   ├── test_step3_memory.py           # Markdown 记忆、write_lock、turn_tag 提交/回滚与目录边界单测
│   ├── test_step4_app.py              # SQLite 成对原子 append_turn、断流防护、前缀冻结与编排单测
│   ├── test_streaming.py              # SSE 流式管道、ThinkingDelta/ToolStart 事件与 /api/chat/stream 单测
│   ├── test_cache_metrics.py          # CacheMonitor 有界窗口、MRU 淘汰与线程隔离单测
│   ├── test_consolidator.py           # 自动阈值沉淀与后台上下文压缩单测
│   ├── test_memory_lifecycle.py       # add/merge/replace/remove、superseded_by 与过期归档单测
│   ├── test_toast_offload.py          # TOAST 工具大输出卸载与懒清理单测
│   ├── test_eval.py                   # 检索、Agent 评测与 evals CLI --model/--judge-model 统一装配单测
│   ├── test_memory_eval.py            # 多轮记忆、shared_setup 与 --workers 保序并发单测
│   ├── test_public_benchmarks.py      # LongMemEval / LoCoMo 公开基准适配器单测
│   └── test_web_chat.py               # 本地网页服务、XSS 安全 Markdown 与 Host/Origin/Content-Type 防 CSRF 单测
├── docs/
│   ├── mvp-design.md                  # 三层洋葱微内核与学习助手架构设计文档
│   └── dev-log.md                     # 工程演进、真实踩坑根因、Benchmark 总账与跨模型交接日志
├── AGENTS.md                          # 仓库协作红线、TDD/评测纪律与并行 Review 规范
├── .env.example
├── .gitignore
├── LICENSE
└── pyproject.toml
```

---

## 演进路线与当前状态

- [x] **Step 1 ~ 4 基础闭环**：极简 ReAct 循环、三级白名单与固定命令、纯 Markdown 记忆与 `StateStore` 持久化
- [x] **Pi 洋葱三层重构与五大主板插槽**：`ai ← agent ← assistant` 漏斗守卫（[tests/test_architecture.py](tests/test_architecture.py)）与 5 大强类型子协议全面实装
- [x] **记忆生命周期与 KV Cache 工程化**：`MEMORY.md` 会话冻结、`add`/`merge`/`replace`/`remove` 软退休、`search_chat_history` 只读回溯、有界 `CacheMonitor` 与成对原子 `append_turn`
- [x] **Step 5 零依赖本地 Web/SSE 服务与纵深安全加固**：[scripts/web_chat.py](scripts/web_chat.py) / [scripts/web_ui.html](scripts/web_ui.html) 全链路流式展示 + `Host`/`Origin`/`Content-Type` 防 CSRF + Shell/Web 防逃逸边界
- [x] **四层可复现评测闭环**：自建 40 题/16 场景 + 公开 LongMemEval & LoCoMo 适配器，支持 `--workers` 并发与独立 `--judge-model`
- [x] **个人学习助手叠加层**：流式可折叠思维链推导视图、前后台 `consolidator` 双模型分槽、`study_tools.py` 纯 Markdown 间隔复习卡片
- [ ] **可选增强**：`plugins/hybrid_rrf_retriever.py`（仅在纯词面检索遇到瓶颈时作为可选 `RetrieverProtocol` 插件引入，保持核心零依赖）

## License

[MIT](LICENSE)
