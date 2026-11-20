# My Helper v2

一个**零第三方运行依赖**的纯 Python 个人 AI Agent 最小 MVP。不套用黑盒框架，从最基础的 ReAct 循环开始，一步步搭起白名单工具沙箱、插件加载器、纯 Markdown 记忆系统与轻量评测模块。

> 详细架构设计见 [docs/mvp-design.md](docs/mvp-design.md)；完整的工程演进、排坑复盘、Benchmark 演进总账与跨模型交接路线图见 [docs/dev-log.md](docs/dev-log.md) 与 [AGENTS.md](AGENTS.md)。

## 核心原则

- **纯 Python 标准库实现**：生产环境不引入任何第三方库（仅用 `urllib`、`sqlite3`、`json`、`pathlib` 等标准库），代码一眼看穿。
- **经典 ReAct 闭环**：基于 OpenAI 兼容 Chat Completions 原生 Function Calling 实现 `Reason -> Act -> Observe` 循环，并将工具参数/执行异常作为 Observation 回传模型自愈。
- **纯白名单与插件化工具**：
  - 记忆采用“目录索引 + 独立主题/事实文件”的纯 Markdown 设计：`MEMORY.md` 仅作限长索引并在会话内冻结前缀（配合追加式多轮历史，最大化 KV Cache 命中率）；随时间增长的事实记入 `facts.md` 按需检索。
  - 核心只保留必要的 Markdown/文件内置工具；其余自定义工具全部写在可自由编辑的 [plugins/custom_tools.py](plugins/custom_tools.py)（或 `plugins/*.py`）中，用 `@tool(...)` 声明并在 `config/tools.json` 白名单中勾选即可热插拔。

## 快速开始

需要 Python 3.12+ 与 [uv](https://docs.astral.sh/uv/)。

```bash
# 1. 初始化虚拟环境并安装开发测试依赖（pytest）
uv sync

# 2. 运行单元测试（验证代码逻辑，不消耗 API Token）
uv run pytest -v

# 3. 运行离线检索评测（40 题内置分类型样本，零 Token；支持 --retriever linear / fts5_bm25 做 A/B 对比）
uv run python -m personal_assistant.eval --mode search --retriever linear
uv run python -m personal_assistant.eval --mode search --retriever fts5_bm25

# 4. 下载并运行公开学术基准（LongMemEval ICLR 2025 + LoCoMo ACL 2024，自动存放在 gitignore 的 evals/public/）
uv run python -m personal_assistant.eval --mode download --dataset all
# 零 Token 跑 200 题公开 LoCoMo 多会话检索基准（默认自动启用 config/tools.json 的 fts5_bm25 插件）
uv run python -m personal_assistant.eval --mode search \
  --data-dir evals/public/locomo_search_data \
  --cases evals/public/locomo_search_cases.jsonl

# 5. 零配置离线演示 ReAct 循环（不用 API Key 即可查看完整调用轨迹）
uv run python scripts/demo.py --show-messages

# 6. 配置环境变量并启动真实对话或端到端评测（单轮 agent / 16 场景 memory / LongMemEval）
cp .env.example .env
# 编辑 .env，设置 LLM_MODE=on、LLM_API_KEY、LLM_BASE_URL 与 LLM_MODEL
uv run python scripts/demo.py --ask "现在几点了？"
uv run python -m personal_assistant.eval --mode agent --output /tmp/eval-agent.json
uv run python -m personal_assistant.eval --mode memory --output /tmp/eval-memory.json
uv run python -m personal_assistant.eval --mode memory \
  --cases evals/public/longmemeval_scenarios.jsonl --max-cases 14 --output /tmp/eval-lme.json
uv run python -m personal_assistant chat
# 或者启动本地网页对话界面（纯标准库，默认 http://127.0.0.1:8765）
uv run python scripts/web_chat.py --open
```

## 评测体系与当前基线

仓库内置两套开箱即复现的评测集，同时通过零依赖适配器接入 **LongMemEval（ICLR 2025，500 题）** 与 **LoCoMo（ACL 2024，1,986 题）** 公开学术数据集（开放题采用与 Agent KV Cache 统计隔离的 `LLM-as-a-Judge` 评分）：

| 评测集 | 类型 | 规模 | 线性基线（`552863d`） | 插件 `fts5_bm25` 改后 | 关键变化 |
| --- | --- | --- | --- | --- | --- |
| `evals/sample_cases.jsonl` | 离线检索（5 分类） | 40 题 | 通过 **80.0%**（32/40），MRR **0.908**，负例干净度 **63.6%** | 通过 **95.0%**（38/40），MRR **0.948**，负例干净度 **100%** | **+15.0pp**，+6 改善 / 0 退步；仅剩纯语义同义改写待向量/RRF |
| LoCoMo Search（ACL 2024） | 公开多会话离线检索 | 200 题抽样 | 通过 **28.5%**，Recall **42.8%**，MRR **0.359**，29.0 ms/题 | 通过 **52.5%**，Recall **68.5%**，MRR **0.622**，9.5 ms/题 | Recall **+25.8pp**，MRR **+0.263**，提速 **~3x**（+51 / -3） |
| `evals/memory_scenarios.jsonl` | 多轮记忆生命周期（6 分类） | 16 场景 | 通过 **81.25%**（13/16），KV Cache 命中 **60.4%** / 前缀复用 **90.4%** | 通过 **81.25%**（13/16），KV Cache 命中 **77.0%** / 前缀复用 **87.9%** | 无检索退步；`temporal` 3/3，待优化替换句旧词残留 |
| LongMemEval（ICLR 2025） | 公开多轮长期记忆（7 分类） | 14 题分层抽样 | 通过 **71.43%**（10/14，LLM Judge），前缀复用 **82.6%** | 未重跑（仅改动检索插件） | 待优化：隐式偏好问答主动检索（`single-session-preference` 0/2） |

## 当前项目结构（Step 4）

```text
my_helper_v2/
├── config/
│   └── tools.json                 # 工具白名单、插件路径、检索后端配置、目录边界与固定命令
├── plugins/
│   ├── custom_tools.py            # 可自由编辑的工具插件文件（@tool 声明即插即用）
│   ├── fts5_retriever.py          # SQLite FTS5 + BM25 检索后端插件（受 tools.json 的 retrieval 段管控）
│   ├── shell_tools.py             # 受控白名单命令插件（run_command，受 tools.json 的 shell 段管控）
│   └── web_tools.py               # 联网搜索与防 SSRF 网页正文提取插件（web_search / fetch_webpage）
├── evals/
│   ├── sample_cases.jsonl         # 40 题 5 分类公开脱敏检索与单轮问答案例集
│   ├── memory_scenarios.jsonl     # 16 场景 6 分类多轮记忆沉淀、冲突替换、过期归档与 KV 缓存场景集
│   ├── sample_data/               # 8 篇主题 Markdown 与规则样例语料
│   └── public/                    # （gitignore）下载转换的 LongMemEval / LoCoMo 公开学术基准
├── docs/
│   ├── mvp-design.md              # 整体设计文档
│   └── dev-log.md                 # 工程演进、真实踩坑根因、评测总账与跨模型交接日志
├── AGENTS.md                      # 仓库协作红线、TDD/评测纪律与跨模型续写协议
├── personal_assistant/            # 严格三层依赖漏斗架构（ai → agent → assistant + eval）
│   ├── __init__.py
│   ├── __main__.py                # `python -m personal_assistant` 入口
│   ├── ai/                        # Layer 1（零内部依赖）：LLM 协议底座，可独立使用
│   │   ├── types.py               # 最小原子契约：ToolSchema、ToolCall、TokenUsage
│   │   ├── client.py              # LLMClient 抽象、OpenAICompatibleLLM 与 Provider 注册分发
│   │   └── cache.py               # KV 缓存观测（服务端 cached_tokens + 本地前缀复用率）
│   ├── agent/                     # Layer 2（仅依赖 ai）：通用 ReAct 引擎，不含业务假设
│   │   ├── tools.py               # Tool(ToolSchema) 类型递进扩展、@tool 与白名单 ToolRegistry
│   │   ├── react.py               # 极简通用 ReAct 核心循环（Reason -> Act -> Observe）
│   │   └── plugins.py             # 统一插件发现器（TOOLS / RETRIEVERS / LLM_PROVIDERS）
│   ├── assistant/                 # Layer 3（个人助手业务层）：记忆、历史、沉淀与入口
│   │   ├── common.py              # 无内部依赖公共底座（REPO_ROOT、load_dotenv、truncate_utf8）
│   │   ├── prompts.py             # 个人助手专属系统提示词组装（从通用 react.py 剥离）
│   │   ├── memory.py              # 核心内置 Markdown 记忆与文件读写工具
│   │   ├── memory_lifecycle.py    # 记忆生命周期（冲突替换 add/replace/remove、过期过滤与归档）
│   │   ├── db.py                  # 线程安全 SQLite 存储（messages 历史 + session_compactions 压缩状态）
│   │   ├── consolidator.py        # 阈值触发后台记忆沉淀与上下文压缩器（标准库 threading）
│   │   ├── commands.py            # tools.json 解析、固定 argv 命令与检索/Provider 插件装配
│   │   ├── app.py                 # AssistantApp 核心编排（串联历史、记忆、工具、后台沉淀与 ReAct）
│   │   └── cli.py                 # 终端交互 REPL（支持 /status、/memory、/search、/refresh）
│   └── eval/                      # 独立评测包（common / search_eval / agent_eval / memory_eval / public_benchmarks / cli）
├── scripts/
│   ├── demo.py                    # 离线演示与单句实测脚本
│   ├── web_chat.py                # 零依赖本地网页对话服务（标准库 ThreadingHTTPServer）
│   └── web_ui.html                # 纯 HTML/JS 极简本地对话页面（含工具轨迹与 KV 缓存显示）
├── tests/                         # 确定性代码单元测试（独立于 evals/，CI 每次必跑）
│   ├── test_architecture.py       # 三层依赖漏斗 AST 守卫、类型递进扩展与子进程独立运行测试
│   ├── test_step1_react.py        # ReAct 循环与客户端单元测试
│   ├── test_step2_tools.py        # 工具白名单与命令执行器单元测试
│   ├── test_plugins.py            # 插件发现、加载与白名单隔离单元测试
│   ├── test_fts5_plugin.py        # SQLite FTS5 + BM25 检索插件单元测试
│   ├── test_shell_plugin.py       # 受控白名单 Shell 插件安全边界单元测试
│   ├── test_web_plugin.py         # 联网搜索与防 SSRF 抓网页插件单元测试
│   ├── test_step3_memory.py       # Markdown 记忆、检索与目录边界单元测试
│   ├── test_step4_app.py          # SQLite 历史、前缀冻结与 AssistantApp 编排单元测试
│   ├── test_cache_metrics.py      # KV 缓存命中与前缀复用率单元测试
│   ├── test_consolidator.py       # 自动阈值沉淀与后台上下文压缩单元测试
│   ├── test_memory_lifecycle.py   # 记忆冲突替换与临时事实过期归档单元测试
│   ├── test_eval.py               # 检索与单轮 Agent 评测器单元测试
│   ├── test_memory_eval.py        # 多轮记忆与生命周期评测器单元测试
│   ├── test_public_benchmarks.py  # LongMemEval / LoCoMo 公开基准适配器单元测试
│   └── test_web_chat.py           # 本地网页对话服务接口单元测试
├── .env.example
├── .gitignore
├── LICENSE
└── pyproject.toml
```

## 开发路线图

- [x] **Step 1**：最小 ReAct 核心循环（`llm.py`, `tools.py`, `react.py`, `cli.py`）与终端 REPL
- [x] **Step 2**：白名单工具系统与固定 `argv` 命令模板（`config/tools.json`, `tools.py`, `commands.py`, `plugins.py`）
- [x] **Step 3**：纯 Markdown 记忆与安全文件读写工具（`memory.py`）
- [x] **Eval**：离线检索与端到端 Agent 评测包（`eval/`）
- [x] **Step 4**：SQLite 会话历史与核心编排（`db.py`, `app.py`）
- [ ] **Step 5**：轻量本地 HTTP API 服务（`server.py`，可选）

## License

[MIT](LICENSE)
