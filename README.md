# My Helper v2

一个**零第三方运行依赖**的纯 Python 个人 AI Agent 最小 MVP。不套用黑盒框架，从最基础的 ReAct 循环开始，一步步搭起白名单工具沙箱、插件加载器、纯 Markdown 记忆系统与轻量评测模块。

> 详细架构与演进规划见 [docs/mvp-design.md](docs/mvp-design.md)。

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

# 3. 运行离线检索评测（使用公开样本，零 Token）
uv run python -m personal_assistant.eval --mode search

# 4. 零配置离线演示 ReAct 循环（不用 API Key 即可查看完整调用轨迹）
uv run python scripts/demo.py --show-messages

# 5. 配置环境变量并启动真实对话或端到端评测（单轮 agent / 多轮记忆生命周期 memory）
cp .env.example .env
# 编辑 .env，设置 LLM_MODE=on、LLM_API_KEY、LLM_BASE_URL 与 LLM_MODEL
uv run python scripts/demo.py --ask "现在几点了？"
uv run python -m personal_assistant.eval --mode agent --output /tmp/eval-agent.json
uv run python -m personal_assistant.eval --mode memory --output /tmp/eval-memory.json
uv run python -m personal_assistant chat
```

## 当前项目结构（Step 4）

```text
my_helper_v2/
├── config/
│   └── tools.json                 # 工具白名单、插件路径、目录边界与固定命令配置
├── plugins/
│   └── custom_tools.py            # 可自由编辑的工具插件文件（@tool 声明即插即用）
├── evals/
│   ├── sample_cases.jsonl         # 公开脱敏检索与单轮问答案例集
│   ├── memory_scenarios.jsonl     # 多轮记忆沉淀、冲突替换、过期归档与 KV 缓存场景集
│   └── sample_data/               # 公开评测样例 Markdown 文件
├── docs/
│   └── mvp-design.md              # 整体设计文档
├── personal_assistant/
│   ├── __init__.py
│   ├── __main__.py                # `python -m personal_assistant` 入口
│   ├── llm.py                     # 零依赖 OpenAI 兼容客户端（含 Token 与缓存 usage 解析）
│   ├── cache_metrics.py           # KV 缓存观测（服务端命中率 + 本地前缀复用率）
│   ├── tools.py                   # Tool、@tool 装饰器与 ToolRegistry 白名单注册表
│   ├── plugins.py                 # 动态加载 plugins/*.py 插件工具
│   ├── commands.py                # 固定 argv 命令模板加载与 ToolsConfig 装配
│   ├── memory.py                  # 核心内置 Markdown 记忆与文件读写工具
│   ├── memory_lifecycle.py        # 记忆生命周期（冲突替换 add/replace/remove、过期过滤与归档）
│   ├── db.py                      # 线程安全 SQLite 存储（messages 历史 + session_compactions 压缩状态）
│   ├── consolidator.py            # 阈值触发后台记忆沉淀与上下文压缩器（标准库 threading）
│   ├── app.py                     # AssistantApp 核心编排（串联历史、记忆、工具、后台沉淀与 ReAct）
│   ├── eval/                      # 评测包（common / search_eval / agent_eval / memory_eval / cli）
│   ├── react.py                   # 极简 ReAct 核心循环（Reason -> Act -> Observe）
│   └── cli.py                     # 终端交互 REPL（支持 /status、/memory、/search、/refresh）
├── scripts/
│   └── demo.py                    # 离线演示与单句实测脚本
├── tests/                         # 确定性代码单元测试（独立于 evals/，CI 每次必跑）
│   ├── test_step1_react.py        # ReAct 循环与客户端单元测试
│   ├── test_step2_tools.py        # 工具白名单与命令执行器单元测试
│   ├── test_plugins.py            # 插件发现、加载与白名单隔离单元测试
│   ├── test_step3_memory.py       # Markdown 记忆、检索与目录边界单元测试
│   ├── test_step4_app.py          # SQLite 历史、前缀冻结与 AssistantApp 编排单元测试
│   ├── test_cache_metrics.py      # KV 缓存命中与前缀复用率单元测试
│   ├── test_consolidator.py       # 自动阈值沉淀与后台上下文压缩单元测试
│   ├── test_memory_lifecycle.py   # 记忆冲突替换与临时事实过期归档单元测试
│   ├── test_eval.py               # 检索与单轮 Agent 评测器单元测试
│   └── test_memory_eval.py        # 多轮记忆与生命周期评测器单元测试
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
