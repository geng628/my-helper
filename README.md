# My Helper v2

一个**零第三方运行依赖**的纯 Python 个人 AI Agent 最小 MVP。不套用黑盒框架，从最基础的 ReAct 循环开始，一步步搭起白名单工具沙箱、纯 Markdown 记忆系统与本地/Telegram 接入层。

> 详细架构与演进规划见 [docs/mvp-design.md](docs/mvp-design.md)。

## 核心原则

- **纯 Python 标准库实现**：生产环境不引入任何第三方库（仅用 `urllib`、`sqlite3`、`json`、`pathlib` 等标准库），代码一眼看穿。
- **经典 ReAct 闭环**：基于 OpenAI 兼容 Chat Completions 原生 Function Calling 实现 `Reason -> Act -> Observe` 循环，并将工具参数/执行异常作为 Observation 回传模型自愈。
- **纯白名单与透明记忆**：
  - 记忆直接读写明文 Markdown（`MEMORY.md` 与 `rules/*.md`），不做复杂的隐式编译与向量黑盒。
  - 工具严格采用白名单机制（`config/tools.json` 管控 `enabled_tools`、允许读写目录与固定 `argv` 命令模板），白名单外直接拒绝。

## 快速开始

需要 Python 3.12+ 与 [uv](https://docs.astral.sh/uv/)。

```bash
# 1. 初始化虚拟环境并安装开发测试依赖（pytest）
uv sync

# 2. 运行单元测试（验证 ReAct 核心循环，不消耗 API Token）
uv run pytest -v

# 3. 零配置离线演示 ReAct 循环（不用 API Key 即可查看完整调用轨迹）
uv run python scripts/demo.py --show-messages

# 4. 配置环境变量并启动真实对话（单次测试或终端交互 REPL）
cp .env.example .env
# 编辑 .env，设置 LLM_MODE=on、LLM_API_KEY、LLM_BASE_URL 与 LLM_MODEL
uv run python scripts/demo.py --ask "现在几点了？"
uv run python -m personal_assistant chat
```

## 当前项目结构（Step 2）

```text
my_helper_v2/
├── config/
│   └── tools.json                 # 工具、目录与固定命令白名单配置
├── docs/
│   └── mvp-design.md              # 整体设计文档
├── personal_assistant/
│   ├── __init__.py
│   ├── __main__.py                # `python -m personal_assistant` 入口
│   ├── llm.py                     # 零依赖 OpenAI 兼容客户端
│   ├── tools.py                   # Tool 与 ToolRegistry 白名单注册表
│   ├── commands.py                # 固定 argv 命令模板加载与安全执行器
│   ├── react.py                   # 极简 ReAct 核心循环（Reason -> Act -> Observe）
│   └── cli.py                     # 终端交互 REPL（加载白名单工具）
├── scripts/
│   └── demo.py                    # 离线演示与单句实测脚本
├── tests/
│   ├── test_step1_react.py        # ReAct 循环与客户端单元测试
│   └── test_step2_tools.py        # 工具白名单与命令执行器单元测试
├── .env.example
├── .gitignore
├── LICENSE
└── pyproject.toml
```

## 开发路线图

- [x] **Step 1**：最小 ReAct 核心循环（`llm.py`, `tools.py`, `react.py`, `cli.py`）与终端 REPL
- [x] **Step 2**：白名单工具系统与固定 `argv` 命令模板（`config/tools.json`, `tools.py`, `commands.py`）
- [ ] **Step 3**：纯 Markdown 记忆与笔记直读检索（`memory.py`）
- [ ] **Step 4**：SQLite 会话历史与核心编排（`db.py`, `app.py`）
- [ ] **Step 5**：纯 Python Telegram Long Polling 与轻量 HTTP 服务（`telegram_bot.py`, `server.py`）

## License

[MIT](LICENSE)
