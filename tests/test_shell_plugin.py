"""受控白名单 Shell 插件（plugins/shell_tools.py）单元测试（TDD）。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from personal_assistant.assistant.commands import ToolsConfig
from plugins.shell_tools import ShellPolicy, execute_safe_command


# 测试插件能通过 config/tools.json 自动发现注册，并正常执行白名单内的只读命令
def test_shell_plugin_loads_via_tools_config_and_executes_allowed_commands(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = ToolsConfig.load(repo_root / "config" / "tools.json", project_root=repo_root)
    assert "run_command" in cfg.enabled_tools
    registry = cfg.build_registry()
    assert "run_command" in registry.names

    (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("ls", "wc", "head", "cat", "pwd", "find", "git"),
    )

    res_ls = execute_safe_command("ls", policy)
    assert "退出码：0" in res_ls
    assert "hello.txt" in res_ls

    res_wc = execute_safe_command("wc -l hello.txt", policy)
    assert "退出码：0" in res_wc
    assert "3" in res_wc

    res_head = execute_safe_command("head -n 2 hello.txt", policy)
    assert "line1" in res_head
    assert "line2" in res_head
    assert "line3" not in res_head


# 测试拦截白名单外程序、带路径可执行文件以及任何管道/分号/重定向/命令替换语法
def test_shell_plugin_blocks_disallowed_executables_and_shell_metachars(tmp_path: Path) -> None:
    policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("ls", "cat", "wc"),
    )
    disallowed = [
        "",
        "rm -rf .",
        "sh -c id",
        "bash -c ls",
        "curl https://example.com",
        "/bin/ls",
        "./run.sh",
        "ls; pwd",
        "ls | wc -l",
        "ls && ls",
        "ls || ls",
        "cat a.txt > out.txt",
        "cat < a.txt",
        "ls $(whoami)",
        "ls `whoami`",
        "ls\npwd",
    ]
    for cmd in disallowed:
        out = execute_safe_command(cmd, policy)
        assert out.startswith("错误："), f"应被拦截但未拦截：{cmd!r} -> {out!r}"


# 测试拦截敏感路径（含软链接逃逸与绝对路径）、目录穿越以及 git/find 危险子命令或参数
def test_shell_plugin_blocks_sensitive_paths_traversal_and_dangerous_flags(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=123\n", encoding="utf-8")
    (tmp_path / "ok.md").write_text("# OK\n", encoding="utf-8")
    os.symlink(tmp_path / ".env", tmp_path / "sneaky.md")

    policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("cat", "ls", "find", "git", "grep", "rg"),
    )

    blocked_cases = [
        "cat .env",
        "cat ./.env",
        "cat sneaky.md",
        "cat /etc/passwd",
        "cat ../secret.txt",
        "ls ..",
        "cat data/state/chat_history.sqlite3",
        "git push",
        "git reset --hard",
        "git clean -fd",
        "git config --list",
        "git -c core.sshCommand=id status",
        "git log --upload-pack=evil",
        "find . -delete",
        "find . -exec rm {} ;",
        "find . -execdir id +",
        "find . -ok ls {} ;",
        "rg --no-ignore SECRET .",
        "rg --hidden SECRET .",
    ]
    for cmd in blocked_cases:
        out = execute_safe_command(cmd, policy)
        assert out.startswith("错误："), f"应被拦截但未拦截：{cmd!r} -> {out!r}"
        assert "SECRET=123" not in out

    # grep 递归搜索当前目录时自动排除 .env，绝不泄漏密钥但可搜到正常文件
    grep_out = execute_safe_command("grep -r SECRET .", policy)
    assert "SECRET=123" not in grep_out
    grep_ok = execute_safe_command("grep -r OK .", policy)
    assert "ok.md" in grep_ok

    # 正常安全的 find 和 cat ok.md 应放行
    ok_find = execute_safe_command("find . -maxdepth 2 -name '*.md'", policy)
    assert "退出码：0" in ok_find
    assert "ok.md" in ok_find


# 测试环境变量脱敏（不泄漏 API Key）、输出截断与命令超时保护
def test_shell_plugin_sanitizes_env_truncates_output_and_handles_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-super-secret-value")

    env_policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("env", "yes", "sleep", "cat"),
        timeout=0.3,
        max_output_bytes=120,
    )

    # 环境变量白名单必须剥离父进程中的 LLM_API_KEY
    env_out = execute_safe_command("env", env_policy)
    assert "退出码：0" in env_out
    assert "sk-super-secret-value" not in env_out

    # 超长输出按字节上限安全截断
    (tmp_path / "big.txt").write_text("A" * 500 + "\n", encoding="utf-8")
    trunc_out = execute_safe_command("cat big.txt", env_policy)
    assert "输出已截断" in trunc_out

    # 超时命令被终止并友好报错
    timeout_out = execute_safe_command("sleep 2", env_policy)
    assert "超时" in timeout_out


# 测试从 config/tools.json 的 enabled_tools 移除 run_command 后，插件立即彻底关闭
def test_shell_plugin_disabled_when_removed_from_enabled_tools(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_file = tmp_path / "tools.json"
    cfg_file.write_text(
        json.dumps(
            {
                "enabled_tools": ["get_current_time"],
                "plugins": ["plugins"],
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = ToolsConfig.load(cfg_file, project_root=repo_root)
    registry = cfg.build_registry()
    assert "run_command" not in registry.names
    assert "不在白名单中" in registry.execute("run_command", '{"command": "ls"}')
