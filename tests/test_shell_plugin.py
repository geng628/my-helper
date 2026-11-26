"""受控白名单 Shell 插件（plugins/tools/shell_tools.py）单元测试（TDD）。"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.memory import MemoryStore
from plugins.tools.shell_tools import ShellPolicy, execute_safe_command


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
        assert "找不到可执行程序" not in out


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
        assert "找不到可执行程序" not in out, f"不能因缺少命令假绿：{cmd!r} -> {out!r}"
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
        offload_dir=tmp_path / "workspace" / "tool_outputs",
    )

    # 环境变量白名单必须剥离父进程中的 LLM_API_KEY
    env_out = execute_safe_command("env", env_policy)
    assert "退出码：0" in env_out
    assert "sk-super-secret-value" not in env_out

    # 超长输出按字节上限安全截断并落盘
    (tmp_path / "big.txt").write_text("A" * 500 + "\n", encoding="utf-8")
    trunc_out = execute_safe_command("cat big.txt", env_policy)
    assert "输出已截断" in trunc_out
    assert "data/workspace/tool_outputs/" in trunc_out

    # 超时命令被终止并友好报错
    timeout_out = execute_safe_command("sleep 2", env_policy)
    assert "超时" in timeout_out


# 测试 run_command 超长输出走 Head+Tail TOAST 卸载：主输出保留头尾汇总，中间省略内容可按行回查
def test_shell_plugin_offloads_long_output_with_head_tail_preview(tmp_path: Path) -> None:
    offload_dir = tmp_path / "workspace" / "tool_outputs"
    log_lines = ["=== test session starts ===", "collecting ... collected 80 items"]
    log_lines.extend(f"trace detail line {idx:03d}: intermediate assertion context_{idx}" for idx in range(1, 55))
    log_lines.append("FAILED tests/test_core.py::test_edge_case - AssertionError: mismatch")
    log_lines.append("=== 1 failed, 79 passed in 1.42s ===")
    (tmp_path / "pytest_out.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("cat",),
        max_output_bytes=520,
        offload_dir=offload_dir,
    )
    out = execute_safe_command("cat pytest_out.txt", policy)
    assert "退出码：0" in out
    assert "=== test session starts ===" in out
    assert "FAILED tests/test_core.py::test_edge_case" in out
    assert "=== 1 failed, 79 passed in 1.42s ===" in out
    assert "输出已截断" in out
    assert "trace detail line 025" not in out

    omit_match = re.search(r"中间省略第 (\d+)-(\d+) 行", out)
    path_match = re.search(r"(data/workspace/tool_outputs/[A-Za-z0-9._-]+\.txt)", out)
    assert omit_match is not None and path_match is not None

    memory = MemoryStore(tmp_path)
    middle = memory.read_markdown(
        path_match.group(1),
        start_line=int(omit_match.group(1)),
        end_line=int(omit_match.group(2)),
    )
    assert "trace detail line 025" in middle

    # 短输出零落盘
    short_dir = tmp_path / "short_out"
    short_policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("cat",),
        max_output_bytes=4096,
        offload_dir=short_dir,
    )
    (tmp_path / "tiny.txt").write_text("ok\n", encoding="utf-8")
    tiny_out = execute_safe_command("cat tiny.txt", short_policy)
    assert tiny_out == "退出码：0\nok"
    assert not short_dir.exists()


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


# 测试在软链接虚拟环境（如 uv .venv）中能定位 .venv/bin/pytest，不因 .resolve() 跳出 venv
def test_shell_plugin_finds_pytest_inside_symlinked_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. 构造与 uv 一致的软链接虚拟环境（bin/python -> 真实解释器，bin/pytest 仅存在于该 venv）
    venv_bin = tmp_path / "fake_venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / "python"
    os.symlink(Path(sys.executable).resolve(), fake_python)
    fake_pytest = venv_bin / "pytest"
    fake_pytest.write_text("#!/bin/sh\necho fake-venv-pytest-ok\n", encoding="utf-8")
    fake_pytest.chmod(0o755)

    policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("pytest",),
        offload_dir=tmp_path / "workspace" / "tool_outputs",
    )
    # 缺少真实 tests/ 目录时，除 --version/-V 外任何 pytest 调用都必须被拦截
    no_tests_out = execute_safe_command("pytest", policy)
    assert no_tests_out.startswith("错误：")
    assert "找不到可执行程序" not in no_tests_out

    (tmp_path / "tests").mkdir()
    with monkeypatch.context() as mp:
        mp.setattr(sys, "executable", str(fake_python))
        out_fake = execute_safe_command("pytest", policy)
    assert "退出码：0" in out_fake
    assert "fake-venv-pytest-ok" in out_fake

    # 2. 真实当前 .venv 环境下也能直接定位并运行 pytest --version / -V
    for ver_cmd in ("pytest --version", "pytest -V"):
        out_real = execute_safe_command(ver_cmd, policy)
        assert "退出码：0" in out_real
        assert "pytest" in out_real.lower()


# 测试 run_command 执行使用 tmp_path 的 pytest 时，TMPDIR 不会指向工作区根目录、不遗留 pytest-of-* 脏目录
def test_shell_plugin_pytest_tmpdir_does_not_pollute_cwd(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    sub_test = tests_dir / "test_mini_tmp.py"
    sub_test.write_text(
        "def test_uses_tmp_path(tmp_path):\n"
        "    p = tmp_path / 'a.txt'\n"
        "    p.write_text('ok')\n"
        "    assert p.read_text() == 'ok'\n",
        encoding="utf-8",
    )
    policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("pytest",),
        offload_dir=tmp_path / "workspace" / "tool_outputs",
    )
    out = execute_safe_command("pytest -q tests/test_mini_tmp.py", policy)
    assert "退出码：0" in out
    leftover = [p.name for p in tmp_path.iterdir() if p.name.startswith("pytest-of-")]
    assert leftover == [], f"pytest 临时目录污染了工作区 cwd：{leftover}"


# 测试阻断 write_file + pytest 两步 RCE 链：限制 pytest 目标位于 tests/、注入 --ignore=data/offload 并拦截高危参数
def test_shell_plugin_blocks_pytest_rce_chain_and_dangerous_flags(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text(
        "def test_pass():\n"
        "    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / "data" / "workspace"
    workspace_dir.mkdir(parents=True)
    custom_workspace = tmp_path / "workspace"
    custom_workspace.mkdir(parents=True)
    pwn_marker = tmp_path / "pwned.txt"
    for target_dir in (workspace_dir, custom_workspace):
        (target_dir / "test_poc.py").write_text(
            f"from pathlib import Path\n"
            f"Path({str(pwn_marker)!r}).write_text('pwned', encoding='utf-8')\n"
            f"def test_evil():\n"
            f"    assert True\n",
            encoding="utf-8",
        )
        (target_dir / "conftest.py").write_text(
            f"from pathlib import Path\n"
            f"Path({str(pwn_marker)!r}).write_text('conftest_pwned', encoding='utf-8')\n",
            encoding="utf-8",
        )
    (tmp_path / "other_test.py").write_text("def test_other(): assert True\n", encoding="utf-8")

    policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("pytest",),
        offload_dir=custom_workspace / "tool_outputs",
    )

    blocked_cmds = [
        "pytest data/workspace/test_poc.py",
        "pytest data/workspace/test_poc.py::test_evil",
        "pytest ./data/workspace/test_poc.py",
        "pytest tests/../data/workspace/test_poc.py",
        "pytest workspace/test_poc.py",
        "pytest other_test.py",
        "pytest .",
        "pytest -p my_plugin",
        "pytest -pno:cacheprovider",
        "pytest -vp evil",
        "pytest -o cache_dir=data/workspace",
        "pytest -ocache_dir=data/workspace",
        "pytest --override-ini=python_files=*.py",
        "pytest -c data/workspace/evil.ini",
        "pytest -cevil.ini",
        "pytest --config-file=evil.ini",
        "pytest --rootdir=data/workspace",
        "pytest --rootdir data/workspace",
        "pytest --confcutdir=data/workspace",
        "pytest --import-mode=importlib",
        "pytest --basetemp=data/workspace/bt",
        "pytest --noconftest",
        "pytest --pyargs os",
        "pytest --fixtures",
        "pytest --funcargs",
        "pytest --markers",
        "pytest --help",
        "pytest -h",
        "pytest -r a",
        "pytest -V tests/",
        "pytest -- data/workspace/test_poc.py",
    ]
    for cmd in blocked_cmds:
        out = execute_safe_command(cmd, policy)
        assert out.startswith("错误："), f"应拦截高危 pytest 命令：{cmd!r} -> {out!r}"
        assert "找不到可执行程序" not in out
    assert not pwn_marker.exists()

    # 无参 pytest 自动注入 --ignore=data、--ignore=workspace 并收敛到 tests/
    bare_out = execute_safe_command("pytest -q", policy)
    assert "退出码：0" in bare_out
    assert "1 passed" in bare_out
    assert not pwn_marker.exists()

    # 合法只读参数与 tests/ 内 ::nodeid 路径正常放行
    node_out = execute_safe_command("pytest -q tests/test_ok.py::test_pass", policy)
    assert "退出码：0" in node_out
    assert "1 passed" in node_out

    opts_out = execute_safe_command(
        "pytest -vl -k 'test_pass' -s --ff --verbose --quiet --co --no-summary --disable-warnings --tb=short --maxfail=1 --durations=5 tests/",
        policy,
    )
    assert "退出码：0" in opts_out

    # 缺失 tests/ 或 tests 为软链时，除 --version/-V 外一律拒绝
    no_tests_root = tmp_path / "no_tests"
    no_tests_root.mkdir()
    (no_tests_root / "test_leak.py").write_text("def test_x(): assert True\n", encoding="utf-8")
    no_tests_policy = ShellPolicy(project_root=no_tests_root, cwd=no_tests_root, allowed_executables=("pytest",))
    assert execute_safe_command("pytest", no_tests_policy).startswith("错误：")
    assert "退出码：0" in execute_safe_command("pytest --version", no_tests_policy)

    sym_root = tmp_path / "sym_tests_root"
    sym_root.mkdir()
    (sym_root / "data").mkdir()
    os.symlink(sym_root / "data", sym_root / "tests")
    sym_policy = ShellPolicy(project_root=sym_root, cwd=sym_root, allowed_executables=("pytest",))
    assert execute_safe_command("pytest -q", sym_policy).startswith("错误：")


# 测试阻断短选项连写、git 全局选项走私与 pathspec 绕过、grep --include/-R、大小写绕过、rg glob 与正则模式不误伤
def test_shell_plugin_blocks_attached_short_flags_git_colon_refs_and_env_variants(
    tmp_path: Path,
) -> None:
    from plugins.tools.shell_tools import _augment_safe_argv

    (tmp_path / ".env").write_text("SECRET=root_secret\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("SECRET=local_secret\n", encoding="utf-8")
    (tmp_path / ".env.production").write_text("SECRET=prod_secret\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("EXAMPLE_KEY=demo_value\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.json").write_text('{"api_key":"json_secret"}\n', encoding="utf-8")
    (tmp_path / "config" / "models.dsl").write_text("api_key dsl_secret\n", encoding="utf-8")
    (tmp_path / "data" / "state").mkdir(parents=True)
    (tmp_path / "data" / "state" / "db.txt").write_text("STATE_SECRET\n", encoding="utf-8")
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "id_rsa").write_text("SSH_SECRET\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("endpoint /api/chat and ../AGENTS.md\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("SECRET = 'safe_code'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello .env world\n", encoding="utf-8")
    os.symlink(tmp_path / ".env.local", tmp_path / "sneaky_local.txt")

    policy = ShellPolicy(
        project_root=tmp_path,
        cwd=tmp_path,
        allowed_executables=("cat", "ls", "grep", "rg", "git", "date"),
    )

    blocked_cmds = [
        "git -C/tmp status",
        "git -C /tmp status",
        "git -Cstate status",
        "git -c core.sshCommand=id status",
        "git -ccore.sshCommand=id status",
        "git --namespace status config core.fsmonitor evil",
        "git --git-dir=.git status",
        "git diff --no-index .env README.md",
        "git diff --ext-diff",
        "git show --textconv HEAD:README.md",
        "git log -p ':(glob).env'",
        "git log -p '.g*/config'",
        "git diff HEAD -- 'data/s*/db.txt'",
        "git diff HEAD -- ':^*.env'",
        "git diff HEAD -- ':!README.md'",
        "git diff HEAD -- '.e*'",
        "git diff HEAD -- '*env*'",
        "git diff HEAD -- '.[e]nv'",
        "git show HEAD:.env",
        "git show HEAD:.env.local",
        "git show :0:.env",
        "git show HEAD:.git/config",
        "git diff HEAD -- .env",
        "grep /etc/passwd -e root",
        "grep .env -e SECRET",
        "grep -f.env README.md",
        "grep -nf.env README.md",
        "grep -f .env README.md",
        "grep --file=.env README.md",
        "grep --file .env README.md",
        "grep --exclude-from=.env README.md",
        "grep -R SECRET .",
        "grep --dereference-recursive SECRET .",
        "grep -r --include='*env*' SECRET .",
        "grep -r --include='*.json' SECRET .",
        "grep -e '/api/chat' ../secret.txt",
        "rg /etc --files",
        "rg -f.env",
        "rg -nf.env",
        "rg -f .env",
        "rg --file=.env",
        "rg -n. SECRET .",
        "rg -nu SECRET .",
        "rg -g .env SECRET .",
        "rg -g '.*' SECRET .",
        "rg -g './.env' SECRET .",
        "rg -g '*env*' SECRET .",
        "rg --glob='*.env' SECRET .",
        "rg --files .env",
        "cat .git/config",
        "cat .GIT/config",
        "ls .git",
        "cat .env.local",
        "cat .ENV",
        "cat .ENV.local",
        "cat .env.production",
        "cat config/MODELS.JSON",
        "cat config/MODELS.DSL",
        "cat data/STATE/db.txt",
        "cat .SSH/id_rsa",
        "cat sneaky_local.txt",
        "date -f.env",
        "date -uf .env",
        "cat -f.env",
    ]
    for cmd in blocked_cmds:
        out = execute_safe_command(cmd, policy)
        assert out.startswith("错误："), f"应拦截绕过命令：{cmd!r} -> {out!r}"
        assert "找不到可执行程序" not in out, f"不能因缺少命令假绿：{cmd!r} -> {out!r}"
        for secret in ("root_secret", "local_secret", "prod_secret", "json_secret", "dsl_secret", "STATE_SECRET", "SSH_SECRET"):
            assert secret not in out

    # grep 递归检索时排除置后且自动过滤 .env.* 与 .git，安全 --include='*.py' 正常放行
    grep_secret = execute_safe_command("grep -r SECRET .", policy)
    assert "root_secret" not in grep_secret
    assert "local_secret" not in grep_secret
    assert "prod_secret" not in grep_secret

    grep_py = execute_safe_command("grep -r --include='*.py' safe_code .", policy)
    assert "退出码：0" in grep_py
    assert "ok.py" in grep_py

    # grep 未用 -e 时第 1 个位置参数是搜索模式而非路径，不应误杀包含 / 或 .. 的正则模式
    grep_pat_slash = execute_safe_command("grep -rn '/api/chat' docs/", policy)
    assert "退出码：0" in grep_pat_slash
    assert "guide.md" in grep_pat_slash

    grep_pat_dotdot = execute_safe_command("grep -rn '../AGENTS.md' docs/", policy)
    assert "退出码：0" in grep_pat_dotdot
    assert "guide.md" in grep_pat_dotdot

    # .env.example 必须正常可读，grep -e 模式与 git --no-pager 正常放行
    cat_example = execute_safe_command("cat .env.example", policy)
    assert "退出码：0" in cat_example
    assert "EXAMPLE_KEY=demo_value" in cat_example

    grep_example = execute_safe_command("grep EXAMPLE_KEY .env.example", policy)
    assert "退出码：0" in grep_example
    assert "EXAMPLE_KEY=demo_value" in grep_example

    grep_pattern = execute_safe_command("grep -e .env README.md", policy)
    assert "退出码：0" in grep_pattern
    assert "hello .env world" in grep_pattern

    # 验证 _augment_safe_argv 为 grep 与 rg 在所有用户选项及后置参数后（或 -- 前）追加排除规则
    aug_grep = _augment_safe_argv(["grep", "-r", "--include=*.py", "safe_code", "."], policy)
    assert aug_grep.index("--include=*.py") < aug_grep.index("safe_code") < aug_grep.index("--exclude=.env.*")

    aug_grep_post = _augment_safe_argv(["grep", "-r", "safe_code", ".", "--include=*.py"], policy)
    assert aug_grep_post.index("--include=*.py") < aug_grep_post.index("--exclude=.env.*")

    aug_grep_dd = _augment_safe_argv(["grep", "-r", "--include=*.py", "--", "safe_code", "."], policy)
    assert aug_grep_dd.index("--include=*.py") < aug_grep_dd.index("--exclude=.env.*") < aug_grep_dd.index("--")

    aug_rg = _augment_safe_argv(["rg", "-n", "foo", "."], policy)
    assert aug_rg.index("-n") < aug_rg.index("foo") < aug_rg.index("--glob=!.env.*")
    assert "--glob=!.git" in aug_rg
    assert "--glob=!**/.git/**" in aug_rg

    # 验证 wc --files0-from 与安全路径通配符 tests/*.py 行为
    wc_policy = ShellPolicy(project_root=tmp_path, cwd=tmp_path, allowed_executables=("wc",))
    wc_out = execute_safe_command("wc --files0-from .env", wc_policy)
    assert wc_out.startswith("错误：") and "找不到可执行程序" not in wc_out

    # 验证 config/tools.json 的 shell.reject_segments 已同步包含 .git，且合法 git --no-pager / 安全通配符放行
    repo_root = Path(__file__).resolve().parent.parent
    raw_tools = json.loads((repo_root / "config" / "tools.json").read_text(encoding="utf-8"))
    assert ".git" in raw_tools["shell"]["reject_segments"]

    repo_policy = ShellPolicy.from_config(repo_root / "config" / "tools.json")
    no_pager_out = execute_safe_command("git --no-pager status --short", repo_policy)
    assert "退出码：0" in no_pager_out
    safe_glob_out = execute_safe_command("git diff HEAD -- '*.py'", repo_policy)
    assert "退出码：0" in safe_glob_out
