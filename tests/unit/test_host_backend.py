import asyncio
import os
import shlex
import sys
from pathlib import Path

import pytest

from kama_claude.core.sandbox.host import HostBackend
from kama_claude.core.sandbox.models import ExecRequest, SandboxLimits


# 功能：构造位于可信运行目录内的 HOME 与 TMPDIR
# 设计：所有允许执行的测试都模拟环境构造器提供的受控运行目录
def _runtime_env(work_dir: Path) -> dict[str, str]:
    home = work_dir / "home"
    tmpdir = work_dir / "tmp"
    home.mkdir(exist_ok=True)
    tmpdir.mkdir(exist_ok=True)
    return {"HOME": str(home), "TMPDIR": str(tmpdir)}


# 功能：轮询确认超时命令派生的子进程已经不再存在
# 设计：有界等待避免测试本身卡住，同时防止测试结束后遗留宿主进程
async def _wait_for_process_exit(pid: int) -> None:
    deadline = asyncio.get_running_loop().time() + 3.0
    while asyncio.get_running_loop().time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.05)
    pytest.fail(f"child process {pid} survived timeout cleanup")


# 功能：执行成功命令并暴露兼容模式后端的稳定属性
# 设计：验证后端使用请求工作目录，且不宣称提供强隔离
async def test_host_backend_runs_command_in_requested_directory(tmp_path: Path) -> None:
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest("pwd", tmp_path, _runtime_env(tmp_path))

    result = await backend.execute(request, limits)

    assert backend.name == "host"
    assert backend.strongly_isolated is False
    assert result.returncode == 0
    assert result.output == f"{tmp_path}\n"
    assert result.timed_out is False
    assert result.truncated is False


# 功能：将标准错误合并到标准输出并保留其顺序
# 设计：调用方只消费一个输出字段，诊断信息不能因 stderr 管道而丢失
async def test_host_backend_merges_stderr_into_output(tmp_path: Path) -> None:
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest("printf out; printf err >&2", tmp_path, _runtime_env(tmp_path))

    result = await backend.execute(request, limits)

    assert result.returncode == 0
    assert result.output == "outerr"


# 功能：保留命令的非零退出码
# 设计：失败命令的输出可用于诊断，但不能被后端转换成成功结果
async def test_host_backend_preserves_nonzero_returncode(tmp_path: Path) -> None:
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest("printf failed; exit 7", tmp_path, _runtime_env(tmp_path))

    result = await backend.execute(request, limits)

    assert result.returncode == 7
    assert result.output == "failed"
    assert result.timed_out is False


# 功能：在 UTF-8 字符中间按原始字节限制输出
# 设计：限制值是字节数而非字符数，截断后的无效尾字节以替换字符安全解码
async def test_host_backend_truncates_output_by_bytes_before_decoding(tmp_path: Path) -> None:
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 3, 1_024, 1.0, 64, 64)
    request = ExecRequest("printf 'éé'", tmp_path, _runtime_env(tmp_path))

    result = await backend.execute(request, limits)

    assert result.output == "é�"
    assert result.truncated is True


# 功能：仅向子进程传入请求指定的环境变量
# 设计：宿主 Secret 即使存在于 pytest 进程环境中也不能穿透精确 env 边界
async def test_host_backend_does_not_inherit_host_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_name = "HOST_BACKEND_TEST_SECRET"
    monkeypatch.setenv(secret_name, "host-only-secret")
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    script = (
        "import os; "
        "print(os.environ.get('ONLY_ALLOWED')); "
        f"print(os.environ.get({secret_name!r}, 'missing'))"
    )
    command = f"exec {shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    request = ExecRequest(command, tmp_path, _runtime_env(tmp_path) | {"ONLY_ALLOWED": "visible"})

    result = await backend.execute(request, limits)

    assert result.returncode == 0
    assert result.output == "visible\nmissing\n"


# 功能：命令超时后返回标准化超时结果而非抛出异常
# 设计：超时通过进程组终止执行，调用方能继续统一处理命令输出和退出码
async def test_host_backend_returns_timeout_result(tmp_path: Path) -> None:
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(1, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest("sleep 60", tmp_path, _runtime_env(tmp_path))

    result = await backend.execute(request, limits)

    assert result.timed_out is True
    assert result.returncode != 0
    assert result.output == ""


# 功能：超时后同时清理父进程及其派生的睡眠子进程
# 设计：父进程以新会话领导进程组，测试读取 PID 文件并有界轮询确认无遗留进程
async def test_host_backend_timeout_kills_entire_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    child_script = "import time; time.sleep(60)"
    parent_script = (
        "import pathlib, subprocess, sys, time; "
        f"child = subprocess.Popen([{sys.executable!r}, '-c', {child_script!r}]); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    command = f"exec {shlex.quote(sys.executable)} -c {shlex.quote(parent_script)}"
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(1, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest(command, tmp_path, _runtime_env(tmp_path))

    result = await backend.execute(request, limits)

    assert result.timed_out is True
    assert pid_file.is_file()
    await _wait_for_process_exit(int(pid_file.read_text()))


# 功能：拒绝缺失 HOME 的精确环境并阻止命令启动
# 设计：登录 shell 不得回退到宿主账户目录读取 profile 或泄露其副作用
async def test_host_backend_rejects_missing_home_before_start(tmp_path: Path) -> None:
    marker = tmp_path / "started"
    env = _runtime_env(tmp_path)
    del env["HOME"]
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest(f"touch {shlex.quote(str(marker))}", tmp_path, env)

    with pytest.raises(ValueError, match="HOME"):
        await backend.execute(request, limits)

    assert marker.exists() is False


# 功能：拒绝缺失 TMPDIR 的精确环境并阻止命令启动
# 设计：兼容后端不能让 shell 在缺少受控临时目录时退回宿主默认路径
async def test_host_backend_rejects_missing_tmpdir_before_start(tmp_path: Path) -> None:
    marker = tmp_path / "started"
    env = _runtime_env(tmp_path)
    del env["TMPDIR"]
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest(f"touch {shlex.quote(str(marker))}", tmp_path, env)

    with pytest.raises(ValueError, match="TMPDIR"):
        await backend.execute(request, limits)

    assert marker.exists() is False


# 功能：拒绝相对 HOME 或 TMPDIR，避免依赖进程当前目录解释运行时环境
# 设计：环境根必须是明确的绝对路径，不能被 shell 或 cwd 间接改变
@pytest.mark.parametrize("name", ["HOME", "TMPDIR"])
async def test_host_backend_rejects_relative_runtime_environment(
    tmp_path: Path, name: str
) -> None:
    marker = tmp_path / "started"
    env = _runtime_env(tmp_path)
    env[name] = "relative-runtime-dir"
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest(f"touch {shlex.quote(str(marker))}", tmp_path, env)

    with pytest.raises(ValueError, match=name):
        await backend.execute(request, limits)

    assert marker.exists() is False


# 功能：拒绝解析后位于可信工作目录外的 HOME 或 TMPDIR
# 设计：绝对路径本身不足以隔离，解析结果必须仍包含在工作目录边界内
@pytest.mark.parametrize("name", ["HOME", "TMPDIR"])
async def test_host_backend_rejects_runtime_environment_outside_work_dir(
    tmp_path: Path, name: str
) -> None:
    marker = tmp_path / "started"
    env = _runtime_env(tmp_path)
    env[name] = str(tmp_path.parent / "outside-runtime-dir")
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest(f"touch {shlex.quote(str(marker))}", tmp_path, env)

    with pytest.raises(ValueError, match=name):
        await backend.execute(request, limits)

    assert marker.exists() is False


# 功能：拒绝经符号链接解析到可信工作目录外的 HOME 或 TMPDIR
# 设计：路径字符串看似位于工作目录内时也必须防止符号链接逃逸
@pytest.mark.parametrize("name", ["HOME", "TMPDIR"])
async def test_host_backend_rejects_symlinked_runtime_environment_outside_work_dir(
    tmp_path: Path, name: str
) -> None:
    marker = tmp_path / "started"
    outside = tmp_path.parent / f"outside-runtime-dir-{name}"
    outside.mkdir()
    link = tmp_path / "runtime-link"
    link.symlink_to(outside, target_is_directory=True)
    env = _runtime_env(tmp_path)
    env[name] = str(link)
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest(f"touch {shlex.quote(str(marker))}", tmp_path, env)

    with pytest.raises(ValueError, match=name):
        await backend.execute(request, limits)

    assert marker.exists() is False


# 功能：构造时拒绝不存在或非目录的工作目录
# 设计：环境路径校验依赖可信根，不能将文件或缺失路径静默视作根目录
@pytest.mark.parametrize("kind", ["missing", "file"])
def test_host_backend_requires_existing_directory_work_dir(tmp_path: Path, kind: str) -> None:
    work_dir = tmp_path / kind
    if kind == "file":
        work_dir.write_text("not a directory")

    with pytest.raises(ValueError, match="work_dir"):
        HostBackend(work_dir)
