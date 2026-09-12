import asyncio
import os
import shlex
import sys
from pathlib import Path

import pytest

from kama_claude.core.sandbox.host import HostBackend
from kama_claude.core.sandbox.models import ExecRequest, SandboxLimits


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
    request = ExecRequest("pwd", tmp_path, {"HOME": str(tmp_path)})

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
    request = ExecRequest("printf out; printf err >&2", tmp_path, {"HOME": str(tmp_path)})

    result = await backend.execute(request, limits)

    assert result.returncode == 0
    assert result.output == "outerr"


# 功能：保留命令的非零退出码
# 设计：失败命令的输出可用于诊断，但不能被后端转换成成功结果
async def test_host_backend_preserves_nonzero_returncode(tmp_path: Path) -> None:
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest("printf failed; exit 7", tmp_path, {"HOME": str(tmp_path)})

    result = await backend.execute(request, limits)

    assert result.returncode == 7
    assert result.output == "failed"
    assert result.timed_out is False


# 功能：在 UTF-8 字符中间按原始字节限制输出
# 设计：限制值是字节数而非字符数，截断后的无效尾字节以替换字符安全解码
async def test_host_backend_truncates_output_by_bytes_before_decoding(tmp_path: Path) -> None:
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(5, 3, 1_024, 1.0, 64, 64)
    request = ExecRequest("printf 'éé'", tmp_path, {"HOME": str(tmp_path)})

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
    request = ExecRequest(command, tmp_path, {"HOME": str(tmp_path), "ONLY_ALLOWED": "visible"})

    result = await backend.execute(request, limits)

    assert result.returncode == 0
    assert result.output == "visible\nmissing\n"


# 功能：命令超时后返回标准化超时结果而非抛出异常
# 设计：超时通过进程组终止执行，调用方能继续统一处理命令输出和退出码
async def test_host_backend_returns_timeout_result(tmp_path: Path) -> None:
    backend = HostBackend(tmp_path)
    limits = SandboxLimits(1, 1_024, 1_024, 1.0, 64, 64)
    request = ExecRequest("sleep 60", tmp_path, {"HOME": str(tmp_path)})

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
    request = ExecRequest(command, tmp_path, {"HOME": str(tmp_path)})

    result = await backend.execute(request, limits)

    assert result.timed_out is True
    assert pid_file.is_file()
    await _wait_for_process_exit(int(pid_file.read_text()))
