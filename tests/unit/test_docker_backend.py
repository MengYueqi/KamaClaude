import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kama_claude.core.sandbox import SandboxUnavailableError
from kama_claude.core.sandbox.docker import DockerBackend
from kama_claude.core.sandbox.errors import SandboxUnavailableError as DirectUnavailableError
from kama_claude.core.sandbox.models import ExecRequest, SandboxLimits
from kama_claude.core.sandbox.workspace import WorkspaceFS


class _FakeProcess:
    def __init__(
        self,
        output: bytes = b"",
        returncode: int = 0,
        *,
        release: asyncio.Event | None = None,
        on_communicate: Callable[[], None] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.output = output
        self.returncode: int | None = returncode
        self.release = release
        self.on_communicate = on_communicate
        self.error = error
        self.started = asyncio.Event()
        self.communicated = False
        self.killed = False

    async def communicate(self) -> tuple[bytes, None]:
        self.started.set()
        if self.on_communicate is not None:
            self.on_communicate()
        if self.release is not None:
            await self.release.wait()
        self.communicated = True
        if self.error is not None:
            raise self.error
        return self.output, None

    def kill(self) -> None:
        self.killed = True
        if self.release is not None:
            self.release.set()


def _limits(
    *, timeout_s: int = 5, output_limit_bytes: int = 1_024
) -> SandboxLimits:
    return SandboxLimits(timeout_s, output_limit_bytes, 1_024, 2.0, 128, 256)


def _backend(tmp_path: Path, *, network: bool = False) -> DockerBackend:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    return DockerBackend(WorkspaceFS(tmp_path), "sandbox-image:latest", network, runtime_dir)


def _request(tmp_path: Path, command: str = "printf ok") -> ExecRequest:
    cwd = tmp_path / "nested"
    cwd.mkdir(exist_ok=True)
    return ExecRequest(
        command,
        cwd,
        {
            "PATH": "/usr/bin",
            "HOME": str(tmp_path / "runtime" / "home"),
            "TMPDIR": str(tmp_path / "runtime" / "tmp"),
            "VISIBLE": "allowed",
        },
    )


# 功能：构造完整且顺序稳定的 Docker 强隔离 argv
# 设计：逐项手写期望值，任一网络、权限、资源、挂载、工作目录或用户限制丢失都会失败
def test_build_argv_applies_every_strong_isolation_constraint(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    request = _request(tmp_path)
    name = "kama-0123456789abcdef"

    argv = backend._build_argv(request, _limits(), name)

    assert argv == [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--memory",
        "1024m",
        "--cpus",
        "2.0",
        "--tmpfs",
        "/tmp:size=256m",
        "--mount",
        f"type=bind,src={tmp_path.resolve()},dst=/workspace,rw",
        "--workdir",
        "/workspace/nested",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/tmp",
        "--env",
        "PATH=/usr/bin",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "VISIBLE=allowed",
        "sandbox-image:latest",
        "/bin/sh",
        "-lc",
        "printf ok",
    ]


# 功能：用户命令无法拆成 Docker 参数或污染安全生成的容器名
# 设计：恶意 flag 文本只允许位于镜像后的单个 shell 参数，名称仅由固定前缀和随机 token 组成
async def test_execute_keeps_command_in_one_argv_element_and_uses_safe_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = "a1b2c3d4e5f60718"
    command = "printf ok --network host --mount type=bind,src=/,dst=/host"
    process = _FakeProcess(b"ok", 0)
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append((argv, kwargs))
        return process

    monkeypatch.setattr("kama_claude.core.sandbox.docker.secrets.token_hex", lambda _: token)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await _backend(tmp_path).execute(_request(tmp_path, command), _limits())

    assert result.output == "ok"
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[argv.index("--name") + 1] == f"kama-{token}"
    assert argv[-4:] == ("sandbox-image:latest", "/bin/sh", "-lc", command)
    assert argv.count(command) == 1
    assert kwargs == {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
    }


# 功能：显式启网时仅选择 Docker bridge 网络
# 设计：直接检查网络参数值，防止强隔离配置被映射为 host 网络
def test_build_argv_maps_enabled_network_to_bridge(tmp_path: Path) -> None:
    argv = _backend(tmp_path, network=True)._build_argv(
        _request(tmp_path), _limits(), "kama-0123456789abcdef"
    )

    assert argv[argv.index("--network") + 1] == "bridge"
    assert "host" not in argv


# 功能：根工作目录精确映射为容器内 /workspace
# 设计：覆盖 WorkspaceFS.relative 的点路径分支，避免生成带尾部点段的容器目录
def test_build_argv_maps_workspace_root_to_container_root(tmp_path: Path) -> None:
    request = ExecRequest("pwd", tmp_path, {})

    argv = _backend(tmp_path)._build_argv(request, _limits(), "kama-0123456789abcdef")

    assert argv[argv.index("--workdir") + 1] == "/workspace"


# 功能：容器环境只来自请求并强制隔离 HOME 与 TMPDIR
# 设计：宿主 Secret 和 runtime_dir 均不得出现在 argv，也不得使用 env-file 或额外挂载
def test_build_argv_passes_only_request_environment_with_safe_runtime_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORE_ONLY_SECRET", "must-not-leak")
    runtime_dir = tmp_path / "host-runtime-secret"
    runtime_dir.mkdir()
    backend = DockerBackend(WorkspaceFS(tmp_path), "image", False, runtime_dir)
    request = ExecRequest("env", tmp_path, {"TOKEN": "visible", "HOME": "host-home"})

    argv = backend._build_argv(request, _limits(), "kama-0123456789abcdef")

    env_values = [argv[index + 1] for index, value in enumerate(argv) if value == "--env"]
    mount_values = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert env_values == ["HOME=/tmp", "TMPDIR=/tmp", "TOKEN=visible"]
    assert mount_values == [f"type=bind,src={tmp_path.resolve()},dst=/workspace,rw"]
    assert "--env-file" not in argv
    assert all("CORE_ONLY_SECRET" not in value for value in argv)
    assert all(str(runtime_dir) not in value for value in argv)
    assert all("docker.sock" not in value for value in argv)


# 功能：正常非零容器退出保持为 ExecResult 并按字节截断输出
# 设计：受控进程同时验证 stderr 合并、非零码不误判 unavailable 以及 UTF-8 截断标志
async def test_execute_preserves_container_failure_and_truncates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess("éé".encode(), 7)

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await _backend(tmp_path).execute(
        _request(tmp_path), _limits(output_limit_bytes=3)
    )

    assert result.returncode == 7
    assert result.output == "é�"
    assert result.timed_out is False
    assert result.truncated is True
    assert process.communicated is True


# 功能：超时后用相同安全名称强制删除容器并排空原 docker run
# 设计：cleanup fake 释放 run 输出，证明 rm argv 先执行且原 communicate task 最终被收割
async def test_execute_timeout_removes_same_container_and_drains_run_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = "1122334455667788"
    release_run = asyncio.Event()
    run_process = _FakeProcess(b"late-output", 137, release=release_run)
    cleanup_process = _FakeProcess(on_communicate=release_run.set)
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append((argv, kwargs))
        return run_process if len(calls) == 1 else cleanup_process

    monkeypatch.setattr("kama_claude.core.sandbox.docker.secrets.token_hex", lambda _: token)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await _backend(tmp_path).execute(
        _request(tmp_path), _limits(timeout_s=0, output_limit_bytes=4)
    )

    assert calls[0][0][:2] == ("docker", "run")
    assert calls[1][0] == ("docker", "rm", "-f", f"kama-{token}")
    assert calls[1][1] == {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
    }
    assert run_process.communicated is True
    assert cleanup_process.communicated is True
    assert result.returncode == 137
    assert result.output == "late"
    assert result.timed_out is True
    assert result.truncated is True


# 功能：调用方取消执行时仍删除容器并收割通信任务
# 设计：先等待 run communicate 启动再取消，断言 cleanup 释放并排空原进程后才传播取消
async def test_execute_cancellation_removes_container_and_reaps_communication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_run = asyncio.Event()
    run_process = _FakeProcess(release=release_run)
    cleanup_process = _FakeProcess(on_communicate=release_run.set)
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        return run_process if len(calls) == 1 else cleanup_process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    task = asyncio.create_task(_backend(tmp_path).execute(_request(tmp_path), _limits()))
    await run_process.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ("docker", "rm", "-f", name)
    assert run_process.communicated is True
    assert cleanup_process.communicated is True


# 功能：Docker CLI 缺失转换为结构化 unavailable 错误且不回退 Host
# 设计：在唯一外部边界抛 FileNotFoundError，断言公开错误类型与脱敏后的稳定字段
async def test_execute_reports_missing_docker_cli_without_fallback_or_secret_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def missing_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        raise FileNotFoundError("docker executable missing SECRET_VALUE")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_exec)

    with pytest.raises(SandboxUnavailableError) as caught:
        await _backend(tmp_path).execute(
            _request(tmp_path, "printf SECRET_VALUE"), _limits()
        )

    assert caught.value.backend == "docker"
    assert caught.value.reason == "cli-not-found"
    assert "SECRET_VALUE" not in str(caught.value)
    assert caught.value.__cause__ is None


# 功能：Docker run 的保留退出码 125 转换为不可用错误
# 设计：输出与请求均含 Secret，错误只暴露后端和稳定原因码而不拼接外部文本
async def test_execute_reports_exit_125_without_secret_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess(b"docker failed with SECRET_VALUE", 125)

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(SandboxUnavailableError) as caught:
        await _backend(tmp_path).execute(
            _request(tmp_path, "printf SECRET_VALUE"), _limits()
        )

    assert caught.value.reason == "startup-failed"
    assert "SECRET_VALUE" not in str(caught.value)


# 功能：daemon 连接失败即使返回非 125 也转换为不可用错误
# 设计：模拟 Docker CLI 的稳定诊断短语，区别于普通容器非零退出结果
async def test_execute_reports_daemon_connection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess(b"Cannot connect to the Docker daemon at unix:///socket", 1)

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(SandboxUnavailableError) as caught:
        await _backend(tmp_path).execute(_request(tmp_path), _limits())

    assert caught.value.reason == "daemon-unreachable"


# 功能：run communicate 异常时仍以同名 rm 清理容器
# 设计：真实 execute 捕获异步输出错误，cleanup 完成后传播原始异常且不遗留通信任务
async def test_execute_communication_error_still_removes_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _FakeProcess(error=RuntimeError("stream failed"))
    cleanup_process = _FakeProcess()
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        return run_process if len(calls) == 1 else cleanup_process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(RuntimeError, match="stream failed"):
        await _backend(tmp_path).execute(_request(tmp_path), _limits())

    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ("docker", "rm", "-f", name)
    assert cleanup_process.communicated is True


# 功能：sandbox 包根稳定导出结构化不可用错误
# 设计：identity 断言防止 package export 被包装成不同异常类型而破坏捕获契约
def test_sandbox_package_exports_unavailable_error() -> None:
    assert SandboxUnavailableError is DirectUnavailableError
