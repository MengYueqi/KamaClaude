import asyncio
import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from kama_claude.core.sandbox import SandboxUnavailableError
from kama_claude.core.sandbox.docker import DockerBackend
from kama_claude.core.sandbox.errors import SandboxUnavailableError as DirectUnavailableError
from kama_claude.core.sandbox.models import ExecRequest, SandboxLimits
from kama_claude.core.sandbox.workspace import WorkspaceFS


class _FakeStdout:
    def __init__(self, process: "_FakeProcess") -> None:
        self._process = process
        self._sent = False

    async def read(self, size: int) -> bytes:
        process = self._process
        process.started.set()
        process.process_task = asyncio.current_task()
        if process.on_communicate is not None:
            process.on_communicate()
        if process.release is not None:
            await process.release.wait()
        if process.error is not None:
            raise process.error
        if self._sent:
            return b""
        self._sent = True
        assert len(process.output) <= size
        return process.output


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
        self.final_returncode = returncode
        self.returncode: int | None = None
        self.release = release
        self.on_communicate = on_communicate
        self.error = error
        self.started = asyncio.Event()
        self.communicated = False
        self.killed = False
        self.process_task: asyncio.Task[Any] | None = None
        self.stdout = _FakeStdout(self)

    async def communicate(self) -> tuple[bytes, None]:
        self.started.set()
        self.process_task = asyncio.current_task()
        if self.on_communicate is not None:
            self.on_communicate()
        if self.release is not None:
            await self.release.wait()
        self.communicated = True
        if self.error is not None:
            raise self.error
        if self.returncode is None:
            self.returncode = self.final_returncode
        return self.output, None

    async def wait(self) -> int:
        self.communicated = True
        if self.returncode is None:
            self.returncode = self.final_returncode
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        if self.release is not None:
            self.release.set()


class _KillErrorProcess(_FakeProcess):
    def __init__(self, error: OSError) -> None:
        super().__init__(release=asyncio.Event())
        self.kill_error = error
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1
        raise self.kill_error


class _CancellationResistantProcess(_FakeProcess):
    def __init__(
        self, *, ignored_cancellations: int, error: BaseException | None = None
    ) -> None:
        super().__init__(error=error)
        self.ignored_cancellations = ignored_cancellations
        self.release_stubborn = asyncio.Event()
        self.killed_event = asyncio.Event()
        self.cancelled_event = asyncio.Event()
        self.finished_event = asyncio.Event()
        self.cancel_count = 0
        self.stdout = self

    async def read(self, size: int) -> bytes:
        self.started.set()
        self.process_task = asyncio.current_task()
        while not self.release_stubborn.is_set():
            try:
                await self.release_stubborn.wait()
            except asyncio.CancelledError:
                self.cancel_count += 1
                self.cancelled_event.set()
                if self.cancel_count > self.ignored_cancellations:
                    raise
        self.communicated = True
        self.finished_event.set()
        if self.error is not None:
            raise self.error
        return b""

    async def communicate(self) -> tuple[bytes, None]:
        await self.read(1)
        return self.output, None

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.killed_event.set()


class _ChunkStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self._index = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        await asyncio.sleep(0)
        if self._index == len(self._chunks):
            return b""
        chunk = self._chunks[self._index]
        self._index += 1
        assert len(chunk) <= size
        return chunk


class _StreamingProcess:
    def __init__(self, chunks: list[bytes], returncode: int = 0) -> None:
        self.stdout = _ChunkStream(chunks)
        self._all_output = b"".join(chunks)
        self.final_returncode = returncode
        self.returncode: int | None = None
        self.communicate_called = False
        self.wait_called = False

    async def communicate(self) -> tuple[bytes, None]:
        self.communicate_called = True
        self.returncode = self.final_returncode
        return self._all_output, None

    async def wait(self) -> int:
        self.wait_called = True
        self.returncode = self.final_returncode
        return self.final_returncode

    def kill(self) -> None:
        self.returncode = -9


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


# 功能：Docker run 输出按固定块持续排空且 retained bytes 始终受 output limit 约束
# 设计：大量受控 chunks 全部读取，4 字节单调 retained buffer 即峰值，并验证 UTF-8 replacement 与截断
async def test_execute_streams_all_output_while_retaining_only_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [b"A", b"\xf0", b"\x9f", b"\x92", *([b"zz"] * 1_000)]
    process = _StreamingProcess(chunks)
    retained_sizes: list[int] = []

    class TrackingBytearray(bytearray):
        def extend(self, chunk: bytes) -> None:
            super().extend(chunk)
            retained_sizes.append(len(self))

    async def fake_exec(*argv: str, **kwargs: Any) -> _StreamingProcess:
        return process

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._OUTPUT_CHUNK_BYTES", 3, raising=False
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker.bytearray", TrackingBytearray, raising=False
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await _backend(tmp_path).execute(
        _request(tmp_path), _limits(output_limit_bytes=4)
    )

    assert result.output == "A�"
    assert result.truncated is True
    assert retained_sizes and max(retained_sizes) == 4
    assert process.stdout.read_sizes == [3] * (len(chunks) + 1)
    assert process.wait_called is True
    assert process.communicate_called is False


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
        _request(tmp_path), _limits(timeout_s=1, output_limit_bytes=4)
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


# 功能：deadline 与 collector 完成竞态按真实完成返回且不误删已由 --rm 移除的容器
# 设计：wait_for 在取得完成结果后确定性抛 TimeoutError，done task 必须被消费并报告 timed_out=False
async def test_execute_timeout_race_uses_completed_collection_without_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _FakeProcess(b"completed", 0)
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        if len(calls) > 1:
            raise AssertionError("spurious cleanup spawn")
        return run_process

    async def timeout_after_completion(awaitable: Any, timeout: float) -> Any:
        await awaitable
        raise TimeoutError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "wait_for", timeout_after_completion)

    result = await _backend(tmp_path).execute(_request(tmp_path), _limits())

    assert result.output == "completed"
    assert result.timed_out is False
    assert result.truncated is False
    assert calls[0][:2] == ("docker", "run")
    assert len(calls) == 1
    assert run_process.communicated is True


# 功能：docker rm communicate 卡住时在有限时间内杀死并收割本地 cleanup CLI
# 设计：fallback 定时释放防止旧实现挂住；新实现必须先命中私有正数 deadline 并返回脱敏错误
async def test_execute_bounds_hung_cleanup_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_run = asyncio.Event()
    release_cleanup = asyncio.Event()
    run_process = _FakeProcess(release=release_run)

    def schedule_fallback_release() -> None:
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, release_cleanup.set)
        loop.call_later(0.10, release_run.set)

    cleanup_process = _FakeProcess(
        release=release_cleanup, on_communicate=schedule_fallback_release
    )
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        return run_process if len(calls) == 1 else cleanup_process

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._CLEANUP_TIMEOUT_S", 0.01, raising=False
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._RUN_REAP_TIMEOUT_S", 0.01, raising=False
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._KILL_REAP_TIMEOUT_S", 0.01, raising=False
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(SandboxUnavailableError) as caught:
        await _backend(tmp_path).execute(_request(tmp_path), _limits(timeout_s=1))

    assert caught.value.reason == "cleanup-failed"
    assert cleanup_process.killed is True
    assert cleanup_process.communicated is True
    assert run_process.killed is True
    assert run_process.communicated is True


# 功能：docker rm 非零且原 run CLI 不退出时仍在有限时间内完成本地收割
# 设计：cleanup 已失败时保留其错误优先级，同时 deadline 必须 kill 并 drain 卡住的 run task
async def test_execute_bounds_run_reaping_after_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_run = asyncio.Event()
    run_process = _FakeProcess(release=release_run)

    def schedule_fallback_release() -> None:
        asyncio.get_running_loop().call_later(0.05, release_run.set)

    cleanup_process = _FakeProcess(returncode=1, on_communicate=schedule_fallback_release)
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        return run_process if len(calls) == 1 else cleanup_process

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._RUN_REAP_TIMEOUT_S", 0.01, raising=False
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._KILL_REAP_TIMEOUT_S", 0.01, raising=False
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(SandboxUnavailableError) as caught:
        await _backend(tmp_path).execute(_request(tmp_path), _limits(timeout_s=1))

    assert caught.value.reason == "cleanup-failed"
    assert cleanup_process.communicated is True
    assert run_process.killed is True
    assert run_process.communicated is True


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


# 功能：调用方取消始终优先于随后发生的 cleanup 失败
# 设计：取消 pending run 后让 rm spawn 抛含 Secret 的异常，仍须 kill/drain run 并传播原 CancelledError
async def test_execute_preserves_cancellation_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _FakeProcess(release=asyncio.Event())
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        if len(calls) == 1:
            return run_process
        raise OSError("cleanup spawn exposed SECRET_VALUE")

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._RUN_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._KILL_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    task = asyncio.create_task(_backend(tmp_path).execute(_request(tmp_path), _limits()))
    await run_process.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert run_process.killed is True
    assert run_process.communicated is True


# 功能：取消与正常完成竞态不会为已由 --rm 删除的容器再发 cleanup
# 设计：wait_for 在 run communicate 完成后设门闩，取消 execute 后必须收集结果并保持唯一一次 spawn
async def test_execute_cancellation_after_run_completion_skips_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _FakeProcess(b"done", 0)
    wait_completed = asyncio.Event()
    hold_result = asyncio.Event()
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        if len(calls) > 1:
            raise AssertionError("spurious cleanup spawn")
        return run_process

    async def gated_wait_for(awaitable: Any, timeout: float) -> Any:
        result = await awaitable
        wait_completed.set()
        await hold_result.wait()
        return result

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "wait_for", gated_wait_for)
    task = asyncio.create_task(_backend(tmp_path).execute(_request(tmp_path), _limits()))
    await wait_completed.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) == 1
    assert run_process.communicated is True


# 功能：kill/reap 等待期间的重复取消始终以 caller CancelledError 为最高优先级
# 设计：首个外层取消命中 reap wait，communicate 抗一次取消后再次取消，禁止降级为 cleanup-failed
async def test_execute_preserves_repeated_cancellation_during_kill_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _CancellationResistantProcess(ignored_cancellations=1)
    cleanup_process = _FakeProcess()
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        return run_process if len(calls) == 1 else cleanup_process

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._RUN_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._KILL_REAP_TIMEOUT_S", 1.0
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    task = asyncio.create_task(
        _backend(tmp_path).execute(_request(tmp_path), _limits(timeout_s=1))
    )

    await asyncio.wait_for(run_process.killed_event.wait(), timeout=2.0)
    fallback = asyncio.get_running_loop().call_later(0.20, run_process.release_stubborn.set)
    try:
        assert task.cancel()
        assert task.cancel()
        await asyncio.wait_for(run_process.cancelled_event.wait(), timeout=0.5)
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        fallback.cancel()
        run_process.release_stubborn.set()
        await asyncio.sleep(0)

    assert task.cancelled()
    assert run_process.killed is True


# 功能：run collector 的 bounded wait 被取消时，kill OSError 不能覆盖 caller cancellation
# 设计：同步到 cleanup 后的 run wait 再取消，含 Secret 的 kill 失败仅作 best-effort 内部错误被抑制
async def test_execute_cancellation_outranks_kill_oserror_during_run_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _KillErrorProcess(OSError("kill exposed CANCEL_KILL_SECRET"))
    cleanup_process = _FakeProcess()
    entered_run_reap = asyncio.Event()
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        return run_process if len(calls) == 1 else cleanup_process

    backend = _backend(tmp_path)
    original_bounded = backend._bounded_process_task

    async def synchronized_bounded(
        process: Any, process_task: asyncio.Task[Any], timeout_s: float
    ) -> Any:
        if process is run_process:
            entered_run_reap.set()
        return await original_bounded(process, process_task, timeout_s)

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._KILL_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._TASK_CANCEL_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(backend, "_bounded_process_task", synchronized_bounded)
    task = asyncio.create_task(
        backend.execute(_request(tmp_path), _limits(timeout_s=1))
    )

    try:
        await asyncio.wait_for(entered_run_reap.wait(), timeout=2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        assert run_process.release is not None
        run_process.release.set()
        await asyncio.sleep(0)

    assert run_process.kill_calls == 1
    assert run_process.process_task is not None
    assert run_process.process_task.done() or run_process.process_task in backend._detached_tasks


# 功能：communicate 持续抗取消时 execute 仍有界返回且最终异常被安全消费
# 设计：硬 deadline 后强引用 detached task；释放后 callback 必须取走 Secret 异常并自动移除引用
async def test_execute_detaches_cancellation_resistant_communication_with_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _CancellationResistantProcess(
        ignored_cancellations=10,
        error=RuntimeError("detached communicate exposed DETACHED_SECRET"),
    )
    cleanup_process = _FakeProcess()
    calls: list[tuple[str, ...]] = []
    unhandled: list[dict[str, Any]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        return run_process if len(calls) == 1 else cleanup_process

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._RUN_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._KILL_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._TASK_CANCEL_TIMEOUT_S", 0.01, raising=False
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    backend = _backend(tmp_path)
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    task = asyncio.create_task(
        backend.execute(_request(tmp_path), _limits(timeout_s=1))
    )

    try:
        await asyncio.wait_for(run_process.killed_event.wait(), timeout=2.0)
        done, _ = await asyncio.wait({task}, timeout=0.08)
        completed_within_bound = task in done
        if not completed_within_bound:
            run_process.release_stubborn.set()
        with pytest.raises(SandboxUnavailableError) as caught:
            await task
        assert completed_within_bound
        assert caught.value.reason == "cleanup-failed"
        assert run_process.process_task is not None
        assert run_process.process_task in backend._detached_tasks

        run_process.release_stubborn.set()
        await asyncio.wait_for(run_process.finished_event.wait(), timeout=0.5)
        await asyncio.sleep(0)
        assert run_process.process_task.done()
        assert not backend._detached_tasks
        assert unhandled == []
    finally:
        run_process.release_stubborn.set()
        if not task.done():
            with suppress(BaseException):
                await task
        loop.set_exception_handler(previous_handler)


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


# 功能：cleanup spawn 的任意失败统一转换为脱敏 cleanup-failed
# 设计：主 run 超时后让 rm spawn 抛含 Secret 的 OSError，并验证原 run 仍被有界 kill/drain
async def test_execute_sanitizes_cleanup_spawn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _FakeProcess(release=asyncio.Event())
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        if len(calls) == 1:
            return run_process
        raise OSError("cleanup spawn exposed SECRET_VALUE")

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._RUN_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._KILL_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(SandboxUnavailableError) as caught:
        await _backend(tmp_path).execute(_request(tmp_path), _limits(timeout_s=1))

    assert caught.value.backend == "docker"
    assert caught.value.reason == "cleanup-failed"
    assert "SECRET_VALUE" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert run_process.killed is True
    assert run_process.communicated is True


# 功能：本地 Docker CLI kill 失败时仍脱敏错误并安全处置 communicate task
# 设计：含 Secret 的 PermissionError 不得成为 cause 或二次 kill 逃逸，task 必须完成或受强引用回调监管
async def test_execute_sanitizes_kill_failure_and_disposes_communication_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _KillErrorProcess(PermissionError("kill denied KILL_SECRET"))
    cleanup_process = _FakeProcess()
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        return run_process if len(calls) == 1 else cleanup_process

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._RUN_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._KILL_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    backend = _backend(tmp_path)

    try:
        with pytest.raises(SandboxUnavailableError) as caught:
            await backend.execute(_request(tmp_path), _limits(timeout_s=1))
    finally:
        assert run_process.release is not None
        run_process.release.set()
        await asyncio.sleep(0)

    assert caught.value.reason == "cleanup-failed"
    assert "KILL_SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert run_process.kill_calls == 1
    assert run_process.process_task is not None
    assert run_process.process_task.done() or run_process.process_task in backend._detached_tasks


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


# 功能：Docker API socket 权限拒绝在非 125 退出时仍识别为 daemon 不可用
# 设计：覆盖 Docker CLI 的另一条稳定诊断短语，不能将宿主 daemon 权限问题误作容器退出
async def test_execute_reports_daemon_api_permission_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess(
        b"permission denied while trying to connect to the Docker API at unix:///socket",
        1,
    )

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(SandboxUnavailableError) as caught:
        await _backend(tmp_path).execute(_request(tmp_path), _limits())

    assert caught.value.reason == "daemon-unreachable"


# 功能：cleanup communicate 异常被脱敏且优先于原 run communication 异常
# 设计：两条通信路径同时抛含 Secret 的错误，安全关键的 cleanup-failed 必须成为唯一对外错误
async def test_cleanup_failure_has_priority_over_run_communication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_process = _FakeProcess(error=RuntimeError("run exposed RUN_SECRET"))
    cleanup_process = _FakeProcess(error=RuntimeError("cleanup exposed CLEANUP_SECRET"))
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        calls.append(argv)
        return run_process if len(calls) == 1 else cleanup_process

    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker._KILL_REAP_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(SandboxUnavailableError) as caught:
        await _backend(tmp_path).execute(_request(tmp_path), _limits())

    assert caught.value.reason == "cleanup-failed"
    assert "RUN_SECRET" not in str(caught.value)
    assert "CLEANUP_SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert cleanup_process.killed is True
    assert cleanup_process.communicated is True


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
