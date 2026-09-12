"""Strongly isolated command execution through the Docker CLI."""

import asyncio
import os
import secrets
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from kama_claude.core.sandbox.base import SandboxBackend
from kama_claude.core.sandbox.errors import SandboxUnavailableError
from kama_claude.core.sandbox.models import ExecRequest, ExecResult, SandboxLimits
from kama_claude.core.sandbox.workspace import WorkspaceFS

_CLEANUP_TIMEOUT_S = 5.0
_RUN_REAP_TIMEOUT_S = 2.0
_KILL_REAP_TIMEOUT_S = 1.0
_TASK_CANCEL_TIMEOUT_S = 1.0
_OUTPUT_CHUNK_BYTES = 64 * 1_024

_TaskResult = TypeVar("_TaskResult")


class _ProcessTimeoutError(TimeoutError):
    """Internal signal that a local Docker CLI exceeded its lifecycle bound."""


@dataclass(frozen=True)
class _RunOutput:
    """Bounded output retained while fully draining a Docker run process."""

    output: bytes
    truncated: bool


class DockerBackend(SandboxBackend):
    """Execute commands in a locked-down Docker container."""

    # 固定 Docker 执行所需的可信工作区、镜像和网络策略
    def __init__(
        self, workspace: WorkspaceFS, image: str, network: bool, runtime_dir: Path
    ) -> None:
        self._workspace = workspace
        self._image = image
        self._network = network
        self._detached_tasks: set[asyncio.Task[Any]] = set()

    # 返回强隔离后端的稳定名称
    @property
    def name(self) -> str:
        return "docker"

    # 表明 Docker 后端提供强隔离能力
    @property
    def strongly_isolated(self) -> bool:
        return True

    # 构造不经过宿主 Shell 的完整 Docker argv
    def _build_argv(
        self, request: ExecRequest, limits: SandboxLimits, container_name: str
    ) -> list[str]:
        relative_cwd = self._workspace.relative(request.cwd)
        container_cwd = (
            "/workspace" if relative_cwd == "." else f"/workspace/{relative_cwd}"
        )
        container_env = dict(request.env)
        container_env["HOME"] = "/tmp"
        container_env["TMPDIR"] = "/tmp"
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "bridge" if self._network else "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(limits.pids_limit),
            "--memory",
            f"{limits.memory_mb}m",
            "--cpus",
            str(limits.cpu_count),
            "--tmpfs",
            f"/tmp:size={limits.tmpfs_mb}m",
            "--mount",
            f"type=bind,src={self._workspace.root},dst=/workspace,rw",
            "--workdir",
            container_cwd,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
        ]
        for key in sorted(container_env):
            argv.extend(("--env", f"{key}={container_env[key]}"))
        argv.extend((self._image, "/bin/sh", "-lc", request.command))
        return argv

    # 判断 Docker CLI 输出是否表示守护进程连接失败
    @staticmethod
    def _is_daemon_connection_failure(output: bytes) -> bool:
        normalized = output.lower()
        return any(
            marker in normalized
            for marker in (
                b"cannot connect to the docker daemon",
                b"error during connect",
                b"is the docker daemon running",
                b"permission denied while trying to connect to the docker api",
            )
        )

    # 消费已结束 detached task 的结果并释放 backend 持有的强引用
    def _consume_detached_task(self, task: asyncio.Task[Any]) -> None:
        with suppress(BaseException):
            task.result()
        self._detached_tasks.discard(task)

    # 强引用仍抗取消的通信任务，并注册绝不抛错的完成回调
    def _detach_process_task(self, task: asyncio.Task[Any]) -> None:
        if task.done():
            self._consume_detached_task(task)
            return
        self._detached_tasks.add(task)
        task.add_done_callback(self._consume_detached_task)

    # 单次尝试杀死本地 Docker CLI，并以两段硬期限处置或监管通信任务
    async def _kill_and_reap(
        self,
        process: asyncio.subprocess.Process,
        process_task: asyncio.Task[Any],
    ) -> None:
        kill_failed = False
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except asyncio.CancelledError:
                raise
            except Exception:
                kill_failed = True
        if process_task.done():
            with suppress(BaseException):
                process_task.result()
        else:
            try:
                done, _ = await asyncio.wait(
                    {process_task}, timeout=_KILL_REAP_TIMEOUT_S
                )
            except asyncio.CancelledError:
                if not process_task.done():
                    process_task.cancel()
                    self._detach_process_task(process_task)
                raise
            if done:
                with suppress(BaseException):
                    process_task.result()
            else:
                process_task.cancel()
                try:
                    done, _ = await asyncio.wait(
                        {process_task}, timeout=_TASK_CANCEL_TIMEOUT_S
                    )
                except asyncio.CancelledError:
                    if not process_task.done():
                        process_task.cancel()
                        self._detach_process_task(process_task)
                    raise
                if done:
                    with suppress(BaseException):
                        process_task.result()
                else:
                    process_task.cancel()
                    self._detach_process_task(process_task)
        if kill_failed:
            raise SandboxUnavailableError("docker", "cleanup-failed") from None

    # 在正数 deadline 内等待进程任务，超界时先 kill 再确保任务被处置
    async def _bounded_process_task(
        self,
        process: asyncio.subprocess.Process,
        process_task: asyncio.Task[_TaskResult],
        timeout_s: float,
    ) -> _TaskResult:
        try:
            return await asyncio.wait_for(
                asyncio.shield(process_task), timeout=timeout_s
            )
        except asyncio.CancelledError:
            with suppress(BaseException):
                await self._kill_and_reap(process, process_task)
            raise
        except TimeoutError:
            if process_task.done():
                return process_task.result()
            await self._kill_and_reap(process, process_task)
            raise _ProcessTimeoutError from None

    # 按固定块排空 run stdout，仅单调保留限制内字节并等待进程退出
    @staticmethod
    async def _collect_run_output(
        process: asyncio.subprocess.Process, output_limit_bytes: int
    ) -> _RunOutput:
        stream = process.stdout
        if stream is None:
            raise SandboxUnavailableError("docker", "startup-failed") from None
        retained = bytearray()
        truncated = False
        while True:
            chunk = await stream.read(_OUTPUT_CHUNK_BYTES)
            if not chunk:
                break
            remaining = max(0, output_limit_bytes - len(retained))
            retained.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        await process.wait()
        return _RunOutput(bytes(retained), truncated)

    # 通过独立 argv 命令强制删除指定容器并收割清理进程
    async def _remove_container(self, container_name: str) -> None:
        try:
            cleanup = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "docker",
                    "rm",
                    "-f",
                    container_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                ),
                timeout=_CLEANUP_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except BaseException:
            raise SandboxUnavailableError("docker", "cleanup-failed") from None
        cleanup_task = asyncio.create_task(cleanup.communicate())
        try:
            await self._bounded_process_task(cleanup, cleanup_task, _CLEANUP_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except SandboxUnavailableError:
            raise
        except BaseException:
            await self._kill_and_reap(cleanup, cleanup_task)
            raise SandboxUnavailableError("docker", "cleanup-failed") from None
        if cleanup.returncode != 0:
            raise SandboxUnavailableError("docker", "cleanup-failed")

    # 删除活动容器后排空原 docker run，确保通信任务不会遗留
    async def _remove_and_collect(
        self,
        process: asyncio.subprocess.Process,
        collection_task: asyncio.Task[_RunOutput],
        container_name: str,
    ) -> _RunOutput:
        cleanup_failure: BaseException | None = None
        communication_failure: BaseException | None = None
        result: _RunOutput | None = None
        try:
            await self._remove_container(container_name)
        except BaseException as error:
            cleanup_failure = error
        try:
            result = await self._bounded_process_task(
                process, collection_task, _RUN_REAP_TIMEOUT_S
            )
        except _ProcessTimeoutError:
            communication_failure = SandboxUnavailableError("docker", "cleanup-failed")
        except SandboxUnavailableError as error:
            communication_failure = error
        except asyncio.CancelledError as error:
            communication_failure = error
        except BaseException as error:
            try:
                await self._kill_and_reap(process, collection_task)
            except SandboxUnavailableError as kill_failure:
                communication_failure = kill_failure
            else:
                communication_failure = error
        if isinstance(cleanup_failure, asyncio.CancelledError):
            raise cleanup_failure
        if isinstance(communication_failure, asyncio.CancelledError):
            raise communication_failure
        if cleanup_failure is not None:
            raise cleanup_failure
        if communication_failure is not None:
            raise communication_failure
        assert result is not None
        return result

    # 在隔离容器中执行请求并标准化输出、超时与后端不可用错误
    async def execute(self, request: ExecRequest, limits: SandboxLimits) -> ExecResult:
        container_name = f"kama-{secrets.token_hex(8)}"
        argv = self._build_argv(request, limits, container_name)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise SandboxUnavailableError("docker", "cli-not-found") from None

        collection_task = asyncio.create_task(
            self._collect_run_output(process, limits.output_limit_bytes)
        )
        timed_out = False
        try:
            collected = await asyncio.wait_for(
                asyncio.shield(collection_task), timeout=limits.timeout_s
            )
        except TimeoutError:
            if collection_task.done():
                collected = collection_task.result()
            else:
                timed_out = True
                collected = await self._remove_and_collect(
                    process, collection_task, container_name
                )
        except asyncio.CancelledError:
            if collection_task.done():
                with suppress(BaseException):
                    collection_task.result()
            else:
                with suppress(BaseException):
                    await self._remove_and_collect(
                        process, collection_task, container_name
                    )
            raise
        except BaseException:
            if not collection_task.done():
                await self._remove_and_collect(process, collection_task, container_name)
            else:
                await self._remove_container(container_name)
            raise

        output = collected.output.decode("utf-8", errors="replace")
        returncode = process.returncode if process.returncode is not None else -1
        if returncode == 125:
            raise SandboxUnavailableError("docker", "startup-failed")
        if returncode != 0 and self._is_daemon_connection_failure(collected.output):
            raise SandboxUnavailableError("docker", "daemon-unreachable")
        return ExecResult(
            returncode, output, timed_out=timed_out, truncated=collected.truncated
        )
