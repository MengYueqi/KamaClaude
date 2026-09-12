"""Strongly isolated command execution through the Docker CLI."""

import asyncio
import os
import secrets
from contextlib import suppress
from pathlib import Path

from kama_claude.core.sandbox.base import SandboxBackend
from kama_claude.core.sandbox.errors import SandboxUnavailableError
from kama_claude.core.sandbox.models import ExecRequest, ExecResult, SandboxLimits
from kama_claude.core.sandbox.workspace import WorkspaceFS

_CLEANUP_TIMEOUT_S = 5.0
_RUN_REAP_TIMEOUT_S = 2.0
_KILL_REAP_TIMEOUT_S = 1.0


class _ProcessTimeoutError(TimeoutError):
    """Internal signal that a local Docker CLI exceeded its lifecycle bound."""


class DockerBackend(SandboxBackend):
    """Execute commands in a locked-down Docker container."""

    # 固定 Docker 执行所需的可信工作区、镜像和网络策略
    def __init__(
        self, workspace: WorkspaceFS, image: str, network: bool, runtime_dir: Path
    ) -> None:
        self._workspace = workspace
        self._image = image
        self._network = network

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

    # 杀死仍存活的本地 Docker CLI，并以有限等待加最终任务取消完成收割
    @staticmethod
    async def _kill_and_reap(
        process: asyncio.subprocess.Process,
        communicate_task: asyncio.Task[tuple[bytes, bytes | None]],
    ) -> None:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        if communicate_task.done():
            with suppress(BaseException):
                communicate_task.result()
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=_KILL_REAP_TIMEOUT_S
            )
        except BaseException:
            if not communicate_task.done():
                communicate_task.cancel()
            with suppress(BaseException):
                await communicate_task

    # 在正数 deadline 内排空本地 CLI，超界时先 kill 再确保通信任务结束
    async def _bounded_communicate(
        self,
        process: asyncio.subprocess.Process,
        communicate_task: asyncio.Task[tuple[bytes, bytes | None]],
        timeout_s: float,
    ) -> tuple[bytes, bytes | None]:
        try:
            return await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=timeout_s
            )
        except asyncio.CancelledError:
            await self._kill_and_reap(process, communicate_task)
            raise
        except TimeoutError:
            if communicate_task.done():
                return communicate_task.result()
            await self._kill_and_reap(process, communicate_task)
            raise _ProcessTimeoutError from None

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
            await self._bounded_communicate(cleanup, cleanup_task, _CLEANUP_TIMEOUT_S)
        except asyncio.CancelledError:
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
        communicate_task: asyncio.Task[tuple[bytes, bytes | None]],
        container_name: str,
    ) -> tuple[bytes, bytes | None]:
        cleanup_failure: BaseException | None = None
        communication_failure: BaseException | None = None
        result: tuple[bytes, bytes | None] | None = None
        try:
            await self._remove_container(container_name)
        except BaseException as error:
            cleanup_failure = error
        try:
            result = await self._bounded_communicate(
                process, communicate_task, _RUN_REAP_TIMEOUT_S
            )
        except _ProcessTimeoutError:
            communication_failure = SandboxUnavailableError("docker", "cleanup-failed")
        except BaseException as error:
            await self._kill_and_reap(process, communicate_task)
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

        communicate_task = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            output_bytes, _ = await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=limits.timeout_s
            )
        except TimeoutError:
            timed_out = True
            output_bytes, _ = await self._remove_and_collect(
                process, communicate_task, container_name
            )
        except asyncio.CancelledError:
            if communicate_task.done():
                with suppress(BaseException):
                    communicate_task.result()
            else:
                with suppress(BaseException):
                    await self._remove_and_collect(
                        process, communicate_task, container_name
                    )
            raise
        except BaseException:
            if not communicate_task.done():
                await self._remove_and_collect(process, communicate_task, container_name)
            else:
                await self._remove_container(container_name)
            raise

        truncated = len(output_bytes) > limits.output_limit_bytes
        output = output_bytes[: limits.output_limit_bytes].decode("utf-8", errors="replace")
        returncode = process.returncode if process.returncode is not None else -1
        if returncode == 125:
            raise SandboxUnavailableError("docker", "startup-failed")
        if returncode != 0 and self._is_daemon_connection_failure(output_bytes):
            raise SandboxUnavailableError("docker", "daemon-unreachable")
        return ExecResult(returncode, output, timed_out=timed_out, truncated=truncated)
