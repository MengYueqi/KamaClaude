"""Compatibility backend that executes constrained commands on the host."""

import asyncio
import os
import signal
from pathlib import Path

from kama_claude.core.sandbox.base import SandboxBackend
from kama_claude.core.sandbox.models import ExecRequest, ExecResult, SandboxLimits


class HostBackend(SandboxBackend):
    """Execute commands in a dedicated host process group without strong isolation."""

    # 保存兼容后端的工作目录标识，供其生命周期与调用方保持一致
    def __init__(self, work_dir: Path) -> None:
        self._work_dir = work_dir

    # 返回兼容模式后端的稳定名称
    @property
    def name(self) -> str:
        return "host"

    # 表明宿主执行只提供弱隔离
    @property
    def strongly_isolated(self) -> bool:
        return False

    # 向进程组发送信号并忽略进程恰好退出的竞态
    @staticmethod
    def _signal_process_group(pid: int, sig: signal.Signals) -> None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return

    # 在超时后终止进程组并收割仍在排空输出的通信任务
    async def _terminate_and_collect(
        self, pid: int, communicate_task: asyncio.Task[tuple[bytes, bytes | None]]
    ) -> tuple[bytes, bytes | None]:
        self._signal_process_group(pid, signal.SIGTERM)
        try:
            return await asyncio.wait_for(asyncio.shield(communicate_task), timeout=1.0)
        except TimeoutError:
            self._signal_process_group(pid, signal.SIGKILL)
            return await communicate_task

    # 在兼容模式下执行命令并标准化超时、输出和退出码
    async def execute(self, request: ExecRequest, limits: SandboxLimits) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-lc",
            request.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=request.cwd,
            env=request.env,
            start_new_session=True,
        )
        communicate_task = asyncio.create_task(proc.communicate())
        timed_out = False
        try:
            try:
                output_bytes, _ = await asyncio.wait_for(
                    asyncio.shield(communicate_task), timeout=limits.timeout_s
                )
            except TimeoutError:
                timed_out = True
                output_bytes, _ = await self._terminate_and_collect(proc.pid, communicate_task)
        except BaseException:
            if not communicate_task.done():
                self._signal_process_group(proc.pid, signal.SIGKILL)
                await asyncio.shield(communicate_task)
            raise

        truncated = len(output_bytes) > limits.output_limit_bytes
        output = output_bytes[: limits.output_limit_bytes].decode("utf-8", errors="replace")
        returncode = proc.returncode if proc.returncode is not None else -1
        return ExecResult(returncode, output, timed_out=timed_out, truncated=truncated)
