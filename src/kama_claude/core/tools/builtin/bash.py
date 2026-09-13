from __future__ import annotations

import os
from collections.abc import Collection
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.sandbox import (
    ExecRequest,
    SandboxBackend,
    SandboxLimits,
    SandboxUnavailableError,
    WorkspaceFS,
    WorkspaceViolationError,
    build_sandbox_env,
)
from kama_claude.core.tools.base import BaseTool, ToolResult

_DEFAULT_TIMEOUT = 60


class BashParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    command: str
    timeout: int = Field(default=_DEFAULT_TIMEOUT, ge=1, le=120)
    cwd: str = "."


class BashTool(BaseTool):
    params_model = BashParams
    name = "bash"
    description = (
        "Execute a shell command in the configured sandbox and return its output "
        "(stdout + stderr combined). Non-interactive only — commands requiring user "
        "input will hang and time out. Prefer short, focused commands."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Maximum seconds to wait (default {_DEFAULT_TIMEOUT}, max 120).",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory relative to the workspace (default '.').",
            },
        },
        "required": ["command"],
    }

    # 固定 Bash 工具共享的工作区、执行后端、资源上限与运行环境
    def __init__(
        self,
        workspace: WorkspaceFS,
        backend: SandboxBackend,
        limits: SandboxLimits,
        env_allowlist: Collection[str],
        runtime_dir: Path,
    ) -> None:
        self._workspace = workspace
        self._backend = backend
        self._limits = limits
        self._env_allowlist = tuple(env_allowlist)
        self._runtime_dir = runtime_dir.resolve()

    # 将后端输出转换为与既有 Bash 工具兼容的用户可见文本
    @staticmethod
    def _format_output(output: str, *, truncated: bool) -> str:
        if truncated:
            return f"{output}\n[truncated]"
        return output

    # 仅公开静态沙箱能力与相对工作目录，不暴露命令和执行环境
    def execution_metadata(self, params: dict[str, object]) -> dict[str, object]:
        parsed = BashParams.model_validate(params)
        return {
            "backend": self._backend.name,
            "strongly_isolated": self._backend.strongly_isolated,
            "cwd": parsed.cwd,
            "workspace_access": "rw",
            "network": "on" if self._backend.network_enabled else "off",
        }

    # 校验工作目录并将命令委托给注入的沙箱后端
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = BashParams.model_validate(params)
        effective_limits = replace(
            self._limits, timeout_s=min(p.timeout, self._limits.timeout_s)
        )

        try:
            cwd = self._workspace.resolve(p.cwd, must_exist=True)
            if not cwd.is_dir():
                raise WorkspaceViolationError("bash cwd must be a directory")
            env = build_sandbox_env(
                os.environ,
                self._env_allowlist,
                self._runtime_dir / "home",
                self._runtime_dir / "tmp",
            )
            request = ExecRequest(command=p.command, cwd=cwd, env=env)
            result = await self._backend.execute(request, effective_limits)
        except WorkspaceViolationError as exc:
            return ToolResult(
                content=str(exc), is_error=True, error_type="sandbox_violation"
            )
        except SandboxUnavailableError as exc:
            return ToolResult(
                content=str(exc), is_error=True, error_type="sandbox_unavailable"
            )
        except Exception:
            return ToolResult(
                content="sandbox execution failed",
                is_error=True,
                error_type="runtime_error",
            )

        if result.timed_out:
            return ToolResult(
                content=f"[timeout after {effective_limits.timeout_s}s]",
                is_error=True,
                error_type="timeout",
            )

        output = self._format_output(result.output, truncated=result.truncated)
        if result.returncode != 0:
            return ToolResult(
                content=f"[exit {result.returncode}]\n{output}",
                is_error=True,
                error_type="command_error",
            )
        return ToolResult(content=output or "[no output]")
