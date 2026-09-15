"""Immutable request, result, and resource-limit models for sandbox execution."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxLimits:
    """Hard resource limits applied to one sandbox execution."""

    timeout_s: int
    output_limit_bytes: int
    memory_mb: int
    cpu_count: float
    pids_limit: int
    tmpfs_mb: int


@dataclass(frozen=True)
class ExecRequest:
    """Command and process context supplied to a sandbox backend."""

    command: str
    cwd: Path
    env: dict[str, str]


@dataclass(frozen=True)
class ExecResult:
    """Normalized result returned by a sandbox backend."""

    returncode: int
    output: str
    timed_out: bool = False
    truncated: bool = False
