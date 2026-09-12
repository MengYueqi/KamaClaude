"""Agent workspace sandbox primitives."""

from kama_claude.core.sandbox.base import SandboxBackend
from kama_claude.core.sandbox.environment import build_sandbox_env
from kama_claude.core.sandbox.errors import SandboxUnavailableError, WorkspaceViolationError
from kama_claude.core.sandbox.factory import create_sandbox_backend
from kama_claude.core.sandbox.models import ExecRequest, ExecResult, SandboxLimits
from kama_claude.core.sandbox.workspace import WorkspaceFS

__all__ = [
    "ExecRequest",
    "ExecResult",
    "SandboxBackend",
    "SandboxLimits",
    "SandboxUnavailableError",
    "WorkspaceFS",
    "WorkspaceViolationError",
    "build_sandbox_env",
    "create_sandbox_backend",
]
