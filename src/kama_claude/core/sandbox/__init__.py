"""Agent workspace sandbox primitives."""

from kama_claude.core.sandbox.errors import WorkspaceViolationError
from kama_claude.core.sandbox.workspace import WorkspaceFS

__all__ = ["WorkspaceFS", "WorkspaceViolationError"]
