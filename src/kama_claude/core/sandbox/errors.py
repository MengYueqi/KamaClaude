"""Errors raised by workspace sandbox boundaries."""


class WorkspaceViolationError(ValueError):
    """Raised when a path escapes the configured workspace."""
