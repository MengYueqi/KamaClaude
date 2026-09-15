"""Errors raised by sandbox boundaries and unavailable backends."""


class WorkspaceViolationError(PermissionError):
    """Raised when a path escapes the configured workspace."""


class SandboxUnavailableError(RuntimeError):
    """Raised when a configured sandbox backend cannot provide execution."""

    # 保存稳定的后端与原因字段，同时避免将外部诊断或 Secret 拼入错误消息
    def __init__(self, backend: str, reason: str) -> None:
        self.backend = backend
        self.reason = reason
        super().__init__(f"{backend} sandbox unavailable ({reason})")
