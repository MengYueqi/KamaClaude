"""Abstract execution contract shared by sandbox backends."""

from abc import ABC, abstractmethod

from kama_claude.core.sandbox.models import ExecRequest, ExecResult, SandboxLimits


class SandboxBackend(ABC):
    """Backend interface for executing one command within sandbox policy."""

    # 返回后端的稳定名称，供日志和错误信息使用
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend's stable display name."""

    # 返回后端是否提供强隔离能力
    @property
    @abstractmethod
    def strongly_isolated(self) -> bool:
        """Return whether the backend enforces strong isolation."""

    # 在后端中执行请求并返回标准化结果
    @abstractmethod
    async def execute(self, request: ExecRequest, limits: SandboxLimits) -> ExecResult:
        """Execute a command under the supplied resource limits."""

    # 释放后端资源；默认后端无需清理
    async def close(self) -> None:
        """Close backend resources when the runner shuts down."""
        return None
