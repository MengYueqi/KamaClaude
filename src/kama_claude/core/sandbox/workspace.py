"""Safe path resolution within an agent workspace."""

from pathlib import Path

from kama_claude.core.sandbox.errors import WorkspaceViolationError


class WorkspaceFS:
    """Resolve user-provided relative paths inside a fixed workspace root."""

    # 固定并校验当前 Agent 可访问的 Workspace 根目录
    def __init__(self, root: Path) -> None:
        resolved = root.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"workspace root is not a directory: {resolved}")
        self._root = resolved

    # 返回规范化后的 Workspace 根目录
    @property
    def root(self) -> Path:
        return self._root

    # 将用户相对路径解析为 Workspace 内规范路径，越界时拒绝
    def resolve(self, relative_path: str, *, must_exist: bool = False) -> Path:
        supplied = Path(relative_path)
        if supplied.is_absolute():
            raise WorkspaceViolationError("absolute paths are not allowed")
        resolved = (self._root / supplied).resolve(strict=False)
        if not resolved.is_relative_to(self._root):
            raise WorkspaceViolationError("path resolves outside workspace")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(relative_path)
        return resolved

    # 将已校验的绝对路径转换为 Workspace 相对 POSIX 路径
    def relative(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(self._root):
            raise WorkspaceViolationError("path resolves outside workspace")
        return resolved.relative_to(self._root).as_posix() or "."
