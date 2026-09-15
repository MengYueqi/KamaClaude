from pathlib import Path

import pytest

from kama_claude.core.sandbox.errors import WorkspaceViolationError
from kama_claude.core.sandbox.workspace import WorkspaceFS


# 功能：验证普通相对路径被解析到规范化 Workspace 根目录下
# 设计：使用 tmp_path 避免依赖仓库目录，并断言返回绝对规范路径
def test_resolve_relative_path_inside_workspace(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    assert workspace.resolve("src/main.py") == tmp_path.resolve() / "src/main.py"


# 功能：验证绝对路径即使位于 Workspace 内也被拒绝
# 设计：禁止绝对路径可避免模型把宿主路径带入工具协议
def test_rejects_absolute_path(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    with pytest.raises(WorkspaceViolationError, match="absolute"):
        workspace.resolve(str(tmp_path / "file.txt"))


# 功能：验证父目录遍历不能离开 Workspace
# 设计：直接覆盖最常见的 ../ 越界输入
def test_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    with pytest.raises(WorkspaceViolationError, match="outside workspace"):
        workspace.resolve("../secret.txt")


# 功能：验证指向 Workspace 外部的符号链接不能逃逸
# 设计：链接位于工作区但目标位于同级目录，必须按 resolve 后路径判断
def test_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    workspace = WorkspaceFS(root)
    with pytest.raises(WorkspaceViolationError, match="outside workspace"):
        workspace.resolve("link/secret.txt")


# 功能：验证 must_exist 对不存在目标给出 FileNotFoundError
# 设计：存在性与越界错误分开，便于工具层保持现有错误分类
def test_must_exist_rejects_missing_path(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    with pytest.raises(FileNotFoundError):
        workspace.resolve("missing.txt", must_exist=True)


# 功能：验证当前目录和规范化嵌套相对路径的解析
# 设计：覆盖点路径及 a/../b 这类合法规范化输入
def test_resolve_normalizes_relative_paths(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    assert workspace.resolve(".") == tmp_path.resolve()
    assert workspace.resolve("a/../b") == tmp_path.resolve() / "b"


# 功能：验证工作区内部符号链接可正常解析
# 设计：链接目标仍在根目录内时应允许访问规范目标路径
def test_allows_internal_symlink(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "target", target_is_directory=True)
    workspace = WorkspaceFS(tmp_path)
    assert workspace.resolve("link") == tmp_path.resolve() / "target"


# 功能：验证工作区根目录必须存在且必须是目录
# 设计：初始化时尽早拒绝无效边界，避免后续产生不明确错误
def test_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        WorkspaceFS(tmp_path / "missing")


# 功能：验证文件不能作为工作区根目录
# 设计：WorkspaceFS 只接受目录作为可访问边界
def test_rejects_file_root(tmp_path: Path) -> None:
    root = tmp_path / "root-file"
    root.touch()
    with pytest.raises(ValueError, match="not a directory"):
        WorkspaceFS(root)


# 功能：验证绝对路径可转换为工作区相对 POSIX 路径
# 设计：相对转换结果供工具协议稳定传递且统一使用斜杠
def test_relative_returns_workspace_relative_posix_path(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    assert workspace.relative(tmp_path / "a" / "b.txt") == "a/b.txt"
    assert workspace.relative(tmp_path) == "."


# 功能：验证 relative 对工作区外路径执行边界检查
# 设计：反向转换同样不得泄漏工作区之外的宿主路径
def test_relative_rejects_outside_path(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = WorkspaceFS(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(WorkspaceViolationError, match="outside workspace"):
        workspace.relative(outside)


# 功能：Workspace 越界错误保持标准权限错误语义
# 设计：继承回归防止调用方的 PermissionError 捕获因错误基类变化而失效
def test_workspace_violation_error_is_permission_error() -> None:
    assert issubclass(WorkspaceViolationError, PermissionError)
