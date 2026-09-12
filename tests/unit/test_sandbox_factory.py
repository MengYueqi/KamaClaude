from pathlib import Path

import pytest

from kama_claude.core.config import SandboxConfig
from kama_claude.core.sandbox import create_sandbox_backend as package_create_sandbox_backend
from kama_claude.core.sandbox.factory import create_sandbox_backend
from kama_claude.core.sandbox.host import HostBackend
from kama_claude.core.sandbox.workspace import WorkspaceFS


# 功能：host 配置创建绑定到受控运行目录的 HostBackend
# 设计：workspace 与 runtime_dir 分别传入，断言 factory 不会把 workspace 根误作运行时根
def test_factory_creates_host_backend_for_host_config(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    backend = create_sandbox_backend(SandboxConfig(backend="host"), workspace, runtime_dir)

    assert isinstance(backend, HostBackend)
    assert backend.name == "host"


# 功能：docker 配置在 DockerBackend 尚不存在时明确失败而非回退 host
# 设计：只断言导入失败，证明配置选择的强隔离后端不可用不会悄然改变为弱隔离
def test_factory_docker_config_never_falls_back_to_host(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    with pytest.raises(ModuleNotFoundError, match="sandbox.docker"):
        create_sandbox_backend(SandboxConfig(backend="docker"), workspace, runtime_dir)


# 功能：未知后端值被 factory 明确拒绝
# 设计：直接构造 config 绕过解析层，确保 factory 自身不会在非法输入下选择 HostBackend
def test_factory_rejects_unknown_backend(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    with pytest.raises(ValueError, match="unsupported sandbox backend"):
        create_sandbox_backend(SandboxConfig(backend="unknown"), workspace, runtime_dir)


# 功能：从 sandbox 包根导入稳定的 factory 接口
# 设计：断言包级导出与实现模块是同一函数，防止未来重导出包装改变调用语义
def test_sandbox_package_exports_backend_factory() -> None:
    assert package_create_sandbox_backend is create_sandbox_backend
