from pathlib import Path

import pytest

from kama_claude.core.config import SandboxConfig
from kama_claude.core.sandbox import create_sandbox_backend as package_create_sandbox_backend
from kama_claude.core.sandbox.docker import DockerBackend
from kama_claude.core.sandbox.factory import create_sandbox_backend
from kama_claude.core.sandbox.host import HostBackend
from kama_claude.core.sandbox.models import ExecRequest, SandboxLimits
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


# 功能：docker 配置创建精确配置的强隔离 DockerBackend
# 设计：具体类型和稳定属性共同防止 factory 将 docker 静默降级为 HostBackend
def test_factory_docker_config_never_falls_back_to_host(tmp_path: Path) -> None:
    workspace = WorkspaceFS(tmp_path)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    backend = create_sandbox_backend(
        SandboxConfig(backend="docker", docker_image="isolated:latest", network=True),
        workspace,
        runtime_dir,
    )

    assert isinstance(backend, DockerBackend)
    assert backend.name == "docker"
    assert backend.strongly_isolated is True
    argv = backend._build_argv(
        ExecRequest("true", tmp_path, {}),
        SandboxLimits(5, 1_024, 64, 1.0, 32, 16),
        "kama-0123456789abcdef",
    )
    assert argv[argv.index("--network") + 1] == "bridge"
    assert argv[argv.index("--mount") + 1] == (
        f"type=bind,src={tmp_path.resolve()},dst=/workspace,rw"
    )
    assert argv[-4] == "isolated:latest"


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
