"""Sandbox backend selection from validated runtime configuration."""

from pathlib import Path
from typing import cast

from kama_claude.core.config import SandboxConfig
from kama_claude.core.sandbox.base import SandboxBackend
from kama_claude.core.sandbox.host import HostBackend
from kama_claude.core.sandbox.workspace import WorkspaceFS


# 根据显式配置创建对应后端，绝不将 Docker 静默降级为 Host
def create_sandbox_backend(
    config: SandboxConfig, workspace: WorkspaceFS, runtime_dir: Path
) -> SandboxBackend:
    if config.backend == "host":
        return HostBackend(runtime_dir)
    if config.backend == "docker":
        from kama_claude.core.sandbox.docker import DockerBackend  # type: ignore[import-not-found]

        return cast(
            SandboxBackend,
            DockerBackend(workspace, config.docker_image, config.network, runtime_dir),
        )
    raise ValueError(f"unsupported sandbox backend: {config.backend}")
