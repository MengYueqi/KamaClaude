"""Opt-in integration coverage for the Docker sandbox boundary.

Run explicitly after building the isolated runtime image:
    docker build --target sandbox-runtime -t kama-sandbox:py312 .
    KAMA_TEST_DOCKER_SANDBOX=1 uv run pytest tests/integration/test_sandbox_docker.py -v
"""

from __future__ import annotations

import os
import shlex
import subprocess
import uuid
from pathlib import Path

import pytest

from kama_claude.core.sandbox import ExecRequest, SandboxLimits, WorkspaceFS
from kama_claude.core.sandbox.docker import DockerBackend

_IMAGE = "kama-sandbox:py312"
_CONTAINER_PATH = "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


# 功能：在显式启用时验证 Docker CLI、守护进程和测试镜像均可用。
def _run_docker_check(*args: str, description: str) -> None:
    """Fail explicitly when the opted-in Docker prerequisite is unavailable."""
    try:
        completed = subprocess.run(
            ["docker", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        pytest.fail(f"Docker CLI is unavailable while {description}")
    except subprocess.TimeoutExpired:
        pytest.fail(f"Docker command timed out while {description}")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        pytest.fail(f"Docker is unavailable while {description}: {detail}")


# 功能：未显式启用时跳过测试，启用后验证 Docker 前置条件。
@pytest.fixture(scope="module", autouse=True)
def require_docker_sandbox() -> None:
    """Keep collection inert unless the explicit Docker integration opt-in is set."""
    if os.environ.get("KAMA_TEST_DOCKER_SANDBOX") != "1":
        pytest.skip("set KAMA_TEST_DOCKER_SANDBOX=1 to run Docker sandbox tests")
    _run_docker_check("info", description="checking the Docker daemon")
    _run_docker_check("image", "inspect", _IMAGE, description=f"checking image {_IMAGE}")


# 功能：为每个测试创建独立且可写的临时工作区边界。
@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceFS:
    root = tmp_path / "dedicated-sandbox-workspace"
    root.mkdir()
    return WorkspaceFS(root)


# 功能：构造固定禁网策略的真实 Docker 沙箱后端。
@pytest.fixture
def backend(workspace: WorkspaceFS, tmp_path: Path) -> DockerBackend:
    return DockerBackend(workspace, _IMAGE, network=False, runtime_dir=tmp_path / "runtime")


# 功能：为容器命令创建只含可信 PATH 的执行请求。
def _request(workspace: WorkspaceFS, command: str) -> ExecRequest:
    return ExecRequest(command, workspace.root, {"PATH": _CONTAINER_PATH})


# 功能：集中生成测试使用的受限资源配额。
def _limits(*, timeout_s: int = 5, pids_limit: int = 32) -> SandboxLimits:
    return SandboxLimits(
        timeout_s=timeout_s,
        output_limit_bytes=8_192,
        memory_mb=128,
        cpu_count=1.0,
        pids_limit=pids_limit,
        tmpfs_mb=64,
    )


# 功能：验证容器工作目录固定为挂载的 /workspace。
# 设计：执行真实 pwd 并断言精确路径，防止工作目录回退到镜像或宿主路径。
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_pwd_reports_workspace(backend: DockerBackend, workspace: WorkspaceFS) -> None:
    result = await backend.execute(_request(workspace, "pwd"), _limits())

    assert result.returncode == 0
    assert result.output.strip() == "/workspace"


# 功能：验证容器可在挂载工作区中写入文件。
# 设计：在容器内写入后由宿主临时工作区读取，覆盖挂载缺失或只读两类回归。
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_echo_writes_file_inside_mounted_workspace(
    backend: DockerBackend, workspace: WorkspaceFS
) -> None:
    result = await backend.execute(
        _request(workspace, "printf sandbox-data > created.txt"), _limits()
    )

    assert result.returncode == 0
    assert (workspace.root / "created.txt").read_text(encoding="utf-8") == "sandbox-data"


# 功能：验证仅存在于宿主环境的秘密不会进入容器。
# 设计：在宿主设置秘密后由容器检查变量是否未定义，防止环境继承泄漏。
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_host_only_secret_is_absent_inside_container(
    backend: DockerBackend, workspace: WorkspaceFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOST_ONLY_SECRET", "must-not-leak")
    command = (
        'if [ -z "${HOST_ONLY_SECRET+x}" ]; then printf absent; else printf present; exit 1; fi'
    )

    result = await backend.execute(_request(workspace, command), _limits())

    assert result.returncode == 0
    assert result.output == "absent"


# 功能：验证工作区同级的宿主文件无法在容器中读取。
# 设计：使用镜像自带 Python pathlib 读取未挂载路径，避免依赖未声明的外部命令。
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_host_only_file_path_is_not_readable(
    backend: DockerBackend, workspace: WorkspaceFS, tmp_path: Path
) -> None:
    host_only = tmp_path / "host-only-secret.txt"
    host_only.write_text("not-mounted", encoding="utf-8")

    command = "python -c " + shlex.quote(
        f"from pathlib import Path; print(Path({str(host_only)!r}).read_text())"
    )
    result = await backend.execute(_request(workspace, command), _limits())

    assert result.returncode != 0
    assert "not-mounted" not in result.output


# 功能：验证禁网容器只暴露回环网卡且无法发起外部请求。
# 设计：先读取真实 /sys/class/net 断言仅有 lo，再保留真实 HTTPS 请求失败断言。
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_network_request_fails_when_network_is_disabled(
    backend: DockerBackend, workspace: WorkspaceFS
) -> None:
    interfaces_command = "python -c " + shlex.quote(
        "from pathlib import Path; print('\\n'.join(sorted(path.name for path in Path('/sys/class/net').iterdir())))"
    )
    interfaces = await backend.execute(_request(workspace, interfaces_command), _limits())

    assert interfaces.returncode == 0
    assert set(interfaces.output.splitlines()) == {"lo"}

    request_command = "python -c " + shlex.quote(
        "import urllib.request; urllib.request.urlopen('https://example.com', timeout=3).read()"
    )

    result = await backend.execute(_request(workspace, request_command), _limits())

    assert result.returncode != 0


# 功能：验证 PID 配额阻止超出上限的并发子进程工作负载。
# 设计：预留启动进程空间后请求远超配额的存活子进程，只接受 EAGAIN 并在 finally 中回收全部子进程。
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_fork_workload_cannot_exceed_pids_limit(
    backend: DockerBackend, workspace: WorkspaceFS
) -> None:
    requested_children = 64
    pids_limit = 8
    command = "python -c " + shlex.quote(
        "import errno, os, signal, sys, time\n"
        f"requested = {requested_children}\n"
        "children = []\n"
        "try:\n"
        "    for _ in range(requested):\n"
        "        child = os.fork()\n"
        "        if child == 0:\n"
        "            time.sleep(30)\n"
        "            os._exit(0)\n"
        "        children.append(child)\n"
        "except OSError as error:\n"
        "    print(f'pids-limit-enforced:{errno.errorcode.get(error.errno)}:{len(children)}')\n"
        "else:\n"
        "    print('pids-limit-not-enforced')\n"
        "    sys.exit(1)\n"
        "finally:\n"
        "    for child in children:\n"
        "        try:\n"
        "            os.kill(child, signal.SIGTERM)\n"
        "        except ProcessLookupError:\n"
        "            pass\n"
        "    for child in children:\n"
        "        try:\n"
        "            os.waitpid(child, 0)\n"
        "        except ChildProcessError:\n"
        "            pass\n"
    )

    result = await backend.execute(_request(workspace, command), _limits(pids_limit=pids_limit))

    assert result.returncode == 0
    marker, errno_name, created = result.output.strip().split(":", maxsplit=2)
    assert marker == "pids-limit-enforced"
    assert errno_name == "EAGAIN"
    assert 0 < int(created) < pids_limit


# 功能：验证超时后后端创建的精确命名容器已不存在。
# 设计：以 UUID 名称执行超时请求并用成功的精确 Docker 查询断言缺席，finally 仅清理该名称。
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_timeout_removes_the_named_container(
    backend: DockerBackend, workspace: WorkspaceFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    container_name = f"kama-task10-timeout-{uuid.uuid4().hex}"
    monkeypatch.setattr(
        "kama_claude.core.sandbox.docker.secrets.token_hex",
        lambda _: container_name.removeprefix("kama-"),
    )

    try:
        result = await backend.execute(
            _request(workspace, "python -c 'import time; time.sleep(30)'"),
            _limits(timeout_s=1),
        )

        assert result.timed_out is True
        completed = subprocess.run(
            [
                "docker",
                "ps",
                "--all",
                "--filter",
                f"name=^/{container_name}$",
                "--format",
                "{{.Names}}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0
        assert container_name not in completed.stdout.splitlines()
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
