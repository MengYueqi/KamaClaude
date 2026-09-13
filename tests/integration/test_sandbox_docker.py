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


@pytest.fixture(scope="module", autouse=True)
def require_docker_sandbox() -> None:
    """Keep collection inert unless the explicit Docker integration opt-in is set."""
    if os.environ.get("KAMA_TEST_DOCKER_SANDBOX") != "1":
        pytest.skip("set KAMA_TEST_DOCKER_SANDBOX=1 to run Docker sandbox tests")
    _run_docker_check("info", description="checking the Docker daemon")
    _run_docker_check("image", "inspect", _IMAGE, description=f"checking image {_IMAGE}")


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceFS:
    root = tmp_path / "dedicated-sandbox-workspace"
    root.mkdir()
    return WorkspaceFS(root)


@pytest.fixture
def backend(workspace: WorkspaceFS, tmp_path: Path) -> DockerBackend:
    return DockerBackend(workspace, _IMAGE, network=False, runtime_dir=tmp_path / "runtime")


def _request(workspace: WorkspaceFS, command: str) -> ExecRequest:
    return ExecRequest(command, workspace.root, {"PATH": _CONTAINER_PATH})


def _limits(*, timeout_s: int = 5, pids_limit: int = 32) -> SandboxLimits:
    return SandboxLimits(
        timeout_s=timeout_s,
        output_limit_bytes=8_192,
        memory_mb=128,
        cpu_count=1.0,
        pids_limit=pids_limit,
        tmpfs_mb=64,
    )


# Break caught: a changed container work directory exposes a host path instead of /workspace.
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_pwd_reports_workspace(backend: DockerBackend, workspace: WorkspaceFS) -> None:
    result = await backend.execute(_request(workspace, "pwd"), _limits())

    assert result.returncode == 0
    assert result.output.strip() == "/workspace"


# Break caught: a missing or read-only bind mount prevents sandbox commands from persisting workspace output.
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_echo_writes_file_inside_mounted_workspace(
    backend: DockerBackend, workspace: WorkspaceFS
) -> None:
    result = await backend.execute(_request(workspace, "printf sandbox-data > created.txt"), _limits())

    assert result.returncode == 0
    assert (workspace.root / "created.txt").read_text(encoding="utf-8") == "sandbox-data"


# Break caught: inheriting the host environment leaks an API-like secret into the container.
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_host_only_secret_is_absent_inside_container(
    backend: DockerBackend, workspace: WorkspaceFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOST_ONLY_SECRET", "must-not-leak")
    command = 'if [ -z "${HOST_ONLY_SECRET+x}" ]; then printf absent; else printf present; exit 1; fi'

    result = await backend.execute(_request(workspace, command), _limits())

    assert result.returncode == 0
    assert result.output == "absent"


# Break caught: adding an unintended host mount makes a sibling host file readable in the container.
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_host_only_file_path_is_not_readable(
    backend: DockerBackend, workspace: WorkspaceFS, tmp_path: Path
) -> None:
    host_only = tmp_path / "host-only-secret.txt"
    host_only.write_text("not-mounted", encoding="utf-8")

    result = await backend.execute(
        _request(workspace, f"cat {shlex.quote(str(host_only))}"), _limits()
    )

    assert result.returncode != 0
    assert "not-mounted" not in result.output


# Break caught: mapping network=false to a reachable Docker network permits outbound requests.
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_network_request_fails_when_network_is_disabled(
    backend: DockerBackend, workspace: WorkspaceFS
) -> None:
    command = "python -c " + shlex.quote(
        "import urllib.request; urllib.request.urlopen('https://example.com', timeout=3).read()"
    )

    result = await backend.execute(_request(workspace, command), _limits())

    assert result.returncode != 0


# Break caught: removing or increasing --pids-limit allows every requested live child to start.
@pytest.mark.integration
@pytest.mark.docker_sandbox
async def test_fork_workload_cannot_exceed_pids_limit(
    backend: DockerBackend, workspace: WorkspaceFS
) -> None:
    requested_children = 64
    command = "python -c " + shlex.quote(
        "import os, signal, sys, time\n"
        f"requested = {requested_children}\n"
        "children = []\n"
        "try:\n"
        "    for _ in range(requested):\n"
        "        child = os.fork()\n"
        "        if child == 0:\n"
        "            time.sleep(30)\n"
        "            os._exit(0)\n"
        "        children.append(child)\n"
        "except OSError:\n"
        "    print(f'pids-limit-enforced:{len(children)}')\n"
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

    result = await backend.execute(_request(workspace, command), _limits(pids_limit=8))

    assert result.returncode == 0
    marker, created = result.output.strip().split(":", maxsplit=1)
    assert marker == "pids-limit-enforced"
    assert 0 <= int(created) < requested_children


# Break caught: timeout cleanup leaves the backend-created container alive after the request returns.
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

    result = await backend.execute(
        _request(workspace, "python -c 'import time; time.sleep(30)'"),
        _limits(timeout_s=1),
    )

    assert result.timed_out is True
    completed = subprocess.run(
        ["docker", "container", "inspect", container_name],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
