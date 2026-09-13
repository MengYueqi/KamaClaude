from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

import kama_claude.core.app as app_module
from kama_claude.core.config import KamaConfig, McpServerConfig
from kama_claude.core.sandbox import (
    ExecRequest,
    ExecResult,
    SandboxBackend,
    SandboxLimits,
    WorkspaceFS,
)


class _Backend(SandboxBackend):
    def __init__(
        self,
        events: list[str],
        *,
        name: str,
        strongly_isolated: bool,
        close_error: Exception | None = None,
    ) -> None:
        self._events = events
        self._name = name
        self._strongly_isolated = strongly_isolated
        self._close_error = close_error
        self.close_calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def strongly_isolated(self) -> bool:
        return self._strongly_isolated

    async def execute(self, request: ExecRequest, limits: SandboxLimits) -> ExecResult:
        raise AssertionError("CoreApp lifecycle tests must not execute commands")

    async def close(self) -> None:
        self.close_calls += 1
        self._events.append("backend.close")
        if self._close_error is not None:
            raise self._close_error


class _ImmediateEvent:
    def set(self) -> None:
        return None

    async def wait(self) -> None:
        return None


class _SignalLoop:
    def add_signal_handler(self, sig: object, callback: Callable[[], None]) -> None:
        return None


class _Broadcaster:
    async def handle(self, event: object) -> None:
        return None


class _PermissionManager:
    def __init__(self, **kwargs: object) -> None:
        return None


class _Store:
    def __init__(self, root: Path) -> None:
        self.root = root


class _Provider:
    def __init__(self, model: str) -> None:
        self.model = model


class _Runner:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        type(self).calls.append((args, kwargs))


class _Sessions:
    runner_factory: Callable[[], object] | None = None

    def __init__(self, store: object, runner_factory: Callable[[], object], **kwargs: object):
        type(self).runner_factory = runner_factory
        runner_factory()
        runner_factory()


class _Trace:
    def __init__(self, path: Path, events: list[str]) -> None:
        self.path = path
        self._events = events

    async def start(self) -> None:
        self._events.append("trace.start")

    async def stop(self) -> None:
        self._events.append("trace.stop")


class _Mcp:
    def __init__(
        self,
        events: list[str],
        *,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        self._events = events
        self._start_error = start_error
        self._stop_error = stop_error

    async def start_all(self, servers: object) -> None:
        self._events.append("mcp.start")
        if self._start_error is not None:
            raise self._start_error

    async def stop_all(self) -> None:
        self._events.append("mcp.stop")
        if self._stop_error is not None:
            raise self._stop_error


class _Server:
    def __init__(
        self,
        events: list[str],
        *,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        self._events = events
        self._start_error = start_error
        self._stop_error = stop_error

    def register(self, method: str, handler: object) -> None:
        return None

    async def start(self) -> str:
        self._events.append("server.start")
        if self._start_error is not None:
            raise self._start_error
        return "test:7437"

    async def stop(self) -> None:
        self._events.append("server.stop")
        if self._stop_error is not None:
            raise self._stop_error


def _config(workspace_root: Path, *, backend: str = "host") -> KamaConfig:
    config = KamaConfig()
    config.trace.enabled = True
    config.sandbox.backend = backend
    config.sandbox.workspace_root = str(workspace_root)
    config.sandbox.network = False
    config.sandbox.timeout_s = 37
    config.sandbox.output_limit_bytes = 4_321
    config.sandbox.memory_mb = 768
    config.sandbox.cpu_count = 1.75
    config.sandbox.pids_limit = 57
    config.sandbox.tmpfs_mb = 91
    return config


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: KamaConfig,
    backend: _Backend,
    events: list[str],
    *,
    mcp_start_error: Exception | None = None,
    mcp_stop_error: Exception | None = None,
    server_start_error: Exception | None = None,
    server_stop_error: Exception | None = None,
) -> list[tuple[object, WorkspaceFS, Path]]:
    factory_calls: list[tuple[object, WorkspaceFS, Path]] = []
    _Runner.calls = []
    _Sessions.runner_factory = None
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KAMA_TEST_SECRET", "must-not-appear")
    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "setup_logging", lambda _: None)
    monkeypatch.setattr(app_module, "load_policy_file", lambda _: [])
    monkeypatch.setattr(app_module, "PermissionManager", _PermissionManager)
    monkeypatch.setattr(app_module, "IpcEventBroadcaster", lambda **_: _Broadcaster())
    monkeypatch.setattr(app_module, "SessionStore", _Store)
    monkeypatch.setattr(app_module, "AnthropicProvider", _Provider)
    monkeypatch.setattr(app_module, "AgentRunner", _Runner)
    monkeypatch.setattr(app_module, "SessionManager", _Sessions)
    monkeypatch.setattr(
        app_module,
        "TraceWriter",
        lambda path: _Trace(path, events),
    )
    monkeypatch.setattr(
        app_module,
        "McpServerManager",
        lambda: _Mcp(
            events,
            start_error=mcp_start_error,
            stop_error=mcp_stop_error,
        ),
    )
    monkeypatch.setattr(
        app_module,
        "SocketServer",
        lambda *args, **kwargs: _Server(
            events,
            start_error=server_start_error,
            stop_error=server_stop_error,
        ),
    )

    def _create_backend(
        sandbox_config: object, workspace: WorkspaceFS, runtime_dir: Path
    ) -> SandboxBackend:
        factory_calls.append((sandbox_config, workspace, runtime_dir))
        return backend

    monkeypatch.setattr(app_module, "create_sandbox_backend", _create_backend, raising=False)
    monkeypatch.setattr(app_module.asyncio, "Event", _ImmediateEvent)
    monkeypatch.setattr(app_module.asyncio, "get_running_loop", lambda: _SignalLoop())
    return factory_calls


# 功能：CoreApp 必须只构造一套已解析沙箱依赖，并把相同对象注入每个 Runner
# 设计：替换所有外部边界，在真实 CoreApp.run 生命周期内调用两次 runner_factory 并检查对象身份和完整限制映射
async def test_core_app_constructs_and_shares_one_sandbox_dependency_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_root = tmp_path / "workspace" / "project"
    workspace_root.mkdir(parents=True)
    config = _config(workspace_root.parent / "project" / ".")
    events: list[str] = []
    backend = _Backend(events, name="host", strongly_isolated=False)
    factory_calls = _install_fakes(monkeypatch, tmp_path, config, backend, events)

    await app_module.CoreApp().run()

    assert len(factory_calls) == 1
    sandbox_config, workspace, runtime_dir = factory_calls[0]
    assert sandbox_config is config.sandbox
    assert workspace.root == workspace_root.resolve()
    assert runtime_dir.is_absolute()
    assert runtime_dir.is_dir()
    assert runtime_dir == (tmp_path / "home" / ".kama" / "sessions" / ".sandbox").resolve()
    assert len(_Runner.calls) == 2
    first_kwargs = _Runner.calls[0][1]
    second_kwargs = _Runner.calls[1][1]
    assert first_kwargs["workspace"] is workspace
    assert second_kwargs["workspace"] is workspace
    assert first_kwargs["sandbox_backend"] is backend
    assert second_kwargs["sandbox_backend"] is backend
    assert first_kwargs["sandbox_limits"] is second_kwargs["sandbox_limits"]
    assert first_kwargs["sandbox_limits"] == SandboxLimits(
        timeout_s=37,
        output_limit_bytes=4_321,
        memory_mb=768,
        cpu_count=1.75,
        pids_limit=57,
        tmpfs_mb=91,
    )
    assert first_kwargs["sandbox_runtime_dir"] is runtime_dir
    assert second_kwargs["sandbox_runtime_dir"] is runtime_dir


# 功能：启动日志必须准确呈现后端隔离状态，Docker 额外显示网络状态，同时不得泄露环境值
# 设计：参数化两个静态后端，用 caplog 检查独立 sandbox 日志记录的完整文本而不依赖日志格式器
@pytest.mark.parametrize(
    ("backend_name", "isolated", "expected_suffix"),
    [
        ("host", False, ""),
        ("docker", True, " network=false"),
    ],
)
async def test_core_app_logs_exact_sanitized_sandbox_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    backend_name: str,
    isolated: bool,
    expected_suffix: str,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config = _config(workspace_root, backend=backend_name)
    config.mcp.servers = [
        McpServerConfig(
            name="sensitive-test",
            command="must-not-appear",
            env={"API_KEY": "must-not-appear"},
        )
    ]
    events: list[str] = []
    backend = _Backend(
        events,
        name=backend_name,
        strongly_isolated=isolated,
    )
    _install_fakes(monkeypatch, tmp_path, config, backend, events)
    caplog.set_level(logging.INFO, logger=app_module.__name__)

    await app_module.CoreApp().run()

    sandbox_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == app_module.__name__
        and record.getMessage().startswith("sandbox backend=")
    ]
    assert sandbox_messages == [
        "sandbox backend="
        f"{backend_name} strongly_isolated={str(isolated).lower()} "
        f"workspace={workspace_root.resolve()}{expected_suffix}"
    ]
    assert "must-not-appear" not in caplog.text


# 功能：正常关闭必须先收割运行任务，再关闭 backend，最后停止 trace，且 backend 只关闭一次
# 设计：放入真实待取消任务并用事件列表记录其 finally 与所有清理组件的可观察顺序
async def test_core_app_reaps_active_runs_before_single_backend_close_and_trace_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config = _config(workspace_root)
    events: list[str] = []
    backend = _Backend(events, name="host", strongly_isolated=False)
    _install_fakes(monkeypatch, tmp_path, config, backend, events)
    app = app_module.CoreApp()

    async def _active_run() -> None:
        try:
            await asyncio.Future()
        finally:
            events.append("run.reaped")

    run_task = asyncio.create_task(_active_run())
    await asyncio.sleep(0)
    app._running_runs.add(run_task)

    await app.run()

    assert backend.close_calls == 1
    assert events.index("run.reaped") < events.index("backend.close")
    assert events.index("backend.close") < events.index("trace.stop")


# 功能：backend 构造后的启动异常仍必须执行全部清理，并保留启动异常作为主异常
# 设计：让 MCP 启动与清理都失败；断言原始启动异常传播，backend/trace 不因清理异常而泄漏
async def test_core_app_preserves_startup_error_while_cleanup_still_closes_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config = _config(workspace_root)
    config.mcp.servers = [McpServerConfig(name="test")]
    events: list[str] = []
    backend = _Backend(events, name="host", strongly_isolated=False)
    _install_fakes(
        monkeypatch,
        tmp_path,
        config,
        backend,
        events,
        mcp_start_error=RuntimeError("mcp startup primary"),
        mcp_stop_error=RuntimeError("mcp cleanup secondary"),
    )

    with pytest.raises(RuntimeError, match="mcp startup primary") as caught:
        await app_module.CoreApp().run()

    assert backend.close_calls == 1
    assert events.count("backend.close") == 1
    assert events.index("mcp.stop") < events.index("backend.close")
    assert events.index("backend.close") < events.index("trace.stop")
    assert any("mcp cleanup secondary" in note for note in caught.value.__notes__)


# 功能：某个正常关闭组件失败时仍须关闭 backend 和 trace，并向调用者暴露首个清理错误
# 设计：令 server.stop 失败，检查后续清理顺序和异常传播，防止 finally 静默吞错或提前退出
async def test_core_app_cleanup_error_does_not_skip_backend_or_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config = _config(workspace_root)
    events: list[str] = []
    backend = _Backend(events, name="host", strongly_isolated=False)
    _install_fakes(
        monkeypatch,
        tmp_path,
        config,
        backend,
        events,
        server_stop_error=RuntimeError("server cleanup failed"),
    )

    with pytest.raises(RuntimeError, match="server cleanup failed"):
        await app_module.CoreApp().run()

    assert backend.close_calls == 1
    assert events.index("server.stop") < events.index("backend.close")
    assert events.index("backend.close") < events.index("trace.stop")
