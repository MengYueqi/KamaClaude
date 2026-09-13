from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

import kama_claude.core.app as app_module
from kama_claude.core.bus.envelope import HandlerError
from kama_claude.core.config import KamaConfig, McpServerConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.sandbox import (
    ExecRequest,
    ExecResult,
    SandboxBackend,
    SandboxLimits,
    WorkspaceFS,
)
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import SpawnAgentTool


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
    task_registry = first_kwargs["task_registry"]
    assert isinstance(task_registry, BackgroundTaskRegistry)
    assert second_kwargs["task_registry"] is task_registry


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


# 功能：直接 session.send_message handler 的完整执行期必须纳入关闭收割范围
# 设计：运行真实 CoreApp handler 并阻塞其 SessionManager await，随后触发真实 run 清理验证跟踪、取消和顺序
async def test_core_app_tracks_and_reaps_blocking_session_send_handler_before_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_event = asyncio.Event
    started = real_event()
    events: list[str] = []

    class _BlockingSessions:
        async def send_message(self, session_id: str, content: str) -> str:
            started.set()
            try:
                await asyncio.Future()
            finally:
                events.append("session-send.reaped")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config = _config(workspace_root)
    backend = _Backend(events, name="host", strongly_isolated=False)
    _install_fakes(monkeypatch, tmp_path, config, backend, events)
    app = app_module.CoreApp()
    app._sessions = _BlockingSessions()  # type: ignore[assignment]
    handler_task = asyncio.create_task(
        app._session_send_handler(
            {"session_id": "sess-test", "content": "run until shutdown"}
        )
    )
    await started.wait()

    assert handler_task in app._running_runs
    await app.run()

    assert handler_task.cancelled()
    assert events.index("session-send.reaped") < events.index("backend.close")
    assert events.index("backend.close") < events.index("trace.stop")


# 功能：CoreApp 共享注册表中的后台 Subagent 必须在 Backend 关闭前取消并收割
# 设计：经真实 BackgroundTaskRegistry.register 注册阻塞任务，同时验证每个 Runner 获得同一注册表实例
async def test_core_app_reaps_shared_background_registry_before_backend_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_event = asyncio.Event
    provider_started = real_event()
    events: list[str] = []

    class _BlockingProvider:
        async def chat(self, *args: object, **kwargs: object) -> object:
            provider_started.set()
            try:
                await asyncio.Future()
            finally:
                events.append("background.reaped")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config = _config(workspace_root)
    backend = _Backend(events, name="host", strongly_isolated=False)
    _install_fakes(monkeypatch, tmp_path, config, backend, events)
    app = app_module.CoreApp()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    spawn_tool = SpawnAgentTool(
        provider=_BlockingProvider(),  # type: ignore[arg-type]
        parent_bus=EventBus(),
        parent_run_id="parent-run",
        permission_manager=None,
        max_steps=1,
        task_registry=app._task_registry,
        runs_dir=tmp_path / "child-runs",
        session_id="sess-test",
        workspace=WorkspaceFS(workspace_root),
        sandbox_backend=backend,
        sandbox_limits=SandboxLimits(10, 1_024, 128, 1.0, 16, 32),
        env_allowlist=("PATH",),
        runtime_dir=runtime_dir,
    )
    result = await spawn_tool.invoke(
        {
            "description": "blocking child",
            "prompt": "wait for shutdown",
            "run_in_background": True,
        }
    )
    run_id = result.content.split("run_id=")[1].split(".")[0]
    entry = app._task_registry.get(run_id)
    assert entry is not None
    background_task, _context = entry
    await provider_started.wait()

    await app.run()

    assert background_task.cancelled()
    assert all(
        call[1]["task_registry"] is app._task_registry for call in _Runner.calls
    )
    assert events.index("background.reaped") < events.index("backend.close")
    assert events.index("backend.close") < events.index("trace.stop")


# 功能：后台任务在取消处理里新注册的嵌套任务也必须在 Backend 关闭前被发现并收割
# 设计：首个注册任务收到取消后同步注册并启动第二个任务；此用例会击穿只读取一次 registry 快照的实现
async def test_core_app_drains_background_tasks_registered_during_quiescing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config = _config(workspace_root)
    events: list[str] = []
    backend = _Backend(events, name="host", strongly_isolated=False)
    _install_fakes(monkeypatch, tmp_path, config, backend, events)
    app = app_module.CoreApp()
    late_tasks: list[asyncio.Task[None]] = []

    async def _late_background() -> None:
        try:
            await asyncio.Future()
        finally:
            events.append("late-background.reaped")

    async def _register_during_cancel() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            late_task = asyncio.create_task(_late_background())
            late_tasks.append(late_task)
            app._task_registry.register(
                "late-background",
                late_task,
                ExecutionContext(
                    run_id="late-background", goal="nested", max_steps=1
                ),
            )
            await asyncio.sleep(0)
            events.append("first-background.reaped")
            raise

    first_task = asyncio.create_task(_register_during_cancel())
    await asyncio.sleep(0)
    app._task_registry.register(
        "first-background",
        first_task,
        ExecutionContext(run_id="first-background", goal="root", max_steps=1),
    )

    try:
        await app.run()
        assert len(late_tasks) == 1
        assert late_tasks[0].cancelled()
        assert events.index("late-background.reaped") < events.index("backend.close")
    finally:
        for task in late_tasks:
            if not task.done():
                task.cancel()
        if late_tasks:
            await asyncio.gather(*late_tasks, return_exceptions=True)


# 功能：关闭门一旦升起，所有可能启动 Runner 的 handler 必须立即拒绝新工作
# 设计：完成一次真实 CoreApp.run 后直接调用两个生产 handler，验证在触及 SessionManager 前返回稳定结构化错误
async def test_core_app_gates_new_agent_and_session_runs_after_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config = _config(workspace_root)
    events: list[str] = []
    backend = _Backend(events, name="host", strongly_isolated=False)
    _install_fakes(monkeypatch, tmp_path, config, backend, events)
    app = app_module.CoreApp()

    await app.run()

    with pytest.raises(HandlerError, match="core shutting down"):
        await app._agent_run_handler({"goal": "too late"})
    with pytest.raises(HandlerError, match="core shutting down"):
        await app._session_send_handler(
            {"session_id": "sess-test", "content": "too late"}
        )


# 功能：backend 构造后的启动异常仍必须执行全部清理，并保留启动异常作为主异常
# 设计：让 MCP 启动与清理都失败；断言原始启动异常传播，backend/trace 不因清理异常而泄漏
async def test_core_app_preserves_startup_error_while_cleanup_still_closes_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
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
        mcp_stop_error=ValueError("SECRET cleanup detail"),
    )
    caplog.set_level(logging.ERROR, logger=app_module.__name__)

    with pytest.raises(RuntimeError, match="mcp startup primary") as caught:
        await app_module.CoreApp().run()

    assert backend.close_calls == 1
    assert events.count("backend.close") == 1
    assert events.index("mcp.stop") < events.index("backend.close")
    assert events.index("backend.close") < events.index("trace.stop")
    assert caught.value.__notes__ == ["mcp cleanup failed (ValueError)"]
    assert "SECRET" not in caplog.text
    assert all("SECRET" not in note for note in caught.value.__notes__)


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


# 功能：清理过程中即使外层连续取消，独立清理任务也必须完成后再传播 CancelledError
# 设计：让被收割运行任务阻塞在取消处理内，连续取消 CoreApp.run 两次后释放并验证 Backend/Trace 顺序
async def test_core_app_repeated_cancellation_cannot_interrupt_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_event = asyncio.Event
    cleanup_blocked = real_event()
    release_cleanup = real_event()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config = _config(workspace_root)
    events: list[str] = []
    backend = _Backend(events, name="host", strongly_isolated=False)
    _install_fakes(monkeypatch, tmp_path, config, backend, events)
    app = app_module.CoreApp()

    async def _slow_to_reap() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_blocked.set()
            await release_cleanup.wait()
            events.append("run.reaped")
            raise

    active_task = asyncio.create_task(_slow_to_reap())
    await asyncio.sleep(0)
    app._running_runs.add(active_task)
    app_task = asyncio.create_task(app.run())
    await cleanup_blocked.wait()

    app_task.cancel()
    await asyncio.sleep(0)
    app_task.cancel()
    await asyncio.sleep(0)
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(app_task, timeout=1)

    assert active_task.cancelled()
    assert events.index("run.reaped") < events.index("backend.close")
    assert events.index("backend.close") < events.index("trace.stop")
