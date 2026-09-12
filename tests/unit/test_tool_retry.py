from __future__ import annotations

from pathlib import Path

import pytest

import kama_claude.core.tools.invocation as inv_mod
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.sandbox import (
    ExecRequest,
    ExecResult,
    SandboxBackend,
    SandboxLimits,
    WorkspaceFS,
)
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.errors import RateLimitedError
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry

# --- stub tools --------------------------------------------------------------

class _FailNTimes(BaseTool):
    """Fails with runtime_error for the first n calls, then succeeds."""
    name = "fail_n"
    description = "Fails n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, n: int, *, error_type: str = "runtime_error") -> None:
        self._remaining = n
        self._error_type = error_type

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        if self._remaining > 0:
            self._remaining -= 1
            return ToolResult(content="transient error", is_error=True, error_type=self._error_type)
        return ToolResult(content="ok")


class _RateLimitedNTimes(BaseTool):
    """Raises RateLimitedError for the first n calls, then succeeds."""
    name = "rate_n"
    description = "Rate-limits n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, n: int) -> None:
        self._remaining = n

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        if self._remaining > 0:
            self._remaining -= 1
            raise RateLimitedError("429 Too Many Requests")
        return ToolResult(content="ok")


class _AlwaysFails(BaseTool):
    name = "always_fail"
    description = "Always fails"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, error_type: str = "runtime_error") -> None:
        self._error_type = error_type

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content="permanent error", is_error=True, error_type=self._error_type)


# --- helper ------------------------------------------------------------------

def _call(name: str, params: dict[str, object] | None = None) -> ToolCallBlock:
    return ToolCallBlock(id="t1", name=name, input=params or {})


async def _run(
    tool: BaseTool,
    *,
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, object] | None = None,
) -> tuple[ToolResult, list]:
    monkeypatch.setattr(inv_mod, "_RETRY_BASE_S", 0.0)
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events: list = []

    async def _collect(e: object) -> None:
        events.append(e)

    bus.subscribe(_collect)
    result = await invoke_tool(registry, _call(tool.name, params), bus, run_id="r")
    return result, events


# --- tests -------------------------------------------------------------------


# 功能：验证 runtime_error 在首次失败后自动重试，最多 2 次，第 2 次成功时返回 ok
# 设计：_FailNTimes(1) 第一次返回 runtime_error，第二次成功；monkeypatch 消除 sleep 延迟
async def test_runtime_error_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_FailNTimes(1), monkeypatch=monkeypatch)
    assert not result.is_error
    assert result.content == "ok"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 1
    assert failed_events[0].attempt == 1  # type: ignore[attr-defined]


# 功能：验证 rate_limited 异常触发重试，重试成功后返回正常结果
# 设计：_RateLimitedNTimes(1) 抛 RateLimitedError 一次，第二次成功；检查 error_class 字段
async def test_rate_limited_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_RateLimitedNTimes(1), monkeypatch=monkeypatch)
    assert not result.is_error
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 1
    assert failed_events[0].error_class == "rate_limited"  # type: ignore[attr-defined]


# 功能：验证 runtime_error 超过 2 次重试后最终返回失败，attempt 字段递增
# 设计：_AlwaysFails 三次都失败；断言最终结果 is_error + 收到 3 个 failed 事件，attempt 为 1/2/3
async def test_runtime_error_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_AlwaysFails("runtime_error"), monkeypatch=monkeypatch)
    assert result.is_error
    assert result.error_type == "runtime_error"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 3
    attempts = [e.attempt for e in failed_events]  # type: ignore[attr-defined]
    assert attempts == [1, 2, 3]


# 功能：验证 rate_limited 耗尽重试后最终返回失败，error_class 为 rate_limited
# 设计：_RateLimitedNTimes(10) 始终抛异常，断言 3 个 failed 事件且 error_class 统一
async def test_rate_limited_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_RateLimitedNTimes(10), monkeypatch=monkeypatch)
    assert result.is_error
    assert result.error_type == "rate_limited"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 3
    assert all(e.error_class == "rate_limited" for e in failed_events)  # type: ignore[attr-defined]


# 功能：验证 schema_error 不触发重试，直接失败
# 设计：_AlwaysFails("schema_error") 首次即 schema 错误，断言只发一次 failed 事件且 attempt=1
async def test_schema_error_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_AlwaysFails("schema_error"), monkeypatch=monkeypatch)
    assert result.is_error
    assert result.error_type == "schema_error"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 1
    assert failed_events[0].attempt == 1  # type: ignore[attr-defined]


# 功能：验证 timeout_error 不触发重试，直接失败
# 设计：SlowTool 配合极短超时触发 TimeoutError；断言只发一次 failed 事件，不重试（重试会再次超时）
async def test_timeout_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    class _SlowTool(BaseTool):
        name = "slow"
        description = "sleeps"
        input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

        async def invoke(self, params: dict[str, object]) -> ToolResult:
            await asyncio.sleep(60)
            return ToolResult(content="done")

    monkeypatch.setattr(inv_mod, "_RETRY_BASE_S", 0.0)
    registry = ToolRegistry()
    registry.register(_SlowTool())
    bus = EventBus()
    events: list = []

    async def _collect(e: object) -> None:
        events.append(e)

    bus.subscribe(_collect)
    result = await invoke_tool(registry, _call("slow"), bus, run_id="r", timeout=0.05)

    assert result.is_error
    assert result.error_type == "timeout"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 1


# 功能：验证成功后的 tool.call_failed 事件中 error_class 字段存在且取值合法
# 设计：_FailNTimes(2) 两次失败后成功，检查所有 failed 事件的 error_class 均在合法枚举内
async def test_failed_event_has_valid_error_class(monkeypatch: pytest.MonkeyPatch) -> None:
    valid_classes = {"runtime_error", "timeout", "schema_error", "permission_denied", "rate_limited"}
    result, events = await _run(_FailNTimes(2), monkeypatch=monkeypatch)
    assert not result.is_error
    for e in events:
        if e.type == "tool.call_failed":  # type: ignore[attr-defined]
            assert e.error_class in valid_classes  # type: ignore[attr-defined]


# 功能：真实 BashTool 的非零命令结果标记为 command_error 后不重试副作用
# 设计：recording backend 统计 execute 次数，通过 invoke_tool 完整走重试决策链
async def test_bash_command_error_executes_backend_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _NonzeroBackend(SandboxBackend):
        name = "fake"
        strongly_isolated = True

        def __init__(self) -> None:
            self.calls = 0

        async def execute(
            self, request: ExecRequest, limits: SandboxLimits
        ) -> ExecResult:
            self.calls += 1
            return ExecResult(9, "failed")

    backend = _NonzeroBackend()
    tool = BashTool(
        WorkspaceFS(tmp_path),
        backend,
        SandboxLimits(30, 1_024, 256, 1.0, 32, 64),
        (),
        tmp_path / ".runtime",
    )

    result, events = await _run(
        tool, monkeypatch=monkeypatch, params={"command": "work"}
    )

    assert result.is_error
    assert result.error_type == "command_error"
    assert backend.calls == 1
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 1


# 功能：真实 BashTool 的瞬时后端异常仍映射 runtime_error 并由调用层重试
# 设计：后端前两次抛异常、第三次成功，断言完整的三次调用计数
async def test_bash_runtime_error_remains_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TransientBackend(SandboxBackend):
        name = "fake"
        strongly_isolated = True

        def __init__(self) -> None:
            self.calls = 0

        async def execute(
            self, request: ExecRequest, limits: SandboxLimits
        ) -> ExecResult:
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("transient backend detail")
            return ExecResult(0, "ok")

    backend = _TransientBackend()
    tool = BashTool(
        WorkspaceFS(tmp_path),
        backend,
        SandboxLimits(30, 1_024, 256, 1.0, 32, 64),
        (),
        tmp_path / ".runtime",
    )

    result, events = await _run(
        tool, monkeypatch=monkeypatch, params={"command": "work"}
    )

    assert not result.is_error
    assert result.content == "ok"
    assert backend.calls == 3
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert [e.attempt for e in failed_events] == [1, 2]  # type: ignore[attr-defined]
