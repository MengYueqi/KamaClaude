from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.sandbox import (
    ExecRequest,
    ExecResult,
    SandboxBackend,
    SandboxLimits,
    SandboxUnavailableError,
    WorkspaceFS,
    WorkspaceViolationError,
)
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.list_dir import ListDirTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool

# ── bash ──────────────────────────────────────────────────────────────────────

class _RecordingBackend(SandboxBackend):
    name = "fake"
    strongly_isolated = True

    def __init__(self, result: ExecResult | BaseException) -> None:
        self.result = result
        self.requests: list[ExecRequest] = []
        self.limits: list[SandboxLimits] = []

    async def execute(self, request: ExecRequest, limits: SandboxLimits) -> ExecResult:
        self.requests.append(request)
        self.limits.append(limits)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _limits(*, timeout_s: int = 90) -> SandboxLimits:
    return SandboxLimits(timeout_s, 12_345, 768, 1.5, 48, 96)


def _bash_tool(
    tmp_path: Path,
    backend: SandboxBackend,
    *,
    timeout_s: int = 90,
    env_allowlist: tuple[str, ...] = ("PATH", "LANG"),
) -> BashTool:
    return BashTool(
        WorkspaceFS(tmp_path),
        backend,
        _limits(timeout_s=timeout_s),
        env_allowlist,
        tmp_path / ".runtime",
    )


# 功能：成功命令仅通过注入后端执行，并传递规范 cwd、白名单环境与完整资源限制
# 设计：recording backend 返回固定结果，直接断言请求边界而不启动任何 shell
@pytest.mark.asyncio
async def test_bash_delegates_success_with_cwd_env_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("PATH", "/allowed/bin")
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    monkeypatch.setenv("KAMA_SECRET", "must-not-leak")
    backend = _RecordingBackend(ExecResult(0, "hello\n"))

    result = await _bash_tool(tmp_path, backend).invoke(
        {"command": "printf hello", "cwd": "project", "timeout": 40}
    )

    assert not result.is_error
    assert result.content == "hello\n"
    assert backend.requests == [
        ExecRequest(
            "printf hello",
            (tmp_path / "project").resolve(),
            {
                "PATH": "/allowed/bin",
                "LANG": "zh_CN.UTF-8",
                "HOME": str(tmp_path / ".runtime" / "home"),
                "TMPDIR": str(tmp_path / ".runtime" / "tmp"),
                "CI": "1",
            },
        )
    ]
    assert "KAMA_SECRET" not in backend.requests[0].env
    assert backend.limits == [SandboxLimits(40, 12_345, 768, 1.5, 48, 96)]
    assert (tmp_path / ".runtime" / "home").is_dir()
    assert (tmp_path / ".runtime" / "tmp").is_dir()


# 功能：工具超时不能放大静态上限，较大调用值必须被配置值压低
# 设计：调用请求 120s 而配置为 25s，同时断言其他限制完全保留
@pytest.mark.asyncio
async def test_bash_caps_timeout_without_changing_other_limits(tmp_path: Path) -> None:
    backend = _RecordingBackend(ExecResult(0, "ok"))

    await _bash_tool(tmp_path, backend, timeout_s=25).invoke(
        {"command": "work", "timeout": 120}
    )

    assert backend.limits == [SandboxLimits(25, 12_345, 768, 1.5, 48, 96)]


# 功能：非零退出码映射为不可重试的 command_error 并保留兼容输出格式
# 设计：后端直接返回退出码与输出，消除真实命令的不确定性
@pytest.mark.asyncio
async def test_bash_nonzero_exit_is_command_error(tmp_path: Path) -> None:
    result = await _bash_tool(
        tmp_path, _RecordingBackend(ExecResult(2, "failed"))
    ).invoke({"command": "fail"})

    assert result.is_error
    assert result.error_type == "command_error"
    assert result.content == "[exit 2]\nfailed"


# 功能：非零命令没有输出时仅保留原有 exit 前缀，不插入成功路径的空输出标记
# 设计：后端返回空输出非零结果，精确回归旧 BashTool 的用户可见文本
@pytest.mark.asyncio
async def test_bash_nonzero_empty_output_preserves_exit_prefix(tmp_path: Path) -> None:
    result = await _bash_tool(
        tmp_path, _RecordingBackend(ExecResult(2, ""))
    ).invoke({"command": "fail silently"})

    assert result.is_error
    assert result.error_type == "command_error"
    assert result.content == "[exit 2]\n"


# 功能：后端超时结果映射为 timeout，并用实际生效上限生成兼容标记
# 设计：后端返回 timed_out=True，断言工具不依赖进程或墙钟超时
@pytest.mark.asyncio
async def test_bash_timeout_result_is_timeout_error(tmp_path: Path) -> None:
    result = await _bash_tool(
        tmp_path,
        _RecordingBackend(ExecResult(137, "partial", timed_out=True)),
        timeout_s=10,
    ).invoke({"command": "slow", "timeout": 30})

    assert result.is_error
    assert result.error_type == "timeout"
    assert result.content == "[timeout after 10s]"


# 功能：空输出与被截断输出保留现有工具的用户可见标记
# 设计：参数化标准化后端结果，精确断言 content 格式
@pytest.mark.parametrize(
    ("exec_result", "expected"),
    [
        (ExecResult(0, ""), "[no output]"),
        (ExecResult(0, "prefix", truncated=True), "prefix\n[truncated]"),
    ],
)
@pytest.mark.asyncio
async def test_bash_formats_empty_and_truncated_output(
    tmp_path: Path, exec_result: ExecResult, expected: str
) -> None:
    result = await _bash_tool(tmp_path, _RecordingBackend(exec_result)).invoke(
        {"command": "work"}
    )

    assert not result.is_error
    assert result.content == expected


# 功能：Bash cwd 不能越出 workspace，且必须是目录而非普通文件
# 设计：分别传入父目录与工作区内文件，两者都应在调用后端前被硬边界拒绝
@pytest.mark.parametrize("cwd", ["../outside", "a-file"])
@pytest.mark.asyncio
async def test_bash_rejects_cwd_outside_workspace_or_not_directory(
    tmp_path: Path, cwd: str
) -> None:
    (tmp_path / "a-file").write_text("not a directory", encoding="utf-8")
    backend = _RecordingBackend(ExecResult(0, "must not run"))

    result = await _bash_tool(tmp_path, backend).invoke({"command": "work", "cwd": cwd})

    assert result.is_error
    assert result.error_type == "sandbox_violation"
    assert backend.requests == []


# 功能：结构化后端不可用错误映射为 sandbox_unavailable 且不泄露底层异常秘密
# 设计：以包含 Secret 的 cause 链接稳定错误，工具只应返回结构化公开诊断
@pytest.mark.asyncio
async def test_bash_maps_sandbox_unavailable_without_leaking_cause(tmp_path: Path) -> None:
    try:
        raise SandboxUnavailableError("docker", "daemon-unreachable") from RuntimeError(
            "API_KEY=super-secret"
        )
    except SandboxUnavailableError as unavailable:
        backend = _RecordingBackend(unavailable)

    result = await _bash_tool(tmp_path, backend).invoke({"command": "work"})

    assert result.is_error
    assert result.error_type == "sandbox_unavailable"
    assert result.content == "docker sandbox unavailable (daemon-unreachable)"
    assert "super-secret" not in result.content


# 功能：未预期后端异常保持 runtime_error，但不向 ToolResult 复制可能含 Secret 的详细信息
# 设计：异常文本故意包含秘密，断言用户可见内容是稳定的脱敏消息
@pytest.mark.asyncio
async def test_bash_sanitizes_unexpected_backend_exception(tmp_path: Path) -> None:
    result = await _bash_tool(
        tmp_path, _RecordingBackend(RuntimeError("API_KEY=super-secret"))
    ).invoke({"command": "work"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert result.content == "sandbox execution failed"
    assert "super-secret" not in result.content


# ── write_file ────────────────────────────────────────────────────────────────

# 功能：验证 write_file 写入文件后内容可以被读取，返回字节数
# 设计：写入临时目录，断言文件存在且内容一致；用 tmp_path fixture 自动清理
@pytest.mark.asyncio
async def test_write_file_creates_and_returns_size(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    result = await WriteFileTool(WorkspaceFS(tmp_path)).invoke(
        {"path": "out.txt", "content": "hello world"}
    )
    assert not result.is_error
    assert "11" in result.content  # "hello world" = 11 bytes
    assert target.read_text() == "hello world"


# 功能：验证 write_file 自动创建不存在的父目录
# 设计：路径包含两层不存在的子目录，确认写入后目录结构被创建
@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.txt"
    result = await WriteFileTool(WorkspaceFS(tmp_path)).invoke(
        {"path": "a/b/file.txt", "content": "x"}
    )
    assert not result.is_error
    assert target.exists()


# 功能：验证 write_file 拒绝绝对路径
# 设计：传入工作区内目标的绝对路径，确保写入仅接受相对于注入工作区的路径
@pytest.mark.asyncio
async def test_write_file_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceViolationError):
        await WriteFileTool(WorkspaceFS(tmp_path)).invoke(
            {"path": str(tmp_path / "secret.txt"), "content": "x"}
        )


# 功能：验证 write_file 拒绝指向工作区外目录的符号链接
# 设计：符号链接位于工作区内但目标目录在外部，写入时必须由 WorkspaceFS 拒绝
@pytest.mark.asyncio
async def test_write_file_rejects_external_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "outside-link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceViolationError):
        await WriteFileTool(WorkspaceFS(tmp_path)).invoke(
            {"path": "outside-link/secret.txt", "content": "x"}
        )


# ── list_dir ──────────────────────────────────────────────────────────────────

# 功能：验证 list_dir 输出包含目录中的文件名
# 设计：在 tmp_path 创建已知结构，断言文件名出现在 content 中；不约束格式细节
@pytest.mark.asyncio
async def test_list_dir_shows_files(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x")
    (tmp_path / "bar.md").write_text("y")
    result = await ListDirTool(WorkspaceFS(tmp_path)).invoke({"path": "."})
    assert not result.is_error
    assert "foo.py" in result.content
    assert "bar.md" in result.content


# 功能：验证 list_dir 按 max_depth 限制递归深度（depth=1 时不展示孙级目录内容）
# 设计：创建 parent/child/grandchild 三层，depth=1 时 grandchild 不应出现在输出中
@pytest.mark.asyncio
async def test_list_dir_respects_max_depth(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    grandchild = child / "grandchild"
    grandchild.mkdir()
    (grandchild / "deep.txt").write_text("x")

    result = await ListDirTool(WorkspaceFS(tmp_path)).invoke({"path": ".", "max_depth": 1})
    assert not result.is_error
    assert "child" in result.content
    assert "deep.txt" not in result.content


# 功能：验证对不存在的路径 list_dir 抛出 FileNotFoundError
# 设计：直接传入不存在的路径字符串，预期抛出标准异常（invocation.py 捕获后返回 error ToolResult）
@pytest.mark.asyncio
async def test_list_dir_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await ListDirTool(WorkspaceFS(tmp_path)).invoke({"path": "missing"})


# 功能：验证 list_dir 拒绝绝对路径
# 设计：传入工作区根目录的绝对路径，确保枚举仅接受相对于注入工作区的路径
@pytest.mark.asyncio
async def test_list_dir_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceViolationError):
        await ListDirTool(WorkspaceFS(tmp_path)).invoke({"path": str(tmp_path)})


# 功能：验证 list_dir 拒绝指向工作区外目录的符号链接
# 设计：目录链接自身在工作区内但解析到外部目录，不能借此枚举外部内容
@pytest.mark.asyncio
async def test_list_dir_rejects_external_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "outside-link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceViolationError):
        await ListDirTool(WorkspaceFS(tmp_path)).invoke({"path": "outside-link"})


# 功能：验证 list_dir 显示工作区内目录符号链接但不递归进入
# 设计：链接目标与链接并列于工作区内，断言别名可见而别名路径下不列出目标文件
@pytest.mark.asyncio
async def test_list_dir_does_not_recurse_into_internal_directory_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "nested.txt").write_text("x", encoding="utf-8")
    (tmp_path / "alias").symlink_to(target, target_is_directory=True)

    result = await ListDirTool(WorkspaceFS(tmp_path)).invoke({"path": ".", "max_depth": 2})

    assert not result.is_error
    assert "alias/" in result.content
    assert "alias/\n│   └── nested.txt" not in result.content
