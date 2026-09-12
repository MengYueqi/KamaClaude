from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.sandbox import WorkspaceFS, WorkspaceViolationError
from kama_claude.core.tools.builtin.read_file import ReadFileTool


# 功能：验证读取存在的文件时返回完整内容且 is_error 为 False
# 设计：写临时文件后读取，断言 content 和 is_error，覆盖正常路径（happy path）
async def test_read_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    result = await ReadFileTool(WorkspaceFS(tmp_path)).invoke({"path": "hello.txt"})
    assert not result.is_error
    assert result.content == "hello world"


# 功能：验证文件不存在时抛出 FileNotFoundError 而非返回错误 ToolResult
# 设计：传入不存在的路径，确认 ReadFileTool 不吞掉异常，让调用方（invoke_tool）负责错误分类和事件发布
async def test_file_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await ReadFileTool(WorkspaceFS(tmp_path)).invoke({"path": "missing.txt"})


# 功能：验证 read_file 拒绝绝对路径
# 设计：传入工作区内文件的绝对路径，确保仅能接受相对于注入工作区的路径
async def test_read_file_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceViolationError):
        await ReadFileTool(WorkspaceFS(tmp_path)).invoke({"path": str(tmp_path / "secret.txt")})


# 功能：验证 read_file 拒绝指向工作区外文件的符号链接
# 设计：链接本身位于工作区，但其解析目标在外部，确保解析边界不会被链接绕过
async def test_read_file_rejects_external_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "outside-link.txt").symlink_to(outside)
    with pytest.raises(WorkspaceViolationError):
        await ReadFileTool(WorkspaceFS(tmp_path)).invoke({"path": "outside-link.txt"})


# 功能：验证超过 512KB 的文件被截断并在末尾追加 [truncated] 标记
# 设计：写 600KB 文件，断言内容以 x×512KB 开头、以 [truncated] 结尾，确认截断不破坏前缀内容
async def test_truncation_over_512kb(tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (600 * 1024))
    result = await ReadFileTool(WorkspaceFS(tmp_path)).invoke({"path": "big.txt"})
    assert not result.is_error
    assert result.content.endswith("[truncated]")
    # Actual text content is exactly 512KB worth of 'x' chars
    assert result.content.startswith("x" * (512 * 1024))


# 功能：验证恰好等于 512KB 的文件不被截断（边界值：超过而非大于等于）
# 设计：boundary check，确认截断阈值为"严格超过 512KB"，防止 off-by-one 错误
async def test_exact_512kb_is_not_truncated(tmp_path: Path) -> None:
    f = tmp_path / "exact.txt"
    f.write_bytes(b"y" * (512 * 1024))
    result = await ReadFileTool(WorkspaceFS(tmp_path)).invoke({"path": "exact.txt"})
    assert not result.is_error
    assert not result.content.endswith("[truncated]")
    assert len(result.content) == 512 * 1024


# 功能：验证空文件返回空字符串而非 None 或错误
# 设计：零字节文件确认 content="" 的正常返回，避免调用方（LLM prompt 组装）对空内容做额外 None 判断
async def test_empty_file_returns_empty_content(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    result = await ReadFileTool(WorkspaceFS(tmp_path)).invoke({"path": "empty.txt"})
    assert not result.is_error
    assert result.content == ""
