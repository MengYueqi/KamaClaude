from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, ConfigDict

from kama_claude.core.sandbox import WorkspaceFS
from kama_claude.core.tools.base import BaseTool, ToolResult

_MAX_BYTES = 1 * 1024 * 1024  # 1 MB


class WriteFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    content: str


class WriteFileTool(BaseTool):
    params_model = WriteFileParams
    name = "write_file"
    description = (
        "Write text content to a file, creating it (and any parent directories) if it "
        "does not exist, or overwriting it if it does. "
        "Path must be relative to the current working directory. "
        "Content size is limited to 1 MB."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file (relative to current working directory).",
            },
            "content": {
                "type": "string",
                "description": "Text content to write.",
            },
        },
        "required": ["path", "content"],
    }

    # 注入固定的 WorkspaceFS 以限制文件写入范围
    def __init__(self, workspace: WorkspaceFS) -> None:
        self._workspace = workspace

    # 原子写入工作区内文件；超 1MB 拒绝并自动创建父目录
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = WriteFileParams.model_validate(params)
        content = p.content
        path = self._workspace.resolve(p.path, must_exist=False)

        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_BYTES:
            return ToolResult(
                content=f"content too large: {len(encoded)} bytes (limit 1 MB)",
                is_error=True,
                error_type="runtime_error",
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path = self._workspace.resolve(p.path, must_exist=False)
        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(delete=False, dir=path.parent) as temporary_file:
                temp_path = Path(temporary_file.name)
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        return ToolResult(content=f"wrote {len(encoded)} bytes to {p.path}")
