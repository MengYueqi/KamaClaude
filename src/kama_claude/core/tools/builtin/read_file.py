from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from kama_claude.core.sandbox import WorkspaceFS
from kama_claude.core.tools.base import BaseTool, ToolResult

_MAX_BYTES = 512 * 1024  # 512 KB


class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str


class ReadFileTool(BaseTool):
    params_model = ReadFileParams
    name = "read_file"
    description = (
        "Read the text content of a file. "
        "Path must be relative to the current working directory. "
        "Files larger than 512 KB are truncated."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file (relative to current working directory).",
            }
        },
        "required": ["path"],
    }

    # 注入固定的 WorkspaceFS 以限制文件读取范围
    def __init__(self, workspace: WorkspaceFS) -> None:
        self._workspace = workspace

    # 读取工作区内普通文件内容；超 512KB 截断
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = ReadFileParams.model_validate(params)
        path = self._workspace.resolve(p.path, must_exist=True)

        if not path.is_file():
            raise IsADirectoryError(f"not a file: {p.path}")

        raw = path.read_bytes()  # raises FileNotFoundError if absent
        truncated = len(raw) > _MAX_BYTES
        text = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
        if truncated:
            text += "\n[truncated]"

        return ToolResult(content=text)
