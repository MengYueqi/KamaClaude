"""Construction of a minimal environment for sandboxed child processes."""

from collections.abc import Collection, Mapping
from pathlib import Path


# 从正向白名单构造环境并固定沙箱目录和 CI 标志
def build_sandbox_env(
    source: Mapping[str, str],
    allowlist: Collection[str],
    home: Path,
    tmpdir: Path,
) -> dict[str, str]:
    """Return an allowlisted environment with isolated HOME and TMPDIR."""
    environment: dict[str, str] = {}
    for name in allowlist:
        if name in source:
            environment[name] = source[name]

    home.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    environment["HOME"] = str(home)
    environment["TMPDIR"] = str(tmpdir)
    environment["CI"] = "1"
    return environment
