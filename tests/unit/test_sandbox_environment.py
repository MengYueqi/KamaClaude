from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from kama_claude.core.sandbox.base import SandboxBackend
from kama_claude.core.sandbox.environment import build_sandbox_env
from kama_claude.core.sandbox.models import ExecRequest, ExecResult, SandboxLimits


# 功能：验证三个沙箱数据模型都是不可变 dataclass
# 设计：执行后限制和请求不能被调用方原地修改，避免绕过既定策略
def test_sandbox_models_are_immutable(tmp_path: Path) -> None:
    limits = SandboxLimits(120, 65_536, 1024, 2.0, 128, 256)
    request = ExecRequest("echo hi", tmp_path, {"PATH": "/bin"})
    result = ExecResult(0, "ok")

    with pytest.raises(FrozenInstanceError):
        limits.timeout_s = 1
    with pytest.raises(FrozenInstanceError):
        request.command = "rm -rf /"
    with pytest.raises(FrozenInstanceError):
        result.output = "changed"


# 功能：验证执行结果的超时和截断标志默认均为 False
# 设计：普通成功结果不应被调用方误判为受限或不完整
def test_exec_result_flags_default_to_false() -> None:
    result = ExecResult(returncode=0, output="done")

    assert result.timed_out is False
    assert result.truncated is False


# 功能：验证沙箱环境只复制白名单且删除 Secret
# 设计：源环境同时包含安全字段和 API Key，断言输出采用正向白名单
def test_build_sandbox_env_uses_allowlist(tmp_path: Path) -> None:
    source = {"PATH": "/bin", "LANG": "C", "ANTHROPIC_API_KEY": "secret"}
    result = build_sandbox_env(
        source,
        frozenset({"PATH", "LANG"}),
        home=tmp_path / "home",
        tmpdir=tmp_path / "tmp",
    )

    assert result["PATH"] == "/bin"
    assert result["LANG"] == "C"
    assert "ANTHROPIC_API_KEY" not in result


# 功能：验证沙箱环境强制覆盖 HOME、TMPDIR 和 CI
# 设计：调用方输入中的运行目录和 CI 标志不能覆盖沙箱固定策略
def test_build_sandbox_env_forces_sandbox_values(tmp_path: Path) -> None:
    home = tmp_path / "home"
    tmpdir = tmp_path / "tmp"
    source = {"HOME": "/host/home", "TMPDIR": "/host/tmp", "CI": "0"}

    result = build_sandbox_env(source, frozenset(source), home=home, tmpdir=tmpdir)

    assert result["HOME"] == str(home)
    assert result["TMPDIR"] == str(tmpdir)
    assert result["CI"] == "1"


# 功能：验证构造沙箱环境时会创建隔离 HOME 和临时目录
# 设计：后端执行前即可安全使用两个目录，即使父目录尚不存在
def test_build_sandbox_env_creates_directories(tmp_path: Path) -> None:
    home = tmp_path / "nested" / "home"
    tmpdir = tmp_path / "nested" / "tmp"

    build_sandbox_env({}, frozenset(), home=home, tmpdir=tmpdir)

    assert home.is_dir()
    assert tmpdir.is_dir()


# 功能：验证沙箱后端接口要求 name、strongly_isolated 和 execute 实现
# 设计：不完整后端必须保持抽象，防止运行时缺少执行能力
def test_sandbox_backend_is_abstract() -> None:
    class IncompleteBackend(SandboxBackend):
        pass

    assert IncompleteBackend.__abstractmethods__ == {"name", "strongly_isolated", "execute"}


# 功能：兼容后端未声明网络策略时默认反映宿主网络可用
# 设计：默认具体属性避免破坏现有第三方 backend，同时提供元数据所需的公开能力查询
def test_sandbox_backend_network_enabled_defaults_true() -> None:
    class FakeBackend(SandboxBackend):
        name = "fake"
        strongly_isolated = False

        async def execute(self, request: ExecRequest, limits: SandboxLimits) -> ExecResult:
            return ExecResult(0, "ok")

    assert FakeBackend().network_enabled is True


# 功能：验证 close 默认是可安全调用的异步 no-op
# 设计：无资源后端不需要覆写 close，Runner 统一清理流程仍可 await
@pytest.mark.asyncio
async def test_sandbox_backend_close_is_async_noop() -> None:
    class FakeBackend(SandboxBackend):
        # 返回测试后端的稳定名称
        @property
        def name(self) -> str:
            return "fake"

        # 表明测试后端不提供强隔离
        @property
        def strongly_isolated(self) -> bool:
            return False

        # 返回最小执行结果以完成后端契约实现
        async def execute(self, request: ExecRequest, limits: SandboxLimits) -> ExecResult:
            return ExecResult(0, request.command)

    backend = FakeBackend()

    assert await backend.close() is None
