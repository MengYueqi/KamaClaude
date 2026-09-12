from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.config import SandboxConfig, get_config


# 将给定内容写入临时 .env 文件
def _write_env(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# 清除会影响沙箱配置解析的系统环境变量
def _clear_sandbox_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "KAMA_CONFIG",
        "KAMA_SANDBOX_BACKEND",
        "KAMA_SANDBOX_WORKSPACE_ROOT",
        "KAMA_SANDBOX_NETWORK",
        "KAMA_SANDBOX_DOCKER_IMAGE",
    ):
        monkeypatch.delenv(name, raising=False)


# 功能：返回使用显式 TOML 路径解析的沙箱配置
# 设计：隔离默认家目录和项目本地配置，确保断言覆盖真实 TOML 读取路径
def _load_sandbox_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toml_content: str
) -> SandboxConfig:
    config_path = tmp_path / "kama.toml"
    config_path.write_text(toml_content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _clear_sandbox_env(monkeypatch)
    monkeypatch.setenv("KAMA_CONFIG", str(config_path))
    return get_config().sandbox


# 功能：验证未设置配置源时产生精确的沙箱默认值
# 设计：从 get_config 的真实默认构造路径读取，避免只断言 dataclass 常量
def test_sandbox_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_sandbox_env(monkeypatch)

    sandbox = get_config().sandbox

    assert sandbox == SandboxConfig()


# 功能：验证合法 [sandbox] TOML 将全部字段解析为规范类型
# 设计：使用真实文件和显式 KAMA_CONFIG，覆盖配置表、列表和 cpu_count 的浮点规范化
def test_sandbox_valid_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _load_sandbox_config(
        tmp_path,
        monkeypatch,
        """[sandbox]
backend = "docker"
workspace_root = "./work"
network = true
docker_image = "registry.example/kama:latest"
timeout_s = 30
output_limit_bytes = 4096
memory_mb = 512
cpu_count = 3
pids_limit = 64
tmpfs_mb = 128
env_allowlist = ["PATH", "CUSTOM"]
""",
    )

    assert sandbox == SandboxConfig(
        backend="docker",
        workspace_root="./work",
        network=True,
        docker_image="registry.example/kama:latest",
        timeout_s=30,
        output_limit_bytes=4096,
        memory_mb=512,
        cpu_count=3.0,
        pids_limit=64,
        tmpfs_mb=128,
        env_allowlist=["PATH", "CUSTOM"],
    )


# 功能：拒绝未知 sandbox 键和所有受限字段的错误类型或非法值
# 设计：通过真实 TOML 解析参数化覆盖严格边界，并专门确认 bool 不能作为数值限制
@pytest.mark.parametrize(
    ("toml_content", "error"),
    [
        ('[sandbox]\nunknown = "value"\n', r"Unknown \[sandbox\] keys"),
        ('[sandbox]\nbackend = "remote"\n', "sandbox.backend"),
        ('[sandbox]\ntimeout_s = 0\n', "sandbox.timeout_s"),
        ('[sandbox]\noutput_limit_bytes = true\n', "sandbox.output_limit_bytes"),
        ('[sandbox]\nmemory_mb = -1\n', "sandbox.memory_mb"),
        ('[sandbox]\ncpu_count = 0\n', "sandbox.cpu_count"),
        ('[sandbox]\npids_limit = 0\n', "sandbox.pids_limit"),
        ('[sandbox]\ntmpfs_mb = 0\n', "sandbox.tmpfs_mb"),
        ('[sandbox]\nnetwork = "false"\n', "sandbox.network"),
        ('[sandbox]\ndocker_image = ""\n', "sandbox.docker_image"),
        ('[sandbox]\ndocker_image = "   "\n', "sandbox.docker_image"),
        ('[sandbox]\nworkspace_root = 1\n', "sandbox.workspace_root"),
        ('[sandbox]\nbackend = ["host"]\n', "sandbox.backend"),
        ('[sandbox]\nbackend = { value = "host" }\n', "sandbox.backend"),
        ('[sandbox]\nenv_allowlist = ["PATH", 1]\n', "sandbox.env_allowlist"),
    ],
)
def test_sandbox_invalid_toml_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toml_content: str, error: str
) -> None:
    with pytest.raises(SystemExit, match=error):
        _load_sandbox_config(tmp_path, monkeypatch, toml_content)


# 功能：验证四个沙箱环境变量覆盖 TOML 与 .env 中的值
# 设计：对每个字段同时提供 TOML、.env 和系统环境变量，确认完整优先级链以系统环境变量结尾
def test_sandbox_system_env_overrides_dotenv_and_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "kama.toml"
    config_path.write_text(
        """[sandbox]
backend = "host"
workspace_root = "toml-workspace"
network = false
docker_image = "toml-image"
""",
        encoding="utf-8",
    )
    _write_env(
        tmp_path / ".env",
        """KAMA_SANDBOX_BACKEND=docker
KAMA_SANDBOX_WORKSPACE_ROOT=dotenv-workspace
KAMA_SANDBOX_NETWORK=true
KAMA_SANDBOX_DOCKER_IMAGE=dotenv-image
""",
    )
    monkeypatch.chdir(tmp_path)
    _clear_sandbox_env(monkeypatch)
    monkeypatch.setenv("KAMA_CONFIG", str(config_path))
    monkeypatch.setenv("KAMA_SANDBOX_BACKEND", "host")
    monkeypatch.setenv("KAMA_SANDBOX_WORKSPACE_ROOT", "system-workspace")
    monkeypatch.setenv("KAMA_SANDBOX_NETWORK", "false")
    monkeypatch.setenv("KAMA_SANDBOX_DOCKER_IMAGE", "system-image")

    sandbox = get_config().sandbox

    assert sandbox.backend == "host"
    assert sandbox.workspace_root == "system-workspace"
    assert sandbox.network is False
    assert sandbox.docker_image == "system-image"


# 功能：拒绝无法明确解释为布尔值的 KAMA_SANDBOX_NETWORK
# 设计：环境变量不允许未知字符串隐式启用网络，防止配置拼写错误扩大权限
def test_sandbox_network_env_rejects_unknown_boolean_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_sandbox_env(monkeypatch)
    monkeypatch.setenv("KAMA_SANDBOX_NETWORK", "sometimes")

    with pytest.raises(SystemExit, match="KAMA_SANDBOX_NETWORK"):
        get_config()


# 功能：拒绝仅含空白的 KAMA_SANDBOX_DOCKER_IMAGE
# 设计：环境变量走与 TOML 相同的最终验证，避免空白镜像名穿过后端安全边界
def test_sandbox_docker_image_env_rejects_whitespace_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_sandbox_env(monkeypatch)
    monkeypatch.setenv("KAMA_SANDBOX_DOCKER_IMAGE", "   ")

    with pytest.raises(SystemExit, match="sandbox.docker_image"):
        get_config()


# 功能：验证 .env 文件中的值被正确加载并覆盖内建默认值
# 设计：写 .env 到临时目录并 chdir 进去，清除同名系统环境变量排除干扰，确认 .env 加载路径有效
def test_dotenv_base_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "KAMA_PORT=9999\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAMA_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 9999


# 功能：验证系统环境变量的优先级高于 .env 文件中的值
# 设计：.env 写 9999，系统环境变量写 8888，确认最终值为 8888，对应四级优先链的顶层约束
def test_system_env_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "KAMA_PORT=9999\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KAMA_PORT", "8888")

    cfg = get_config()

    assert cfg.port == 8888


# 功能：验证 .env 文件不存在时静默跳过，使用内建默认值（不抛异常）
# 设计：chdir 到空目录，清除系统环境变量，确认 get_config() 不因 .env 缺失而崩溃，默认端口为 7437
def test_missing_env_file_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAMA_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 7437


# 功能：验证 .env 中设置的 KAMA_CONFIG 能正确影响 TOML 配置文件的加载路径
# 设计：.env 指向自定义 TOML 文件，TOML 中写入不同端口，确认 .env 在 TOML 加载前被读取（优先级链的正确顺序）
def test_dotenv_before_toml_kama_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_path = tmp_path / "custom.toml"
    toml_path.write_bytes(b'[core]\nport = 5555\n')

    env_file = tmp_path / ".env"
    _write_env(env_file, f"KAMA_CONFIG={toml_path}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAMA_CONFIG", raising=False)
    monkeypatch.delenv("KAMA_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 5555


# 功能：验证同一变量经过完整四级优先链后，最终值为最高优先级来源（系统环境变量）
# 设计：同时设置默认值(7437)/TOML(6000)/.env(7000)/系统环境变量(8000)，确认最终值为 8000，是优先级链的综合正确性验证
def test_priority_chain_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 默认值：7437
    # TOML：6000
    # .env：7000
    # 系统环境变量：8000（最高）
    toml_path = tmp_path / "kama.toml"
    toml_path.write_bytes(b'[core]\nport = 6000\n')

    env_file = tmp_path / ".env"
    _write_env(env_file, "KAMA_PORT=7000\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KAMA_CONFIG", str(toml_path))
    monkeypatch.setenv("KAMA_PORT", "8000")

    cfg = get_config()

    assert cfg.port == 8000
