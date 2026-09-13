# 运维手册（RUNBOOK）

## 日常操作

### 启动守护进程

```bash
uv run kama-core
```

默认监听 `127.0.0.1:7437`，按 `Ctrl+C` 优雅退出。

### 验证连通

```bash
uv run kama ping
# → pong server=0.0.1 uptime=12ms latency=2ms
```

### 停止守护进程

```bash
kill $(pgrep -f kama-core)
```

---

## 配置

优先级（低 → 高）：**内建默认值 → `~/.kama/config.toml` → `.env` → 系统环境变量**。

### `~/.kama/config.toml`

```toml
[core]
host = "127.0.0.1"
port = 7437

[logging]
level  = "INFO"
file   = "~/.kama/logs/core.log"
format = "text"    # "text" | "json"

[sandbox]
backend = "host"                  # "host" | "docker"；默认 host 是弱隔离兼容模式
workspace_root = "."              # daemon 启动时解析并固定
network = false                    # Docker 默认禁网；true 时使用 bridge 网络
docker_image = "kama-sandbox:py312"
timeout_s = 120
output_limit_bytes = 65536
memory_mb = 1024
cpu_count = 2.0
pids_limit = 128
tmpfs_mb = 256
env_allowlist = ["PATH", "LANG", "LC_ALL", "TERM"]
```

### `.env`

从 `.env.example` 复制后修改，存放本机配置与密钥（不提交 git）：

```bash
cp .env.example .env
```

### 系统环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KAMA_CONFIG` | `~/.kama/config.toml` | 覆盖配置文件路径 |
| `KAMA_HOST` | `127.0.0.1` | TCP 监听地址 |
| `KAMA_PORT` | `7437` | TCP 监听端口 |
| `KAMA_LOG_LEVEL` | `INFO` | 日志级别（DEBUG / INFO / WARNING / ERROR） |
| `KAMA_LOG_FILE` | `~/.kama/logs/core.log` | 日志文件路径（留空则仅输出 stderr） |
| `KAMA_LOG_FORMAT` | `text` | 日志格式（`text` 或 `json`） |
| `KAMA_SANDBOX_BACKEND` | `host` | 沙箱后端：`host` 或 `docker` |
| `KAMA_SANDBOX_WORKSPACE_ROOT` | `.` | 启动时固定的 Workspace 根目录 |
| `KAMA_SANDBOX_NETWORK` | `false` | Docker 网络；接受 `true`/`false`、`1`/`0`、`yes`/`no`、`on`/`off` |
| `KAMA_SANDBOX_DOCKER_IMAGE` | `kama-sandbox:py312` | Docker 沙箱镜像名称 |

### Sandbox 行为与迁移

文件工具的 `path` 和 `bash.cwd` 必须是 Workspace 相对路径。绝对路径、越过根目录的 `..` 路径和指向 Workspace 外部的符号链接都会被拒绝；请将已有调用中的绝对路径改为相对路径，例如 `src/main.py`。

`host` 是默认兼容模式，不是强沙箱：它会限制命令环境、cwd、超时、输出和进程组，但无法阻止经批准的命令访问宿主文件或网络。Docker 后端才提供容器边界。两种后端都会从 `env_allowlist` 正向构造命令环境；默认白名单不包含 API Key，命令不会自动继承 API Key。不要把密钥加入该白名单。

Docker 后端只挂载 Workspace 到 `/workspace`，默认使用 `--network none`，并采用只读根文件系统、非 root 用户、capability 移除、禁止提权、PID/CPU/内存和 `/tmp` 限制。Workspace 目前以读写方式挂载：获得批准的命令仍可破坏 Workspace 内部文件。Copy-on-Write 与 Git worktree 变更审批尚未实现。

### 启用 Docker 强隔离

先构建本地测试/运行时镜像：

```bash
docker build --target sandbox-runtime -t kama-sandbox:py312 .
```

然后在配置文件或环境变量中显式选择 Docker：

```toml
[sandbox]
backend = "docker"
workspace_root = "."
network = false
docker_image = "kama-sandbox:py312"
```

```bash
KAMA_SANDBOX_BACKEND=docker KAMA_SANDBOX_NETWORK=false uv run kama-core
```

Docker CLI 缺失、daemon 不可连接或镜像无法启动时，`bash` 会返回 `sandbox_unavailable`（例如 `docker sandbox unavailable (cli-not-found)`、`daemon-unreachable` 或 `startup-failed`）。这是显式失败，系统绝不会静默降级到 `host`。请安装/启动 Docker，并重新运行上面的镜像构建命令；若要使用弱隔离兼容模式，必须由操作者明确设置 `backend = "host"`。

---

## 开发

```bash
uv run ruff check src tests scripts   # lint
uv run mypy src                       # 类型检查
uv run pytest tests/ -v               # 全量测试
uv run pytest tests/unit/ -v         # 仅单元测试（无需启动 daemon）

# Sandbox release gate（Docker 标记测试由下方单独显式启用）
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/unit -v
uv run pytest tests/integration -m "not docker_sandbox" -v
uv run python scripts/gen_protocol_doc.py --check

# Docker release gate（需要本地 Docker daemon）
docker build --target sandbox-runtime -t kama-sandbox:py312 .
KAMA_TEST_DOCKER_SANDBOX=1 uv run pytest tests/integration/test_sandbox_docker.py -v

make docs                             # 重新生成 WIRE_PROTOCOL.md
make verify-s0                        # 完整验证（lint + 类型 + 测试 + 协议同源检查）
```

### Docker 手工安全冒烟检查

用 Docker 模式启动 `kama-core` 和 `kama-tui`，在 TUI 中请求下列命令并记录结果。测试 Workspace 请使用临时目录，避免将仓库根目录作为破坏性检查目标。

```text
pwd                                      → /workspace
env                                      → 不含 API Key
cat ~/.ssh/id_rsa                        → 不可用
python -c 'import socket; ...'           → 网络不可用
echo ok > sandbox-smoke.txt              → 文件只出现在 Workspace 内
sleep 130                                → 超时，容器被删除
```

---

## 日志

```bash
tail -f ~/.kama/logs/core.log
```

---

## 常见错误

| 报错 | 原因 | 处理 |
|------|------|------|
| `core already running at 127.0.0.1:7437` | 已有守护进程在运行 | `kill $(pgrep -f kama-core)` |
| `core not running` | 未启动守护进程 | `uv run kama-core` |
| `Address already in use` | 端口被其他进程占用 | `KAMA_PORT=8000 uv run kama-core` |
| `Config error: KAMA_PORT must be an integer` | `.env` 或环境变量中端口值非整数 | 检查 `KAMA_PORT` 的值 |
| `docker sandbox unavailable (cli-not-found)` | 未安装或 PATH 中找不到 Docker CLI | 安装 Docker CLI 后重启 daemon |
| `docker sandbox unavailable (daemon-unreachable)` | Docker daemon 未运行或当前用户无权限连接 | 启动 Docker daemon，并检查 Docker 访问权限 |
| `docker sandbox unavailable (startup-failed)` | 镜像不存在、Docker 启动失败或 `docker run` 返回 125 | 运行 `docker build --target sandbox-runtime -t kama-sandbox:py312 .`，再检查 Docker 日志 |
