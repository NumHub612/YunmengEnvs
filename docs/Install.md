# Install

项目依赖 Python 3.12，使用 Pixi 作为环境管理工具，下载并配置所有依赖（包括 Python 3.12、NumPy、SciPy、VTK、PyTorch 等）。

---

## windows 平台

基于 Windows 10/11 系统，使用 Pixi 作为环境管理工具，Python 3.12 作为运行环境。

### 1.1 清理旧环境（如使用过 Anaconda/Miniconda）

如果之前安装过 Anaconda/Miniconda，建议先禁用自动激活，避免干扰：

```powershell
conda config --set auto_activate_base false
```

### 1.2 指定安装路径（推荐安装到 D 盘）

以**管理员身份**打开 PowerShell，执行以下命令：

```powershell
# 设置 Pixi 主目录到 D 盘（避免 C 盘空间不足）
$env:PIXI_HOME = "D:\pixi"

# 执行安装
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

> **注意**：如果执行过程中出现 `Add-Type` 编译错误（与系统中残留的 Visual Studio 或 Anaconda 环境变量有关），通常不影响 Pixi 下载。请手动检查 `D:\pixi\bin` 目录下是否存在 `pixi.exe`。

### 1.3 手动添加 PATH（如自动添加失败）

```powershell
# 临时生效（当前会话）
$env:Path = "D:\pixi\bin;" + $env:Path

# 永久生效（推荐）
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";D:\pixi\bin", "User")
```

**关闭当前 PowerShell，重新打开新窗口**，验证安装：

```powershell
pixi --version
# 预期输出：pixi 0.41.0 或更高版本
```

### 1.4 配置全局环境目录

```powershell
pixi config set detached-environments "D:\pixi-envs" --global
```

此后所有项目的虚拟环境将统一存放在 `D:\pixi-envs` 下，而非项目目录内的 `.pixi/envs/`（避免项目目录臃肿）。

### 1.5 安装项目依赖

在项目根目录下，执行以下命令：

```powershell
pixi install
```

Pixi 会自动读取 `pixi.toml` 和 `pixi.lock`，下载并配置所有依赖（包括 Python 3.12、NumPy、SciPy、VTK、PyTorch 等）。

> **首次安装耗时较长**，因为需要下载大量科学计算二进制包。后续通过 `pixi.lock` 可实现秒级还原。

### 1.6 验证环境

```powershell
pixi run python --version
# 预期输出：Python 3.12.x

pixi run test
# 自动执行项目测试，并输出coverage报告

pixi shell -e new_env_name
# 临时切换环境并进入交互式 Shell
```

## linux 平台

Ubuntu 22.04/24.04 系统，使用 Pixi 作为环境管理工具，Python 3.12 作为运行环境。也可用于 CI/CD 环境（GitHub Actions、Docker 等）。

### 1.1 安装基础工具

```bash
sudo apt update
sudo apt install -y git curl wget build-essential
```

### 1.2 安装 headless 显示依赖（PyVista/VTK 需要）

```bash
sudo apt install -y libgl1-mesa-glx libgl1-mesa-dev xvfb
```

> 云梦项目依赖 PyVista 进行 3D 可视化，在 Linux 无图形界面环境下需要 Mesa 和 Xvfb 支持。

### 1.3 安装脚本

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

安装脚本会自动：

- 下载最新版 Pixi 到 `~/.pixi/bin`
- 添加 `~/.pixi/bin` 到 `~/.bashrc` 的 PATH 中

### 1.4 激活环境变量

```bash
source ~/.bashrc

pixi --version
# 预期输出：pixi 0.41.0 或更高版本
```

### 1.5 安装项目依赖

在项目根目录下，执行以下命令：

```bash
pixi install
```

Pixi 会自动读取 `pixi.toml` 和 `pixi.lock`，从 conda-forge 下载并配置所有依赖。

> **首次安装耗时较长**，因为需要下载 NumPy、SciPy、VTK、PyTorch 等科学计算二进制包。后续通过 `pixi.lock` 可实现秒级还原。

### 1.6 验证环境

```powershell
pixi run python --version
# 预期输出：Python 3.12.x

pixi run test
# 自动执行项目测试，并输出coverage报告
```

## Pixi 包管理

在 Pixi 中增加依赖的核心命令是 `pixi add`，`pixi.toml` 和 `pixi.lock` 都会**自动同步更新**，无需手动编辑。

### 3.1 添加 conda-forge 包（默认）

```bash
# 添加最新版
pixi add numpy

# 添加指定版本
pixi add numpy=1.26.4

# 添加版本范围
pixi add "numpy>=1.26,<2.0"

# 添加时自动更新 lockfile
pixi add scipy pandas matplotlib
```

执行后，Pixi 会：

- 自动修改 `pixi.toml` 的 `[dependencies]` 段落
- 自动更新 `pixi.lock`
- 自动下载并安装到当前环境

### 3.2 添加 PyPI 独有包（conda-forge 没有或版本不匹配）

```bash
pixi add --pypi torch
pixi add --pypi "torch>=2.6.0"
pixi add --pypi gunicorn
```

> `--pypi` 标志告诉 Pixi 从 PyPI 而非 conda-forge 获取。Pixi 内部使用 uv 引擎解析，速度和原生 pip 一样快。

### 3.3 添加到特定环境（feature）

你的 `pixi.toml` 定义了 `dev`、`docs`、`test` 等 feature。添加依赖到特定 feature：

```bash
# 添加到 dev 环境（开发工具）
pixi add --feature dev pytest
pixi add --feature dev pytest-cov
pixi add --feature dev ruff

# 添加到 docs 环境（文档构建）
pixi add --feature docs mkdocs
pixi add --feature docs mkdocstrings-python

# 添加到 test 环境（测试专用）
pixi add --feature test pytest-xdist
```

执行后 `pixi.toml` 会自动更新：

```toml
[feature.dev.dependencies]
pytest = ">=9.0.2"
pytest-cov = ">=6.0.0"
ruff = ">=0.9.0"

[feature.docs.dependencies]
mkdocs = ">=1.6.0"
```

### 3.4 更新 lockfile 的时机

`pixi.lock` 是确定性构建的核心，以下操作会自动更新它：

| 操作               | 是否更新 lockfile | 说明                         |
| ------------------ | ----------------- | ---------------------------- |
| `pixi add <pkg>`   | ✅ 自动更新       | 添加新包时                   |
| `pixi update`      | ✅ 强制更新       | 手动升级所有依赖到最新兼容版 |
| `pixi install`     | ❌ 不更新         | 只按现有 lockfile 安装       |
| 手动改 `pixi.toml` | ❌ 不自动         | 必须再执行 `pixi update`     |

**最佳实践**：

- 日常开发：`pixi add` 即可，lockfile 自动同步
- 定期维护：`pixi update` 升级所有依赖到最新兼容版，然后跑一遍测试确认稳定
- 提交代码时：**同时提交 `pixi.toml` 和 `pixi.lock`**，确保团队成员和 CI 使用完全一致的依赖

### 3.5 移除依赖

```bash
pixi remove numpy
pixi remove --feature dev pytest
```

同样会自动更新 `pixi.toml` 和 `pixi.lock`。

### 3.6 查看当前依赖

```bash
pixi list                    # 列出已安装的所有包
pixi list --json             # JSON 格式输出
pixi tree                    # 树形展示依赖关系
pixi info                    # 查看项目环境信息
```
