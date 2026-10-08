在 Pixi 环境中，依赖管理主要通过 `pixi.toml` 文件进行。与传统的 `conda` 或 `pip` 直接修改全局环境不同，**Pixi 的核心思想是声明式管理**：你通过命令添加依赖，Pixi 会自动更新 `pixi.toml` 文件，并根据该文件在本地创建隔离的 `.pixi` 环境目录。

以下是在 Pixi 中安装和管理依赖的详细方法：

### 1. 安装基本依赖 (来自 conda-forge)
默认情况下，Pixi 会从 `conda-forge` 频道安装包。使用 `pixi add` 命令：

```bash
# 安装单个包
pixi add numpy

# 安装多个包
pixi add pandas matplotlib

# 指定版本号
pixi add "python=3.11"
pixi add "numpy>=1.20"
```
*执行后，Pixi 会自动解析依赖、更新 `pixi.toml` 中的 `[dependencies]` 部分，并自动安装到本地环境。*

### 2. 安装 PyPI 依赖 (使用 pip/PyPI 源)
有些包在 conda-forge 上没有，或者你需要使用 PyPI 上的版本。可以使用 `--pypi` 标志：

```bash
# 安装 PyPI 包
pixi add --pypi requests

# 指定 PyPI 包的版本
pixi add --pypi "tensorflow==2.15.0"
```
*这会将包添加到 `pixi.toml` 的 `[pypi-dependencies]` 部分中。*

### 3. 为特定平台安装依赖
Pixi 的一个强大特性是支持跨平台开发。你可以为不同的操作系统指定不同的依赖：

```bash
# 仅为 Linux 安装 gcc
pixi add --platform linux-64 gcc

# 仅为 Windows 安装 pywin32
pixi add --platform win-64 pywin32

# 仅为 macOS (Apple Silicon) 安装某个特定的包
pixi add --platform osx-arm64 some-mac-specific-lib
```
*这会在 `pixi.toml` 中生成 `[target.linux-64.dependencies]` 等特定的字段。*

### 4. 指定特定 Channel (频道) 安装
如果包不在 conda-forge 上，而在其他频道（如 bioconda、pytorch），你可以指定频道：

```bash
# 从 pytorch 频道安装 pytorch
pixi add --channel pytorch pytorch

# 从 bioconda 频道安装 samtools
pixi add --channel bioconda samtools
```
*这会在 `pixi.toml` 中生成带有 `{ channel = "pytorch" }` 标注的依赖项。*

### 5. 从现有文件同步环境
如果你克隆了一个包含 `pixi.toml` 的项目，或者手动修改了 `pixi.toml`，你需要运行以下命令来安装所有依赖：

```bash
pixi install
```
*此命令会读取 `pixi.toml`，解析 `pixi.lock`（如果存在）或重新解析依赖，并在 `.pixi` 目录下构建环境。*

### 6. 依赖管理其他常用操作

*   **查看已安装依赖：**
    ```bash
    pixi list
    ```
*   **移除依赖：**
    ```bash
    pixi remove numpy
    pixi remove --pypi requests
    ```
*   **更新依赖：**
    ```bash
    # 更新所有依赖到符合限制的最新版本
    pixi update

    # 仅更新特定包
    pixi update numpy
    ```
*   **升级 Pixi 项目（无视版本限制升级）：**
    如果你需要突破 `pixi.toml` 中的版本限制强行升级所有包到最新版：
    ```bash
    pixi upgrade
    ```

### 7. 在环境中运行命令
安装好依赖后，不需要手动 `activate` 环境（虽然你可以用 `pixi shell`），直接使用 `pixi run` 即可在隔离环境中执行命令：

```bash
# 运行 Python 脚本
pixi run python main.py

# 启动 Jupyter Lab
pixi run jupyter lab

# 激活环境进入交互式终端
pixi shell
```

### 总结与最佳实践
1. **始终使用 `pixi add` 而不是手动编辑 `pixi.toml` 来添加复杂依赖**，因为 Pixi 会立即帮你检查依赖冲突。
2. **将 `pixi.toml` 和 `pixi.lock` 提交到 Git**，这样其他开发者拉取代码后，只需运行 `pixi install` 就能得到与你完全一致的环境。
3. **不要将 `.pixi` 目录提交到 Git**（在 `.gitignore` 中添加 `.pixi/`）。