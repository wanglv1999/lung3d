# Lung3D 安装包构建指南

将分割重建软件打包为可在其他电脑按常规方式安装的 Windows 安装包。

## 产物
- 安装包：`Lung3D-Setup-1.2.0.exe`（约 2.7GB）
- 安装位置：`%LOCALAPPDATA%\Programs\Lung3D`（无需管理员权限）
- 自包含：内置 Python 运行时、TotalSegmentator 权重（离线可用）、MONAI 结节权重
- 支持无 GPU 电脑（自动默认 CPU 模式，速度较慢）

## 构建流程

### 1. 准备打包目录
```powershell
# 复制代码（排除 .git/.venv/缓存）
robocopy . D:\lung3d_build\Lung3D /E /XD ".git" ".venv" "__pycache__" ".cache" /XF "*.pyc"

# 复制 venv（瘦身：去 pycache）
robocopy .venv D:\lung3d_build\Lung3D\.venv /E /XD "__pycache__" /XF "*.pyc" "*.pyo"

# 复制自包含 Python 运行时（python312.dll + Lib 标准库 + DLLs + 基础 python.exe）
robocopy "C:\Users\...\Python312\Lib" D:\lung3d_build\Lung3D\runtime\Lib /E
robocopy "C:\Users\...\Python312\DLLs" D:\lung3d_build\Lung3D\runtime\DLLs /E
Copy-Item python312.dll,python3.dll,vcruntime140.dll,vcruntime140_1.dll,Lib,python.exe,pythonw.exe D:\lung3d_build\Lung3D\runtime\
```

### 2. 复制 TotalSegmentator 权重（离线）
`lung_vessels` 任务需要两个模型，放入 `totalseg_weights/`：
- `Dataset117_lung_airways_arteries_veins_282subj`（主模型）
- `Dataset297_TotalSegmentator_total_3mm_1559subj`（裁剪模型）

来源：`%USERPROFILE%\.totalsegmentator\nnunet\results\<Dataset>`

### 3. 修改打包目录 .venv\pyvenv.cfg
```ini
home = D:\lung3d_build\Lung3D\runtime
include-system-site-packages = false
version = 3.12.10
```
（安装时 `fix_pyvenv.bat` 会把 `home` 改写成实际安装路径）

### 4. 编译安装包
```powershell
"C:\...\Inno Setup 6\ISCC.exe" D:\lung3d_build\Lung3D.iss
```
输出：`D:\lung3d_build\installer\Lung3D-Setup-<版本>.exe`

## 说明
- **离线权重定位**：`lung3d_gui.py` 启动时检测安装目录下的 `totalseg_weights/`，
  存在则设置 `TOTALSEG_WEIGHTS_PATH`/`TOTALSEG_HOME_DIR`（用户级配置在 `%LOCALAPPDATA%\Lung3D\totalseg`）。
- **license**：`python_api` 推理不强制 license；TotalSegmentator 商用需自行注册 license。
- **安装包体积大**（约 2.7GB）主要来自 CUDA 版 PyTorch（4.4GB），无法通过 GitHub 发布，
  需用网盘/U盘分发。