# Lung3D 肺CT三维重建与查看软件

针对肺薄层 CT 的**三维重建**与**3D 查看**工具集，包含两套软件：

1. **分割重建软件**（`lung3d_reconstruct.py`）：TotalSegmentator(`lung_vessels`) 自动分割肺动脉/肺静脉/气管支气管/气道壁，MONAI RetinaNet(LUNA16) 检测肺结节，合并为统一标签体数据 `combined.nii.gz`，并导出每个结构的 STL/OBJ 三维模型与体积报告。
2. **3D 查看器**（`viewer3d.py`）：VTK + PySide6 桌面查看器，支持缩放/旋转/平移、复选框切换显示指定模型、CT 体渲染、窗宽窗位预设、模型透明度与网格精度调节、截图与自动旋转。

此外还附带：

- **Web 后端**（`lung3d_api.py`，FastAPI）：上传已分割的 nii.gz / 病例 zip，提取各结构 GLB 网格供网页/小程序展示，并支持微信小程序码与二维码。
- **网页版查看器**（`web/`）：Three.js 实现的浏览器端 3D 查看器。
- **微信小程序**（`miniprogram/`）：小程序端 3D 查看页面。
- **MONAI 结节模型**（`monai_nodule/`）：LUNA16 预训练 RetinaNet 模型 bundle（Apache-2.0，来自 MONAI）。

---

## 目录结构

```
.
├── lung3d_reconstruct.py     # 软件一：分割重建批处理脚本
├── viewer3d.py               # 软件二：桌面 3D 查看器（VTK+PySide6）
├── lung3d_api.py             # Web 后端（FastAPI）
├── view_scene.py             # 3D Slicer 查看脚本（可选）
├── run_lung3d.bat            # 一键启动重建
├── run_viewer.bat            # 一键启动查看器
├── run_web.bat               # 一键启动 Web 后端
├── view_in_slicer.bat        # 一键在 3D Slicer 中查看
├── requirements.txt          # 完整依赖（含 torch / TotalSegmentator / MONAI）
├── requirements-web.txt      # Web 后端轻量依赖（无 GPU 推理）
├── Dockerfile                # 微信云托管镜像（仅 Web）
├── wx_config.example.json    # 微信小程序码配置模板（真实配置放 wx_config.json，勿提交）
├── monai_nodule/             # MONAI 肺结节检测模型 bundle
│   ├── models/               #   model.pt / model.ts 预训练权重
│   ├── configs/              #   训练/推理/评估配置
│   └── scripts/              #   MONAI bundle 脚本
├── web/                      # 网页版 3D 查看器（Three.js）
├── miniprogram/              # 微信小程序端
├── deploy/                   # 云托管/镜像部署脚本
└── web_output/               # Web 后端运行时病例输出（示例）
```

## 环境要求

- Windows 10/11，Python 3.12
- 推荐 CUDA 12.x GPU（CPU 也可运行，速度较慢）
- 3D 查看器需要显卡支持 OpenGL

## 安装

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch CUDA 版请按注释单独安装（见 `requirements.txt` 顶部说明），例如：

```bat
.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

首次运行 TotalSegmentator 会自动下载模型权重；MONAI 结节权重已含于 `monai_nodule/models/`。

## 使用

### 1. 分割重建（软件一）

支持单个 DICOM 目录、NIfTI 文件，或含多个病例的父目录（自动批量）：

```bat
run_lung3d.bat --input D:\病例\case01 --output .\output
run_lung3d.bat --input D:\全部病例 --output .\output --device cuda --nodule-score 0.3
```

常用选项：

| 参数 | 说明 |
|---|---|
| `--input` | DICOM 目录 / NIfTI 文件 / 父目录 |
| `--output` | 输出根目录（默认 `lung3d_output`） |
| `--device` | `cuda` 或 `cpu` |
| `--fast` | TotalSegmentator 快速模式（省显存） |
| `--no-nodules` | 跳过肺结节检测 |
| `--nodule-score` | 结节检测分数阈值（默认 0.3） |
| `--nodule-model` | MONAI 结节模型目录（默认 `monai_nodule`） |

输出结构（每个病例一个 `case_XXX_名称/` 目录）：

```
case_001_case01/
├── ct.nii.gz                 # 转换后的 CT
├── combined.nii.gz           # 统一标签体数据：1肺动 2肺静 3气管 4气道壁 5结节
├── report.json               # 各结构体积(cm³) + 结节列表
├── seg_totalseg/             # TotalSegmentator 输出（*.nii.gz）
└── mesh/                     # 各结构 STL/OBJ 三维模型
```

### 2. 桌面 3D 查看器（软件二）

```bat
run_viewer.bat                            # 弹出文件对话框
run_viewer.bat D:\output\case_001_case01  # 直接打开病例目录
run_viewer.bat D:\xxx\combined.nii.gz     # 或单个分割文件
```

操作说明：

- 左键旋转 / 中键平移 / 右键或滚轮缩放
- 复选框切换显示：肺动脉 / 肺静脉 / 气管支气管 / 气道壁 / 肺结节 / CT 体渲染
- 点击 3D 对象 = 选中；快捷键 `H` 隐藏选中、`U` 全部显示
- `r` 复位视角 / `s` 截图 / `a` 自动旋转
- 右侧面板可调重建精度、透明度、CT 窗宽窗位预设

### 3. Web 后端 / 网页查看 / 微信小程序

```bat
run_web.bat
```

启动后：

- 网页版：打开 `http://localhost:8000`，上传 nii.gz 或病例 zip 即可在线查看 3D。
- 接口：`POST /api/process`、`GET /api/case/{id}`、`GET /api/mesh/{case_id}/{name}.glb`、`GET /api/cases`。
- 微信小程序码：将真实 `appid/secret` 写入 `wx_config.json`（模板见 `wx_config.example.json`），调用 `GET /api/wxacode?case=<id>`。**请勿把真实密钥提交到 git。**

## 3D Slicer 微调

自动分割结果可在 [3D Slicer](https://slicer.org) 中微调（需安装 TotalSegmentator 扩展）。`view_in_slicer.bat` 提供一键查看脚本（`view_scene.py`），加载 CT 与 STL 模型。

## 常见问题

- **显存不足**：重建时加 `--fast`，或将 `--device cpu`。
- **查看器白屏**：更新显卡驱动，确保 OpenGL 3.2+。
- **Web 上传超时**：大 nii.gz 请先压缩为病例 zip 再上传。

## 版权与许可

- 本工程代码未附加许可声明。
- `monai_nodule/` 为 MONAI 官方 bundle，遵循 Apache-2.0 许可（见其 `LICENSE` 与 `docs/README.md`），模型训练基于 LUNA16/LIDC-IDRI 公开数据集。
- TotalSegmentator 有其自身许可要求（学术用途免费，商用需许可），请自行确认。