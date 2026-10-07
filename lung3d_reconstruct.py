#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肺薄层CT三维重建批处理脚本（TotalSegmentator + MONAI 肺结节检测 + VTK 3D导出）

功能：
  1. 读取 DICOM 序列或 NIfTI 文件
  2. TotalSegmentator(lung_vessels) 自动分割: 肺动脉 / 肺静脉 / 气管（气道壁不再输出）
  3. MONAI RetinaNet(LUNA16) 肺结节检测 -> 结节3D边界框
  4. 合并为统一标签体数据 combined.nii.gz
  5. 每个结构分别导出 STL(binary) / OBJ 三维模型
  6. 生成 report.json 体积报告 + 结节列表
  7. 生成 labels.json 结构标注(显示名/配色/角色)，供查看端读取；
     同一份标注同时内嵌进 combined.nii.gz 的 NIfTI 头扩展（文件自描述）

用法：
  单个病例:  python lung3d_reconstruct.py --input DICOM目录或.nii.gz --output 输出目录
  批量病例:  python lung3d_reconstruct.py --input 含多个病例子目录的父目录 --output 输出目录
  常用选项:
    --device cuda / cpu        推理设备
    --fast                     使用 TotalSegmentator 快速模式(低分辨率,省显存)
    --nodule-score 0.3         结节检测分数阈值
    --no-nodules               跳过结节检测
    --nodule-model 目录        MONAI 结节模型目录(默认 lung3d/monai_nodule)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 减少 PyTorch CUDA 显存碎片化（需在 torch 初始化 CUDA 前设置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# nnU-Net 预处理 worker 数：必须在 import nnunetv2 之前设置（其 configuration 模块在导入时读取）
os.environ.setdefault("nnUNet_def_n_proc", "2")
os.environ.setdefault("nnUNet_def_n_proc_export", "2")

DEVICE_DEFAULT = "cuda"


def free_gpu_memory():
    """强制释放 Python/GC 可回收对象并归还 PyTorch 显存缓存，供下一步骤使用。"""
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 工具：DICOM -> NIfTI
# ---------------------------------------------------------------------------
def is_dicom_file(p: Path) -> bool:
    if p.suffix.lower() in (".dcm", ".dicom"):
        return True
    try:
        import pydicom
        with pydicom.dcmread(str(p), stop_before_pixels=True):
            return True
    except Exception:
        return False


def is_nifti_file(p: Path) -> bool:
    return p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))


def find_dicom_cases(root: Path):
    """root 里找病例：root 本身含DICOM -> 返回[root]；否则返回含DICOM或NIfTI的子目录。"""
    if root.is_file():
        return [root] if is_nifti_file(root) else []
    files = list(root.iterdir()) if root.is_dir() else []
    dicom_files = [f for f in files if f.is_file() and is_dicom_file(f)]
    nifti_files = [f for f in files if is_nifti_file(f)]
    if dicom_files or nifti_files:
        return [root]
    cases = []
    if root.is_dir():
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            try:
                sub_files = list(sub.iterdir())
            except OSError:
                continue
            if any(f.is_file() and is_dicom_file(f) for f in sub_files) or any(
                is_nifti_file(f) for f in sub_files
            ):
                cases.append(sub)
    return cases


def _pick_best_series_file(dicom_dir: Path, series_uid: str = ""):
    """挑选用于重建的 DICOM 序列文件列表。

    背景：一个病例目录里常含定位像/多种序列（本次问题：910 个文件分属 4 个
    SeriesInstanceUID）。SimpleITK 的 GetGDCMSeriesFileNames 在含中文路径或
    多序列时可能只返回 1 个文件，导致读出 (X, Y, 1) 全 0 的空体数据，
    后续分割必然失败。

    策略：
      1. 若指定 series_uid，直接取该序列；
      2. 用 pydicom 统计所有序列，按「文件数最多」优先（真 CT 序列切片数远多于定位像），
         同数时取像素非空者；
      3. 同一序列内按 ImagePositionPatient / InstanceNumber 排序，保证层序正确；
      4. 过滤掉没有像素位置信息、像素全 0 的定位像序列。
    """
    files = sorted(p for p in dicom_dir.iterdir() if p.is_file() and is_dicom_file(p))
    if not files:
        return []
    if series_uid:
        selected = []
        for p in files:
            try:
                import pydicom
                if getattr(pydicom.dcmread(str(p), stop_before_pixels=True), "SeriesInstanceUID",
                           None) == series_uid:
                    selected.append(p)
            except Exception:
                pass
        return _sort_series_files(selected)

    groups = {}
    for p in files:
        try:
            import pydicom
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        groups.setdefault(uid, []).append((p, ds))

    def series_rank(item):
        _, ds = item
        ipp = getattr(ds, "ImagePositionPatient", None)
        return (0 if ipp else 1, 0 if int(getattr(ds, "Rows", 0) or 0) > 1 else 1)

    best = None
    for uid, items in groups.items():
        items.sort(key=series_rank)
        # 定位像通常只有 1 个文件；优先选文件数最多的序列
        cand = (len(items), -sum(series_rank(i)[0] for i in items), uid or "")
        if best is None or cand > best[0]:
            best = (cand, items)
    if best is None:
        return []
    return _sort_series_files([p for p, _ in best[1]])


def _sort_series_files(paths):
    """按 InstanceNumber / ImagePositionPatient 的 z 分层排序。"""
    import numpy as np
    import pydicom

    def key(p):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            return (0, 0.0)
        inst = getattr(ds, "InstanceNumber", None)
        inst = float(inst) if inst not in (None, "") else 0.0
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else 0.0
        return (inst, z)

    try:
        return sorted(paths, key=key)
    except Exception:
        return paths


def dicom_to_nifti(dicom_dir: Path, out_nii: Path, series_uid: str = "",
                   out_full: Path = None):
    """读取 DICOM 序列并写出 NIfTI。

    out_full: 若提供，先把原始全分辨率体数据写到该路径（未降采样），
              供肺结节检测使用——小结节在面内 2 倍抽取后会丢失/变形，
              检测必须用原始分辨率；分割与网格仍用降采样后的 out_nii。
              out_full 应位于 ASCII 临时目录（ITK 不支持中文路径写出）。
    """
    import shutil
    import tempfile
    import SimpleITK as sitk

    file_list = _pick_best_series_file(dicom_dir, series_uid)
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    if file_list:
        reader.SetFileNames([str(p) for p in file_list])
        img = reader.Execute()
    else:
        # 退化为直接读取目录（部分未标号数据）
        img = sitk.ReadImage(str(dicom_dir))

    # 自检：空体数据（形状含 1 或像素全 0）说明序列选择失败，直接报错更明确
    import numpy as np
    arr = sitk.GetArrayFromImage(img)          # (z, y, x)
    if arr.ndim != 3 or min(arr.shape) < 8 or not np.any(arr):
        raise ValueError(
            f"DICOM 序列读取异常：得到体数据形状 {arr.shape}（层数过少或像素全 0）。\n"
            f"该目录可能含多个序列/定位像，请在 3D Slicer 中确认正确的 CT 序列，"
            f"或用单序列的 nii.gz 输入。")

    # 前端整数抽取降采样（纯切片、无插值，直接降低后续全部计算量）：
    #   - 面内 1024x1024 / 0.379mm -> 2 取 1 -> 512x512 / 0.758mm
    #   - 层间 0.379mm -> 3 取中间 -> 等效 1.14mm 层厚
    # 用 floor 保证只有明显薄于目标的轴才抽取（1mm 左右规格的数据不受影响）。
    sp = np.array(img.GetSpacing(), dtype=float)            # (x, y, z)
    tgt_xy = float(os.environ.get("CT_TARGET_XY_SPACING_MM", "0.76"))
    tgt_z = float(os.environ.get("CT_TARGET_Z_SPACING_MM", "1.14"))
    targets = (tgt_xy, tgt_xy, tgt_z)
    strides = [max(1, int(np.floor(t / s + 1e-6))) for t, s in zip(targets, sp)]
    if any(s > 1 for s in strides):
        # 抽取前先落盘原始全分辨率副本（结节检测用），此时 img 仍是原始分辨率
        if out_full is not None:
            out_full.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img, str(out_full))
        ox, oy, oz = [s // 2 for s in strides]              # 每组取中间
        arr = arr[oz::strides[2], oy::strides[1], ox::strides[0]]
        if arr.size == 0 or not np.any(arr):
            raise ValueError("DICOM 降采样后体数据为空，请检查切片方向与层厚信息。")
        img2 = sitk.GetImageFromArray(arr)
        img2.SetSpacing(tuple(float(v) for v in (sp * np.array(strides))))
        img2.SetOrigin(img.TransformIndexToPhysicalPoint((ox, oy, oz)))
        img2.SetDirection(img.GetDirection())
        img = img2
        del arr

    # ITK 的 nifti C 库无法写入含非 ASCII(如中文)字符的路径，
    # 因此先写到 ASCII 临时目录，再用 Python(Unicode 安全)复制到目标路径。
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="lung3d_"))
    tmp_file = tmp_dir / "ct.nii.gz"
    try:
        sitk.WriteImage(img, str(tmp_file))
        shutil.copyfile(tmp_file, out_nii)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_nii


def load_nifti(path: Path):
    import nibabel as nib

    return nib.load(str(path))


# ---------------------------------------------------------------------------
# 1) TotalSegmentator lung_vessels
# ---------------------------------------------------------------------------
LUNG_VESSEL_CLASSES = {
    "lung_arteries": 1,
    "lung_veins": 2,
    "lung_airways": 3,
    # 气道壁: 仅保留标签映射供历史数据/pack_data 使用，新重建不再输出(EXCLUDED_OUTPUT_KEYS)
    "lung_airways_wall": 4,
}
NODULE_LABEL = 5

# 气道壁网格表面毛刺多，新重建不再输出该结构（combined/labels.json/mesh 均不含）
EXCLUDED_OUTPUT_KEYS = {"lung_airways_wall"}

# 结构标注表：key -> (显示名, RGB 配色, 处理角色)
# 说明：这些名称属于「数据侧的标注信息」，会随 combined.nii.gz 一并写入输出目录的
# labels.json。查看端（网页 / 小程序 / 移动端）不内置任何结构名称，一律读取该文件，
# 从而让查看端保持为与领域无关的通用三维模型查看器。
# role: vessel=管状结构(按管径剪除远端细支) / wall=管壁类(单独降采样且不平滑)
#       其余取值(airway / lesion 等)仅作标注，处理上按普通结构对待。
# 配色：肺动脉=蓝、肺静脉=深红、气管=绿、肺结节=黄。
#       lung_airways_wall 条目仅用于历史数据标注，新重建不再产出该结构。
STRUCTURE_LABELS = {
    "lung_arteries":     ("肺动脉",     [0.20, 0.40, 1.00], "vessel"),
    "lung_veins":        ("肺静脉",     [0.55, 0.10, 0.10], "vessel"),
    "lung_airways":      ("气管",       [0.20, 1.00, 0.30], "airway"),
    "lung_airways_wall": ("气道壁",     [0.90, 0.60, 0.20], "wall"),
    "lung_nodules":      ("肺结节",     [1.00, 1.00, 0.10], "lesion"),
}


def build_labels_payload(labels):
    """构造 labels.json 同款标注结构：{schema, structures:[{label,key,name,role,color}]}。"""
    items = []
    for label, key in sorted(labels.items()):
        name, color, role = STRUCTURE_LABELS.get(key, (key, None, ""))
        item = {"label": int(label), "key": key, "name": name, "role": role}
        if color:
            item["color"] = list(color)
        items.append(item)
    return {"schema": "lung3d.labels/1", "structures": items}


def write_labels_json(case_dir, labels):
    """把结构标注（显示名 / 配色 / 角色）写入 case_dir/labels.json，供查看端读取。

    labels: {标签值: 结构key}，与 report.json 的 labels 字段一致。
    未在 STRUCTURE_LABELS 中登记的结构，显示名回退为 key、配色由查看端自动分配。
    """
    path = Path(case_dir) / "labels.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_labels_payload(labels), f, ensure_ascii=False, indent=2)
    return path


def save_combined_nifti(combined, affine, path, labels):
    """保存 combined.nii.gz，并把结构标注（与 labels.json 同款 JSON）写入 NIfTI 头扩展。

    这样单个 combined.nii.gz 即自描述：即使 labels.json 丢失，查看端/API
    仍能从文件头读回各标签值对应的显示名/配色/角色。
    """
    import nibabel as nib

    img = nib.Nifti1Image(combined, affine)
    payload = json.dumps(build_labels_payload(labels), ensure_ascii=False).encode("utf-8")
    # 扩展码 4 = NIfTI Comment（通用、各工具容忍度最高）
    img.header.extensions.append(nib.nifti1.Nifti1Extension(4, payload))
    nib.save(img, str(path))
    return Path(path)


def ts_device(device: str) -> str:
    """TotalSegmentator 用 'gpu'/'cpu'，MONAI 用 'cuda'/'cpu'，做统一转换。"""
    return "gpu" if device == "cuda" else "cpu"


def _limit_nnunet_workers(n: int):
    """限制 nnU-Net 预处理后台进程数。

    nnU-Net 默认按 CPU 核数开启多个预处理 worker，每个 worker 会把整个 CT
    体数据载入内存；层数多/体素大的 CT（如 900+ 层）会直接把内存耗尽，
    导致 "Background workers died"。这里通过环境变量强制限制为 n。
    """
    os.environ["nnUNet_def_n_proc"] = str(max(1, int(n)))
    os.environ.setdefault("nnUNet_def_n_proc_export", str(max(1, int(n))))


def run_lung_vessels(nii_path: Path, seg_dir: Path, device: str, fast: bool):
    from totalsegmentator.python_api import totalsegmentator

    seg_dir.mkdir(parents=True, exist_ok=True)
    # 默认把 nnU-Net 后台 worker 限制为 2 个，避免大体积 CT 预处理时爆内存
    _limit_nnunet_workers(int(os.environ.get("LUNG3D_NNUNET_PROC", "2")))

    t0 = time.time()
    nr_thr_resamp = int(os.environ.get("LUNG3D_NR_THR_RESAMP", "1"))
    # 导出阶段每个 worker 都要把整个分割体数据读成 float64(~1GB/512^3 级)，
    # TS 默认 nr_thr_saving=6 会在 16GB 机器上直接耗尽提交内存(MemoryError)。
    nr_thr_saving = int(os.environ.get("LUNG3D_NR_THR_SAVING", "1"))
    totalsegmentator(
        str(nii_path),
        str(seg_dir),
        task="lung_vessels",
        device=ts_device(device),
        fast=fast,
        nora_tag=True,
        quiet=True,
        nr_thr_resamp=nr_thr_resamp,
        nr_thr_saving=nr_thr_saving,
    )
    elapsed = time.time() - t0
    masks = {}
    for name in LUNG_VESSEL_CLASSES:
        p = seg_dir / f"{name}.nii.gz"
        if not p.exists():
            continue
        if name in EXCLUDED_OUTPUT_KEYS:
            # 已排除结构（如气道壁，网格毛刺多）：删除掩膜，避免带入 combined/labels/mesh
            try:
                p.unlink()
            except OSError:
                pass
            continue
        masks[name] = p
    return masks, elapsed


# ---------------------------------------------------------------------------
# 2) MONAI 肺结节检测
# ---------------------------------------------------------------------------
class NoduleDetector:
    def __init__(self, model_dir, device="cuda", score_thresh=0.3):
        import torch
        from monai.apps.detection.networks.retinanet_network import (
            RetinaNet,
            resnet_fpn_feature_extractor,
        )
        from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
        from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
        from monai.networks.nets import resnet

        self.torch = torch
        self.score_thresh = score_thresh
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        backbone = resnet.resnet50(
            spatial_dims=3,
            n_input_channels=1,
            conv1_t_stride=[2, 2, 1],
            conv1_t_size=[7, 7, 7],
        )
        feature_extractor = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[1, 2, 4],
            base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
        )
        self.network = RetinaNet(
            spatial_dims=3,
            num_classes=1,
            num_anchors=3,
            feature_extractor=feature_extractor,
            size_divisible=[16, 16, 8],
            use_list_output=False,
        ).to(self.device)

        ckpt = torch.load(
            os.path.join(model_dir, "models", "model.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.network.load_state_dict(ckpt)
        self.network.eval()

        self.detector = RetinaNetDetector(
            network=self.network,
            anchor_generator=anchor_generator,
            debug=False,
            spatial_dims=3,
            num_classes=1,
            size_divisible=[16, 16, 8],
        )
        self.detector.eval()  # 关键：切换到推理模式，否则会要求 GT 目标
        self.detector.set_target_keys(box_key="box", label_key="label")
        self.detector.set_box_selector_parameters(
            score_thresh=0.02, topk_candidates_per_level=1000, nms_thresh=0.22, detections_per_img=300
        )

    @staticmethod
    def _preprocess(nii_path):
        from monai.transforms import (
            EnsureChannelFirst,
            EnsureType,
            LoadImage,
            Orientation,
            ScaleIntensityRange,
            Spacing,
        )

        img = LoadImage(reader="itkreader")(str(nii_path))  # MetaTensor
        img = EnsureChannelFirst()(img)
        img = Orientation(axcodes="RAS")(img)
        img = Spacing(pixdim=[0.703125, 0.703125, 1.25])(img)
        img = ScaleIntensityRange(a_min=-1024.0, a_max=300.0, b_min=0.0, b_max=1.0, clip=True)(img)
        img = EnsureType()(img)
        return img

    def predict(self, nii_path):
        """返回 (boxes_cccwhd_world_RAS, scores, labels)"""
        from monai.apps.detection.transforms.dictionary import (
            AffineBoxToWorldCoordinated,
            ClipBoxToImaged,
            ConvertBoxModed,
        )
        from monai.transforms import Compose

        img = self._preprocess(nii_path)  # MetaTensor (C,H,W,D)，携带 affine
        batch_img = img.unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            if self.device.type == "cuda":
                with self.torch.autocast("cuda", dtype=self.torch.float16):
                    preds = self.detector([batch_img[0]])
            else:
                preds = self.detector([batch_img[0]])

        pred = preds[0]
        box = pred["box"].to("cpu")
        label = pred["label"].to("cpu")
        score = pred["label_scores"].to("cpu")

        data = {
            "box": box,
            "label": label,
            "label_scores": score,
            "image": img.to("cpu"),  # 保留 MetaTensor 元数据供后处理使用
        } 
        post = Compose(
            [
                ClipBoxToImaged(box_keys="box", label_keys="label", box_ref_image_keys="image", remove_empty=True),
                # itkreader+affine_lps_to_ras=True 时 meta affine 已是 RAS(与nibabel一致)，
                # 因此这里用 affine_lps_to_ras=False，输出的世界坐标即 RAS，可直接用 nibabel affine 转回体素。
                AffineBoxToWorldCoordinated(box_keys="box", box_ref_image_keys="image", affine_lps_to_ras=False),
                ConvertBoxModed(box_keys="box", src_mode="xyzxyz", dst_mode="cccwhd"),
            ]
        )
        out = post(data)
        boxes = out["box"].numpy()
        scores = out["label_scores"].numpy()

        keep = scores >= self.score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        return boxes, scores, np.ones(len(boxes), dtype=np.int64)


def box_to_voxel_mask(boxes_cccwhd, nifti, shape):
    """把世界坐标(RAS,mm)下的 cccwhd 框转换成原图体素空间的椭球掩膜。"""
    aff = nifti.affine
    inv = np.linalg.inv(aff)
    mask = np.zeros(shape, dtype=np.uint8)

    def world_to_voxel(xyz):
        h = np.ones(4)
        h[:3] = xyz
        return inv @ h

    for i, b in enumerate(boxes_cccwhd):
        cx, cy, cz, w, h, d = b
        hw, hh, hd = w / 2.0, h / 2.0, d / 2.0
        corners = [
            world_to_voxel([cx + sx * hw, cy + sy * hh, cz + sz * hd])
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        c = np.array(corners)
        vmin = np.floor(c[:, :3].min(axis=0)).astype(int)
        vmax = np.ceil(c[:, :3].max(axis=0)).astype(int)
        center = world_to_voxel([cx, cy, cz])[:3]

        # 用角点平均步长估算体素半径
        radii = []
        for ax in range(3):
            steps = np.abs(c[:, ax] - center[ax])
            steps = steps[steps > 1e-6]
            radii.append(float(steps.mean()) if len(steps) else 1.0)

        x0 = max(0, int(vmin[0]))
        x1 = min(shape[0] - 1, int(vmax[0]))
        y0 = max(0, int(vmin[1]))
        y1 = min(shape[1] - 1, int(vmax[1]))
        z0 = max(0, int(vmin[2]))
        z1 = min(shape[2] - 1, int(vmax[2]))

        if x0 > x1 or y0 > y1 or z0 > z1:
            continue
        xx, yy, zz = np.mgrid[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1]
        rx, ry, rz = max(radii[0], 1e-3), max(radii[1], 1e-3), max(radii[2], 1e-3)
        inside = (
            ((xx - center[0]) / rx) ** 2
            + ((yy - center[1]) / ry) ** 2
            + ((zz - center[2]) / rz) ** 2
            <= 1.0
        )
        mask[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1][inside] = i + 1
    return mask


def build_lung_field(nifti, tissue_hu=-500.0, air_hu=-400.0):
    """从 CT 粗分割肺野（含肺内实性结构），返回与图像同形状的 bool 掩膜，失败返回 None。

    逐层：组织(tissue_hu 以上)填孔得身体轮廓 -> 轮廓内的低密度区(air_hu 以下)为空气；
    3D 连通域只保留肺大小量级的分量（排除肠气/气泡等），再整体填孔，
    使血管、结节等肺内实性结构包含在肺野内。
    16GB 内存约束：逐层处理，仅保留 air/field 两个布尔体，不做全量 float 转换。
    """
    from scipy import ndimage

    data = np.asanyarray(nifti.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    # 与 viewer3d 相同的启发式：未应用 RescaleIntercept 的原始存储值(0..4095) -> HU
    if data.dtype.kind in ("u", "i") and data.size and int(data.min()) >= 0 and int(data.max()) <= 4095:
        data = data.astype(np.int16) - 1024

    shape = data.shape[:3]
    air = np.zeros(shape, dtype=bool)
    for k in range(shape[2]):
        sl = data[:, :, k]
        body = ndimage.binary_fill_holes(sl > tissue_hu)
        air[:, :, k] = body & (sl < air_hu)
    if not air.any():
        return None

    lab, _ = ndimage.label(air)
    sizes = np.bincount(lab.ravel())
    if sizes.size <= 1:
        return None
    comp_sizes = sizes[1:]                       # 0 是背景
    ids = np.arange(1, sizes.size)
    biggest = int(comp_sizes.max())
    if biggest <= 0:
        return None
    # 只保留 >= 最大分量 10% 的（左右肺通常经气管连成 1~2 个大分量），最多取 4 个
    sel = ids[comp_sizes >= 0.10 * biggest]
    if sel.size > 4:
        sel = sel[np.argsort(comp_sizes[comp_sizes >= 0.10 * biggest])[-4:]]
    del ids, comp_sizes, sizes, air

    field = np.isin(lab, sel)
    field = ndimage.binary_fill_holes(field)     # 肺内实性结构(血管/结节)算入肺野
    return field


def filter_nodules_outside_lung(boxes, scores, nifti, lung_field, margin_mm=None):
    """剔除中心落在肺组织以外的检测框。

    中心在肺野内 -> 保留；中心不在肺野内但在距肺野 margin_mm 以内 -> 保留
    （容许紧贴胸膜/纵隔的结节，减小误删）；超出 -> 删除。
    margin_mm 默认取环境变量 NODULE_LUNG_MARGIN_MM（缺省 8）。
    返回 (boxes, scores, removed_count)。lung_field 为 None 时原样返回。
    """
    if lung_field is None or len(boxes) == 0:
        return boxes, scores, 0
    from scipy import ndimage

    if margin_mm is None:
        margin_mm = float(os.environ.get("NODULE_LUNG_MARGIN_MM", "8"))
    inv = np.linalg.inv(nifti.affine)
    shape = lung_field.shape
    spacing = np.linalg.norm(nifti.affine[:3, :3], axis=0)   # 各轴体素间距(mm)
    spacing = np.where(spacing < 1e-6, 1.0, spacing)
    # 窗口半径：margin 对应的体素数 + 1（窗口内 EDT 在 <=margin 范围内等价于全场距离）
    rad = np.ceil(margin_mm / spacing).astype(int) + 1

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    keep = np.zeros(len(boxes), dtype=bool)
    for i, b in enumerate(boxes):
        c = inv @ np.array([b[0], b[1], b[2], 1.0])
        idx = np.rint(c[:3]).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.array(shape)):
            continue  # 图像外 -> 删除
        if lung_field[tuple(idx)]:
            keep[i] = True
            continue
        lo = np.maximum(idx - rad, 0)
        hi = np.minimum(idx + rad + 1, np.array(shape))
        sub = lung_field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if not sub.any():
            continue  # 窗口内无肺野 -> 删除
        dist = ndimage.distance_transform_edt(~sub, sampling=spacing)
        if float(dist[tuple(idx - lo)]) <= margin_mm:
            keep[i] = True
    removed = int((~keep).sum())
    return boxes[keep], scores[keep], removed


# ---------------------------------------------------------------------------
# 3) 合并 + 体积报告
# ---------------------------------------------------------------------------
def build_combined(mask_paths, shape, affine, nodule_mask=None, nodule_label=5):
    combined = np.zeros(shape, dtype=np.uint8)
    import nibabel as nib

    for name, label in LUNG_VESSEL_CLASSES.items():
        p = mask_paths.get(name)
        if not p:
            continue
        img = nib.load(str(p))
        data = img.get_fdata()
        data = np.asarray(data > 0.5)
        if data.shape != shape:
            data = np.pad(data, [(0, max(0, shape[i] - data.shape[i])) for i in range(3)])
            data = data[: shape[0], : shape[1], : shape[2]]
        combined[data] = label
    if nodule_mask is not None:
        combined[nodule_mask > 0] = nodule_label
    return combined


def export_mesh(mask2d, affine, stl_path, obj_path=None, label_name="", smooth=8,
                ds_factor=2, prune_mm=0.0):
    """用 skimage MarchingCubes + VTK 平滑把二值掩膜转成网格并导出 STL/OBJ。

    - vtkMarchingCubes 对细长管状掩膜会输出碎片网格, 改用 skimage.marching_cubes
      (保证连通性), 再用 vtkWindowedSinc 轻度平滑。
    - 坐标: 体素索引 -> RAS(仿射) -> LPS(翻转 X/Y), 使 Slicer 按 STL 的 LPS 约定正确摆放。
    - prune_mm > 0: 在降采样后按管径阈值(mm)剪除远端细血管，大幅减少三角面数。
    """
    import vtk
    import vtk.util.numpy_support as nps
    from scipy import ndimage
    from skimage import measure

    arr0 = (mask2d > 0).astype(np.uint8)
    if arr0.sum() == 0:
        return False

    # 按物理体素尺寸做整数隔层抽取到 target_spacing(mm) 左右：
    # 0.379mm 薄层每 3 层取中间一层(等效 1.14mm 层厚)，纯切片操作，
    # 无插值、省显存，直接把运算量降一个量级。
    base_spacing = np.linalg.norm(affine[:3, :3], axis=0)
    target_spacing = float(os.environ.get("MESH_TARGET_SPACING_MM", "1.14"))
    stride = np.ones(3, dtype=int)
    if target_spacing > 0:
        for ax in range(3):
            if base_spacing[ax] > 1e-6 and base_spacing[ax] < target_spacing * 0.999:
                s = int(round(target_spacing / base_spacing[ax]))
                if s > 1:
                    stride[ax] = s
    if np.any(stride > 1):
        sl = tuple(slice(s // 2, None, int(s)) for s in stride)
        arr = arr0[sl]
        if arr.sum() == 0:
            arr = arr0
            spacing = base_spacing.copy()
        else:
            spacing = base_spacing * stride
    else:
        arr = arr0
        spacing = base_spacing.copy()
    del arr0

    # ds_factor 大于 1 时(调用方要求的额外降采样倍数)，再整体缩小
    extra = float(max(ds_factor, 1))
    if extra > 1.0:
        # 保留 zoom 前的 arr 作回退（arr0 此时可能已释放；且 base spacing 未必正确）
        arr_pre = arr
        arr = (ndimage.zoom(arr.astype(np.float32), 1.0 / extra, order=1) > 0.5).astype(np.uint8)
        if arr.sum() == 0:
            arr = arr_pre
        else:
            spacing = spacing * extra

    # 细血管剪除：在降采样后的体数据上，按到边界的物理距离(mm)剔除远端细枝，
    # 这是控制血管网格面数最有效的手段。内存不足时跳过剪除。
    if prune_mm > 0 and arr.sum() > 0:
        try:
            dist = ndimage.distance_transform_edt(arr, sampling=spacing)
            pruned = (dist >= prune_mm).astype(np.uint8)
            # 剪除后若几乎全空(阈值过大)，保留原图避免丢失主干
            if pruned.sum() > 0.01 * arr.sum():
                arr = pruned
            del dist
        except MemoryError:
            pass

    origin = (affine @ np.array([0, 0, 0, 1]))[:3]

    verts, faces, _, _ = measure.marching_cubes(arr, level=0.5)
    # skimage 返回顶点列顺序 = 数组轴顺序 (axis0,axis1,axis2) = (i,j,k)，无需交换

    # 体素索引 -> 世界坐标(LPS)
    R = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)[np.newaxis, :]
    flipxy = np.diag([-1.0, -1.0, 1.0])
    A = np.eye(3) * np.nan
    A = flipxy @ R @ np.diag(spacing)
    trans = flipxy @ origin
    world = verts @ A.T + trans

    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(world.astype(np.float64), deep=True))
    pd0 = vtk.vtkPolyData()
    pd0.SetPoints(pts)
    farr = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(len(faces), nps.numpy_to_vtk(farr, deep=True, array_type=vtk.VTK_ID_TYPE))
    pd0.SetPolys(cells)

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(pd0)
    smoother.SetNumberOfIterations(int(smooth))
    smoother.SetPassBand(0.001)
    smoother.SetFeatureAngle(120)
    smoother.BoundarySmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    poly = smoother.GetOutput()

    if stl_path or obj_path:
        import shutil
        import tempfile

        # VTK 写文件同样使用 C 库 fopen，无法写含中文等非 ASCII 的路径，
        # 先写到 ASCII 临时目录再复制到目标。
        tmp_dir = Path(tempfile.mkdtemp(prefix="lung3d_mesh_"))
        try:
            if stl_path:
                tmp_stl = tmp_dir / "out.stl"
                writer = vtk.vtkSTLWriter()
                writer.SetFileName(str(tmp_stl))
                writer.SetFileTypeToBinary()
                writer.SetInputData(poly)
                writer.Write()
                shutil.copyfile(tmp_stl, stl_path)

            if obj_path:
                tmp_obj = tmp_dir / "out.obj"
                ow = vtk.vtkOBJWriter()
                ow.SetFileName(str(tmp_obj))
                ow.SetInputData(poly)
                ow.Write()
                shutil.copyfile(tmp_obj, obj_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process_case(case_input: Path, case_dir: Path, args):
    """处理单个病例；负责创建并在结束后清理临时全分辨率 CT 目录。"""
    import shutil
    import tempfile

    tmp_root = Path(tempfile.mkdtemp(prefix="lung3d_full_"))
    try:
        return _process_case_impl(case_input, case_dir, args, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _process_case_impl(case_input: Path, case_dir: Path, args, tmp_root: Path):
    import nibabel as nib

    case_dir.mkdir(parents=True, exist_ok=True)
    log = []
    log.append(f"[{case_input.name}] 开始处理")

    # 1. 转 NIfTI
    # 结节检测用的原始全分辨率 CT（tempfile: ASCII 路径，ITK 才能写出）。
    # 分割/网格仍用降采样后的 nii_path。
    full_nii = None
    if case_input.is_dir():
        niftis = [f for f in case_input.iterdir() if is_nifti_file(f)]
        has_dicom = any(f.is_file() and is_dicom_file(f) for f in case_input.iterdir())
        if has_dicom:
            nii_path = case_dir / "ct.nii.gz"
            full_nii = tmp_root / "ct_full.nii.gz"
            t0 = time.time()
            dicom_to_nifti(case_input, nii_path, out_full=full_nii)
            # 未发生抽取时没有 full 副本，检测直接用 nii_path
            if not full_nii.exists():
                full_nii = None
            log.append(f"  DICOM->NIfTI 完成 ({time.time()-t0:.1f}s)")
        elif niftis:
            nii_path = niftis[0]
        else:
            raise ValueError(f"无法识别的输入(目录内无DICOM/NIfTI): {case_input}")
    elif is_nifti_file(case_input):
        nii_path = case_input
    else:
        raise ValueError(f"无法识别的输入: {case_input}")

    if not nii_path.exists():
        raise FileNotFoundError(f"找不到图像: {nii_path}")

    # 肺结节检测在原始全分辨率 CT 上进行（降采样会丢小结节）；
    # 检出框是世界坐标(mm)，后续直接投影到降采样网格，天然对齐。
    det_nii = full_nii if full_nii is not None else nii_path

    # 2. TotalSegmentator lung_vessels
    seg_dir = case_dir / "seg_totalseg"
    masks, ts_elapsed = run_lung_vessels(nii_path, seg_dir, args.device, args.fast)
    log.append(f"  TotalSegmentator lung_vessels 完成 ({ts_elapsed:.1f}s)")
    log.append("    输出结构: " + ", ".join(masks.keys()))

    # 释放 TotalSegmentator 占用的显存，避免下一步 MONAI 检测显存不足
    free_gpu_memory()

    # 3. 肺结节检测
    nodule_mask = None
    nodule_report = []
    n_removed_outside = 0
    if not args.no_nodules:
        devices = [args.device]
        if args.device == "cuda":
            devices.append("cpu")  # GPU 显存不足时降级到 CPU
        for dev in devices:
            try:
                detector = NoduleDetector(args.nodule_model, device=dev, score_thresh=args.nodule_score)
                t0 = time.time()
                # 检测输入用原始全分辨率(det_nii)；掩膜落在降采样网格(nii_path)
                boxes, scores, _ = detector.predict(det_nii)
                log.append(f"  MONAI 肺结节检测完成 ({time.time()-t0:.1f}s)，检出 {len(boxes)} 个"
                           + ("" if dev == args.device else f"，(设备: {dev})"))
                nifti = nib.load(str(nii_path))
                if len(boxes):
                    # 删除落在肺组织以外的检测结果（肠气、体壁、床板等误检）
                    if os.environ.get("NODULE_LUNG_FILTER", "1") != "0":
                        try:
                            lung_field = build_lung_field(nifti)
                            boxes, scores, removed = filter_nodules_outside_lung(
                                boxes, scores, nifti, lung_field)
                            n_removed_outside += removed
                            if removed:
                                log.append(f"  肺外检测过滤: 剔除 {removed} 个肺野外的检测结果"
                                           f"（保留 {len(boxes)} 个，容差 {os.environ.get('NODULE_LUNG_MARGIN_MM', '8')}mm）")
                        except MemoryError:
                            log.append("  肺外检测过滤: 内存不足，跳过过滤（保留全部检测）")
                        except Exception as fe:
                            log.append(f"  肺外检测过滤失败（保留全部检测）: {fe}")
                    if len(boxes):
                        nodule_mask = box_to_voxel_mask(boxes, nifti, nifti.shape)
                for i, (b, s) in enumerate(zip(boxes, scores)):
                    nodule_report.append(
                        {
                            "index": i + 1,
                            "center_mm_ras": [round(float(x), 2) for x in b[:3]],
                            "size_mm": [round(float(x), 2) for x in b[3:]],
                            "score": round(float(s), 4),
                        }
                    )
                break
            except Exception as e:
                msg = str(e)
                if dev == args.device and args.device == "cuda" and "out of memory" in msg.lower():
                    log.append(f"  !!! GPU 显存不足: {msg}\n      自动改用 CPU 重试（较慢）…")
                    free_gpu_memory()
                    continue
                log.append(f"  !!! 结节检测失败: {e}")
                break
        free_gpu_memory()

    # 4. 合并标签体数据（结构标注同时写入 NIfTI 头扩展，文件自描述）
    nifti = nib.load(str(nii_path))
    shape = nifti.shape
    affine = nifti.affine
    combined = build_combined(masks, shape, affine, nodule_mask, nodule_label=NODULE_LABEL)
    labels = {
        **{v: k for k, v in LUNG_VESSEL_CLASSES.items()
           if k not in EXCLUDED_OUTPUT_KEYS},
        NODULE_LABEL: "lung_nodules",
    }
    combined_path = case_dir / "combined.nii.gz"
    save_combined_nifti(combined, affine, combined_path, labels)
    log.append(f"  已保存 combined.nii.gz（结构标注已内嵌 NIfTI 头扩展）")

    # 5. 三维模型导出 STL / OBJ
    mesh_dir = case_dir / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    voxel_volume = float(np.abs(np.linalg.det(affine)))
    volumes = {}
    # 网格统一按 1mm 各向同性重采样(MESH_TARGET_SPACING_MM)，并按体积目标自适应
    # 追加细血管剪除与额外降采样，保证每个 STL 可在小程序 WebGL 顺畅加载。
    target_mb = float(os.environ.get("MESH_TARGET_MB", "1.0"))
    base_ds = 1.0
    # 细血管剪除阈值(mm)：远端细血管是三角面数量的主要来源
    prune_mm = float(os.environ.get("DISTAL_PRUNE_MM", "1.0"))

    for label, name in labels.items():
        m = combined == label
        n_vox = int(m.sum())
        volumes[name] = round(float(n_vox * voxel_volume / 1000.0), 3)  # cm^3
        if n_vox == 0:
            continue
        stl = mesh_dir / f"{name}.stl"
        obj = mesh_dir / f"{name}.obj"
        ok = False
        # 血管类先用剪除大幅降面数；仍超标再逐级加大剪除/降采样（最多 5 轮）
        extra_ds = 1.0
        cur_prune = 0.0 if name not in ("lung_arteries", "lung_veins") else prune_mm
        for attempt in range(5):
            kw = {}
            if name in ("lung_arteries", "lung_veins"):
                kw["prune_mm"] = cur_prune
            ok = export_mesh(m, affine, stl, obj, label_name=name,
                             ds_factor=extra_ds, **kw)
            if not ok:
                break
            size_mb = stl.stat().st_size / (1024 * 1024)
            if size_mb <= target_mb:
                break
            # 先加大剪除（对血管最有效），接近上限后再降采样
            if name in ("lung_arteries", "lung_veins") and cur_prune < 4.0:
                cur_prune = max(cur_prune * 1.8, cur_prune + 0.6)
            else:
                extra_ds *= 1.6
            log.append(f"    {name}.stl {size_mb:.1f}MB 超过 {target_mb}MB，"
                       f"剪除={cur_prune:.1f}mm 降采样=x{extra_ds:.1f} 重试")
        if ok:
            log.append(f"  mesh: {stl.name} ({stl.stat().st_size / (1024 * 1024):.2f}MB)")

    # 6. 报告
    report = {
        "case": case_input.name,
        "input": str(case_input),
        "nifti": str(nii_path),
        "labels": labels,
        "volumes_cm3": volumes,
        "nodules": nodule_report,
        "nodules_removed_outside_lung": n_removed_outside,
        "voxel_spacing_mm": [round(float(x), 4) for x in np.linalg.norm(affine[:3, :3], axis=0)],
    }
    with open(case_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    labels_path = write_labels_json(case_dir, labels)
    log.append(f"  已保存 {labels_path.name}（结构标注，供查看端读取）")

    for line in log:
        print(line)
    print(f"[{case_input.name}] 完成 -> {case_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="肺CT三维重建批处理: TotalSegmentator(肺动静脉/气管) + MONAI(肺结节) + STL/OBJ导出"
    )
    parser.add_argument("--input", required=True, help="DICOM目录 / NIfTI 文件 / 含多个病例的父目录")
    parser.add_argument("--output", default="lung3d_output", help="输出根目录")
    parser.add_argument("--device", default=DEVICE_DEFAULT, help="推理设备: cuda 或 cpu")
    parser.add_argument("--fast", action="store_true", help="TotalSegmentator 快速模式(省显存)")
    parser.add_argument("--no-nodules", action="store_true", help="跳过肺结节检测")
    parser.add_argument("--nodule-score", type=float, default=0.3, help="结节检测分数阈值(默认0.3)")
    parser.add_argument(
        "--nodule-model",
        default=str(Path(__file__).resolve().parent / "monai_nodule"),
        help="MONAI 肺结节模型目录",
    )
    args = parser.parse_args()

    root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        sys.exit(f"输入不存在: {root}")

    cases = find_dicom_cases(root)
    if not cases and (root.is_file() and root.name.lower().endswith((".nii", ".nii.gz"))):
        cases = [root]
    if not cases:
        sys.exit("未找到可处理的病例（DICOM目录或NIfTI）。")

    print(f"共 {len(cases)} 个病例，输出到 {out_root}")
    failed = []
    for i, case in enumerate(cases, 1):
        case_dir = out_root / f"case_{i:03d}_{case.name}"
        try:
            process_case(case, case_dir, args)
        except Exception as e:
            import traceback

            print(f"!!! 病例 {case} 处理失败: {e}")
            traceback.print_exc()
            failed.append(str(case))

    if failed:
        print("\n以下病例失败:")
        for f in failed:
            print("  " + f)
        sys.exit(1)
    print("\n全部完成。")


if __name__ == "__main__":
    main()