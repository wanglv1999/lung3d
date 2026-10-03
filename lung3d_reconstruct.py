#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肺薄层CT三维重建批处理脚本（TotalSegmentator + MONAI 肺结节检测 + VTK 3D导出）

功能：
  1. 读取 DICOM 序列或 NIfTI 文件
  2. TotalSegmentator(lung_vessels) 自动分割: 肺动脉 / 肺静脉 / 气管支气管 / 气道壁
  3. MONAI RetinaNet(LUNA16) 肺结节检测 -> 结节3D边界框
  4. 合并为统一标签体数据 combined.nii.gz
  5. 每个结构分别导出 STL(binary) / OBJ 三维模型
  6. 生成 report.json 体积报告 + 结节列表
  7. 生成 labels.json 结构标注(显示名/配色/角色)，供查看端读取

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


def dicom_to_nifti(dicom_dir: Path, out_nii: Path, series_uid: str = ""):
    import shutil
    import tempfile
    import SimpleITK as sitk

    if series_uid:
        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(dicom_dir), series_uid))
    else:
        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir)))
    if reader.GetFileNames():
        img = reader.Execute()
    else:
        # 退化为直接读取目录（部分未标号数据）
        img = sitk.ReadImage(str(dicom_dir))

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
    "lung_airways_wall": 4,
}
NODULE_LABEL = 5

# 结构标注表：key -> (显示名, RGB 配色, 处理角色)
# 说明：这些名称属于「数据侧的标注信息」，会随 combined.nii.gz 一并写入输出目录的
# labels.json。查看端（网页 / 小程序 / 移动端）不内置任何结构名称，一律读取该文件，
# 从而让查看端保持为与领域无关的通用三维模型查看器。
# role: vessel=管状结构(按管径剪除远端细支) / wall=管壁类(单独降采样且不平滑)
#       其余取值(airway / lesion 等)仅作标注，处理上按普通结构对待。
STRUCTURE_LABELS = {
    "lung_arteries":     ("肺动脉",     [1.00, 0.20, 0.20], "vessel"),
    "lung_veins":        ("肺静脉",     [0.20, 0.40, 1.00], "vessel"),
    "lung_airways":      ("气管支气管", [0.20, 1.00, 0.30], "airway"),
    "lung_airways_wall": ("气道壁",     [0.90, 0.60, 0.20], "wall"),
    "lung_nodules":      ("肺结节",     [1.00, 1.00, 0.10], "lesion"),
}


def write_labels_json(case_dir, labels):
    """把结构标注（显示名 / 配色 / 角色）写入 case_dir/labels.json，供查看端读取。

    labels: {标签值: 结构key}，与 report.json 的 labels 字段一致。
    未在 STRUCTURE_LABELS 中登记的结构，显示名回退为 key、配色由查看端自动分配。
    """
    items = []
    for label, key in sorted(labels.items()):
        name, color, role = STRUCTURE_LABELS.get(key, (key, None, ""))
        item = {"label": int(label), "key": key, "name": name, "role": role}
        if color:
            item["color"] = list(color)
        items.append(item)
    path = Path(case_dir) / "labels.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema": "lung3d.labels/1", "structures": items}, f,
                  ensure_ascii=False, indent=2)
    return path


def ts_device(device: str) -> str:
    """TotalSegmentator 用 'gpu'/'cpu'，MONAI 用 'cuda'/'cpu'，做统一转换。"""
    return "gpu" if device == "cuda" else "cpu"


def run_lung_vessels(nii_path: Path, seg_dir: Path, device: str, fast: bool):
    from totalsegmentator.python_api import totalsegmentator

    seg_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    totalsegmentator(
        str(nii_path),
        str(seg_dir),
        task="lung_vessels",
        device=ts_device(device),
        fast=fast,
        nora_tag=True,
        quiet=True,
    )
    elapsed = time.time() - t0
    masks = {}
    for name in LUNG_VESSEL_CLASSES:
        p = seg_dir / f"{name}.nii.gz"
        if p.exists():
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


def export_mesh(mask2d, affine, stl_path, obj_path=None, label_name="", smooth=8, ds_factor=2):
    """用 skimage MarchingCubes + VTK 平滑把二值掩膜转成网格并导出 STL/OBJ。

    - vtkMarchingCubes 对细长管状掩膜会输出碎片网格, 改用 skimage.marching_cubes
      (保证连通性), 再用 vtkWindowedSinc 轻度平滑。
    - 坐标: 体素索引 -> RAS(仿射) -> LPS(翻转 X/Y), 使 Slicer 按 STL 的 LPS 约定正确摆放。
    """
    import vtk
    import vtk.util.numpy_support as nps
    from scipy import ndimage
    from skimage import measure

    arr0 = (mask2d > 0).astype(np.uint8)
    if arr0.sum() == 0:
        return False

    if ds_factor > 1:
        scale = 1.0 / ds_factor
        arr = (ndimage.zoom(arr0.astype(np.float32), scale, order=1) > 0.5).astype(np.uint8)
        if arr.sum() == 0:
            arr = arr0
    else:
        arr = arr0

    spacing = np.linalg.norm(affine[:3, :3], axis=0) * ds_factor
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
    import nibabel as nib

    case_dir.mkdir(parents=True, exist_ok=True)
    log = []
    log.append(f"[{case_input.name}] 开始处理")

    # 1. 转 NIfTI
    if case_input.is_dir():
        niftis = [f for f in case_input.iterdir() if is_nifti_file(f)]
        has_dicom = any(f.is_file() and is_dicom_file(f) for f in case_input.iterdir())
        if has_dicom:
            nii_path = case_dir / "ct.nii.gz"
            t0 = time.time()
            dicom_to_nifti(case_input, nii_path)
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
    if not args.no_nodules:
        devices = [args.device]
        if args.device == "cuda":
            devices.append("cpu")  # GPU 显存不足时降级到 CPU
        for dev in devices:
            try:
                detector = NoduleDetector(args.nodule_model, device=dev, score_thresh=args.nodule_score)
                t0 = time.time()
                boxes, scores, _ = detector.predict(nii_path)
                log.append(f"  MONAI 肺结节检测完成 ({time.time()-t0:.1f}s)，检出 {len(boxes)} 个"
                           + ("" if dev == args.device else f"，(设备: {dev})"))
                nifti = nib.load(str(nii_path))
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

    # 4. 合并标签体数据
    nifti = nib.load(str(nii_path))
    shape = nifti.shape
    affine = nifti.affine
    combined = build_combined(masks, shape, affine, nodule_mask, nodule_label=NODULE_LABEL)
    combined_path = case_dir / "combined.nii.gz"
    nib.save(nib.Nifti1Image(combined, affine), combined_path)
    log.append(f"  已保存 combined.nii.gz")

    # 5. 三维模型导出 STL / OBJ
    mesh_dir = case_dir / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    voxel_volume = float(np.abs(np.linalg.det(affine)))
    volumes = {}
    labels = {
        **{v: k for k, v in LUNG_VESSEL_CLASSES.items()},
        NODULE_LABEL: "lung_nodules",
    }
    for label, name in labels.items():
        m = combined == label
        n_vox = int(m.sum())
        volumes[name] = round(float(n_vox * voxel_volume / 1000.0), 3)  # cm^3
        if n_vox == 0:
            continue
        stl = mesh_dir / f"{name}.stl"
        obj = mesh_dir / f"{name}.obj"
        ok = export_mesh(m, affine, stl, obj, label_name=name)
        if ok:
            log.append(f"  mesh: {stl.name}")

    # 6. 报告
    report = {
        "case": case_input.name,
        "input": str(case_input),
        "nifti": str(nii_path),
        "labels": labels,
        "volumes_cm3": volumes,
        "nodules": nodule_report,
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