#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lung3D 数据目录打包工具

把一个「已分割好的多标签体数据」整理成查看端可直接上传的数据目录：

    <out>/<name>/
        combined.nii.gz   多标签分割体数据（标签 1..N）
        labels.json       结构标注（显示名 / 配色 / 处理角色）
        report.json       体积报告（含数据名）
    可选再压成 <out>/<name>.zip —— 直接在后端 /api/process 上传即可。

为什么要这个工具：
  查看端（网页 / 小程序）不内置任何结构名称，名称一律来自数据目录里的
  labels.json。凡是要上传的数据，都应当带上这个文件，否则结构只会显示为
  「结构 1 / 结构 2 …」。本工具用于把历史产物或裸体数据补齐成规范目录。

两种输入：
  1. 单个多标签 nii.gz（如 combined.nii.gz）—— 按标签序号套用内置结构表
  2. 一个已有的输出目录            —— 读其中的 combined.nii.gz 并复用已有 labels.json

用法：
  单份数据，补标注并打包：
    python lung3d_pack_data.py --input combined.nii.gz --name 示例数据 --zip

  已有输出目录，原样整理（沿用其中的 labels.json）：
    python lung3d_pack_data.py --input ./lung3d_output/case1 --name 示例数据 --zip

  批量：目录下每个 *.nii.gz 各自打成一个数据包：
    python lung3d_pack_data.py --input ./nii_dir --batch --zip

  不带结构表（纯中性命名「结构 N」）：
    python lung3d_pack_data.py --input combined.nii.gz --name 示例数据 --generic
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

# 结构表以 lung3d_reconstruct.py 为单一来源；脚本被单独拷贝时退回内置副本。
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lung3d_reconstruct import (  # type: ignore
        LUNG_VESSEL_CLASSES,
        NODULE_LABEL,
        STRUCTURE_LABELS,
    )
except Exception:  # pragma: no cover - 单文件分发时的兜底
    LUNG_VESSEL_CLASSES = {
        "lung_arteries": 1,
        "lung_veins": 2,
        "lung_airways": 3,
        "lung_airways_wall": 4,
    }
    NODULE_LABEL = 5
    STRUCTURE_LABELS = {
        "lung_arteries": ("肺动脉", [0.20, 0.40, 1.00], "vessel"),
        "lung_veins": ("肺静脉", [0.85, 0.17, 0.17], "vessel"),
        "lung_airways": ("气管",       [0.20, 1.00, 0.30], "airway"),
        "lung_airways_wall": ("气道壁", [0.90, 0.60, 0.20], "wall"),
        "lung_nodules": ("肺结节", [1.00, 1.00, 0.10], "lesion"),
    }

DEFAULT_LABEL_MAP = {**{v: k for k, v in LUNG_VESSEL_CLASSES.items()}, NODULE_LABEL: "lung_nodules"}


def label_map_for(shape_data, existing=None):
    """给定多标签体数据，返回 {标签值: 结构key}。

    优先沿用 existing（目录内已有的 labels.json）；否则按内置结构表套用，
    表外的标签值退化为 label_<n>（查看端会显示为「结构 n」）。
    """
    if existing:
        return dict(existing)
    labels = np.unique(shape_data)
    labels = [int(v) for v in labels if int(v) > 0]
    return {lab: DEFAULT_LABEL_MAP.get(lab, "label_%d" % lab) for lab in labels}


def build_labels_payload(label_map, generic=False):
    """构造 labels.json 的内容。generic=True 时只给 key，不给显示名/配色。"""
    items = []
    for lab, key in sorted(label_map.items()):
        if generic:
            items.append({"label": int(lab), "key": key})
            continue
        name, color, role = STRUCTURE_LABELS.get(key, (key, None, ""))
        item = {"label": int(lab), "key": key, "name": name, "role": role}
        if color:
            item["color"] = list(color)
        items.append(item)
    return {"schema": "lung3d.labels/1", "structures": items}


def read_label_map_from_dir(src_dir):
    """读取已有目录里的 labels.json / structures.json，返回 {标签值: key} 或 None。"""
    for fn in ("labels.json", "structures.json"):
        p = Path(src_dir) / fn
        if not p.is_file():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = raw.get("structures") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            continue
        out = {}
        for i, it in enumerate(items, start=1):
            if not isinstance(it, dict):
                continue
            try:
                lab = int(it.get("label", i))
            except Exception:
                lab = i
            out[lab] = str(it.get("key") or ("label_%d" % lab))
        if out:
            return out
    return None


def pack_one(src, out_root, name, want_zip=False, generic=False):
    """把一份数据整理成规范目录（+ 可选 zip）。

    返回 (目录路径, zip路径或 None, {标签值: 结构key})。
    """
    import nibabel as nib

    src = Path(src)
    out_root = Path(out_root)
    dst = out_root / name
    dst.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        combined_src = src / "combined.nii.gz"
        if not combined_src.is_file():
            cands = sorted(p for p in src.glob("*.nii.gz"))
            if not cands:
                raise SystemExit(f"[打包] 目录内找不到 .nii.gz：{src}")
            combined_src = cands[0]
        existing = read_label_map_from_dir(src)
    else:
        combined_src = src
        existing = None

    img = nib.load(str(combined_src))
    data = np.asarray(img.dataobj)
    if data.ndim != 3:
        raise SystemExit(f"[打包] 期望三维体数据，实际维度 {data.ndim}：{combined_src}")

    label_map = label_map_for(data, existing)
    payload = build_labels_payload(label_map, generic=generic)

    # 1) 体数据
    shutil.copyfile(combined_src, dst / "combined.nii.gz")

    # 2) 标注
    (dst / "labels.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) 报告
    voxel_mm3 = float(abs(np.linalg.det(img.affine)))
    counts = {lab: int((data == lab).sum()) for lab in sorted(label_map)}
    report = {
        "case": name,
        "source": str(combined_src),
        "labels": {str(k): v for k, v in sorted(label_map.items())},
        "volumes_cm3": {
            label_map[lab]: round(cnt * voxel_mm3 / 1000.0, 3) for lab, cnt in counts.items()
        },
        "voxel_counts": {label_map[lab]: cnt for lab, cnt in counts.items()},
        "voxel_spacing_mm": [
            round(float(x), 4) for x in np.linalg.norm(img.affine[:3, :3], axis=0)
        ],
        "shape": [int(x) for x in data.shape],
        "generated_by": "lung3d_pack_data.py",
    }
    (dst / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4) 目录内如果已有 mesh/，一并带走（可选，查看端不依赖）
    mesh_src = (src / "mesh") if src.is_dir() else None
    if mesh_src and mesh_src.is_dir():
        shutil.copytree(mesh_src, dst / "mesh", dirs_exist_ok=True)

    zip_path = None
    if want_zip:
        zip_path = out_root / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(dst.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(dst))
    return dst, zip_path, label_map


def main():
    ap = argparse.ArgumentParser(
        description="把多标签体数据整理成查看端可上传的数据目录（combined.nii.gz + labels.json + report.json）"
    )
    ap.add_argument("--input", required=True, help="多标签 .nii.gz 文件，或含 combined.nii.gz 的目录，或批量父目录")
    ap.add_argument("--name", help="数据名（输出目录名）。批量模式忽略，用文件名派生")
    ap.add_argument("--out", default="web_output/packed", help="输出根目录（默认 web_output/packed）")
    ap.add_argument("--zip", action="store_true", help="同时打包成 <name>.zip，直接在后台上传")
    ap.add_argument("--batch", action="store_true", help="把 --input 目录下每个 *.nii.gz 各自打成一包")
    ap.add_argument("--generic", action="store_true", help="只写 key，不写显示名/配色（查看端显示「结构 N」）")
    args = ap.parse_args()

    src = Path(args.input)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.batch:
        if not src.is_dir():
            raise SystemExit("--batch 需要 --input 是一个目录")
        for p in sorted(src.glob("*.nii.gz")):
            jobs.append((p, p.name[:-7]))
        if not jobs:
            raise SystemExit(f"[打包] {src} 下没有 *.nii.gz")
    else:
        if not args.name:
            raise SystemExit("非批量模式必须指定 --name")
        jobs.append((src, args.name))

    for s, name in jobs:
        dst, zp, lmap = pack_one(s, out_root, name, want_zip=args.zip, generic=args.generic)
        print(f"[打包] {name}")
        print(f"  目录: {dst}")
        if zp:
            print(f"  zip : {zp}  ({zp.stat().st_size / 1024 / 1024:.2f} MB)")
        print("  结构: " + ", ".join(f"标签{k}->{v}" for k, v in sorted(lmap.items())))


if __name__ == "__main__":
    main()
