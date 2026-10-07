# -*- coding: utf-8 -*-
"""诊断 4：把 NIfTI 的层索引直接映射回 DICOM 实例（按图像内容匹配）。

目的：判断 NIfTI 体积的层序是否与 DICOM 的物理位置一致、是否单调。
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

DICOM_DIR = Path(sys.argv[1])
WORK = Path(sys.argv[2])

img = nib.load(str(WORK / "ct.nii.gz"))
arr = np.asanyarray(img.dataobj).astype(np.float32)
nz = arr.shape[2]
print("NIfTI shape", arr.shape, "affine z:", img.affine[2, 2], img.affine[2, 3])

# 读入所有 DICOM 层（1024），2x 抽取后与 NIfTI 网格一致
files = sorted([p for p in DICOM_DIR.iterdir() if p.is_file()],
               key=lambda p: int(getattr(pydicom.dcmread(str(p), stop_before_pixels=True),
                                        "InstanceNumber", 0) or 0))
slices = []
meta = []
for p in files:
    ds = pydicom.dcmread(str(p))
    if not hasattr(ds, "PixelData") or not hasattr(ds, "ImagePositionPatient"):
        continue
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    pa = ds.pixel_array.astype(np.float32)
    sl = ds.RescaleSlope if hasattr(ds, "RescaleSlope") else 1.0
    ic = ds.RescaleIntercept if hasattr(ds, "RescaleIntercept") else 0.0
    pa = pa * float(sl) + float(ic)
    slices.append(pa)
    meta.append((int(ds.InstanceNumber), float(ds.ImagePositionPatient[2])))

print("读入 DICOM 层数:", len(slices), " 每层 shape:", slices[0].shape)

# 构造 64x64 指纹（块平均下采样），只用中心区域避免边界
def fp(a, n=64):
    s = a.shape[0] // n
    if s > 1:
        a = a[:n * s, :n * s].reshape(n, s, n, s).mean(axis=(1, 3))
    elif a.shape[0] != n:
        a = a[:n, :n]
    b = a.astype(np.float32)
    b = b - b.mean()
    nn = np.linalg.norm(b)
    return (b / nn) if nn > 0 else b

dcm_fp = np.stack([fp(s) for s in slices])
nii_fp = np.stack([fp(arr[:, :, k]) for k in range(nz)])

print("\nNIfTI k | 最佳匹配 InstanceNumber | 该层 IPP z | 相关系数 | affine 推算 z")
for k in list(range(0, nz, 25)) + [nz - 1]:
    v = nii_fp[k].ravel()
    sim = dcm_fp.reshape(len(slices), -1) @ v
    j = int(np.argmax(sim))
    affine_z = img.affine[2, 2] * k + img.affine[2, 3]
    print("%7d | %22d | %11.3f | %8.4f | %12.3f" % (
        k, meta[j][0], meta[j][1], sim[j], affine_z))
