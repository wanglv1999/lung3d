# -*- coding: utf-8 -*-
"""诊断 5：像素级精确映射 NIfTI 层号 <-> DICOM 实例号。

转换时是面内 2 取 1（取偏移 1，即 [1::2,1::2]），因此 NIfTI 每一层必然与
某个 DICOM 实例的 [1::2,1::2] 逐像素完全相同 —— 可唯一确定映射。
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

DICOM_DIR = Path(sys.argv[1])
WORK = Path(sys.argv[2])

img = nib.load(str(WORK / "ct.nii.gz"))
arr = np.asanyarray(img.dataobj).astype(np.int32)
nz = arr.shape[2]

rows = []
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows.append((int(ds.InstanceNumber), float(ds.ImagePositionPatient[2]), p))
rows.sort(key=lambda t: t[0])
print("DICOM 层数:", len(rows))

dcm = {}
for inst, z, p in rows:
    ds = pydicom.dcmread(str(p))
    pa = ds.pixel_array.astype(np.int32)
    sl = float(getattr(ds, "RescaleSlope", 1) or 1)
    ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
    dcm[inst] = (pa * sl + ic)[1::2, 1::2]          # 与转换管线同样的抽取

mapping = {}
print("\nNIfTI k -> DICOM InstanceNumber (IPP z)   匹配残差")
for k in range(nz):
    sl = arr[:, :, k]
    best = None
    for inst, ref in dcm.items():
        d = int(np.abs(sl - ref).max())
        if best is None or d < best[1]:
            best = (inst, d)
            if d == 0:
                break
    mapping[k] = best
    if k % 25 == 0 or k in (nz - 1,):
        z = dict((i, zz) for i, zz, _ in rows)[best[0]]
        print("  k=%3d -> inst=%3d (IPP z=%.3f)  maxdiff=%d" % (k, best[0], z, best[1]))

bad = [k for k, (i, d) in mapping.items() if d != 0]
print("\n无法精确匹配(残差>0)的层数:", len(bad))
if bad[:10]:
    print("  示例:", [(k, mapping[k]) for k in bad[:10]])

# 单调性检查
seq = [mapping[k][0] for k in range(nz)]
inc = sum(1 for a, b in zip(seq, seq[1:]) if b > a)
dec = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
eq = sum(1 for a, b in zip(seq, seq[1:]) if b == a)
print("相邻层 instance 递增/递减/相等 次数: %d / %d / %d" % (inc, dec, eq))
print("首尾: k=0 -> inst %d ; k=%d -> inst %d" % (seq[0], nz - 1, seq[-1]))

import json
Path(WORK, "slice_map.json").write_text(json.dumps(
    {"map": {str(k): mapping[k][0] for k in range(nz)},
     "inst_to_z": {str(i): z for i, z, _ in rows}}, indent=1), encoding="utf-8")
print("\n已写出 slice_map.json")
