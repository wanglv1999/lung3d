# -*- coding: utf-8 -*-
"""诊断 9：放大显示指定 DICOM 实例的左/右肺细节，用于观察结节（含 HU 测量）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pydicom                      # noqa: E402

DICOM_DIR = Path(sys.argv[1])
WORK = Path(sys.argv[2])
i0, i1 = int(sys.argv[3]), int(sys.argv[4])

rows = {}
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows[int(ds.InstanceNumber)] = p

insts = [i for i in range(i0, i1 + 1) if i in rows]
print("渲染实例:", insts)

# 1024 图像：col 增大 = 患者左；row 增大 = 后方
# 左肺大约 col 512..1000, row 100..900 —— 自动定位：取 HU<-400 的连通区域质心
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = np.atleast_2d(axes)
for ax, inst in zip(axes.ravel(), insts):
    ds = pydicom.dcmread(str(rows[inst]))
    hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    d = hu[::4, ::4]                                   # 256x256
    ax.imshow(d, cmap="gray", vmin=-1000, vmax=350, origin="upper", aspect=1.0)
    ax.set_title("inst=%d  z=%.1f" % (inst, float(ds.ImagePositionPatient[2])), fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.ravel()[len(insts):]:
    ax.axis("off")
fig.suptitle("instances %d-%d  (4x downsample).  image LEFT = patient RIGHT" % (i0, i1),
             fontsize=16)
fig.tight_layout()
out = WORK / ("zoom_inst_%d_%d.png" % (i0, i1))
fig.savefig(str(out), dpi=92)
plt.close(fig)
print("saved", out)

# 统计：左肺区域（col 增大方向）内 HU 介于 -600..200 的"实性"像素分布
print("\n实例 | 左肺区(col>512) 软组织像素数(-600..300) | 最大连通直径估计")
from scipy import ndimage
for inst in insts:
    ds = pydicom.dcmread(str(rows[inst]))
    hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    left = hu[:, 512:]
    m = (left > -600) & (left < 300)
    lab, n = ndimage.label(m)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        big = int(sizes.max())
    else:
        big = 0
    print("  %3d | %6d | %6d" % (inst, int(m.sum()), big))
