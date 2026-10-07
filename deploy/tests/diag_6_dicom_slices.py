# -*- coding: utf-8 -*-
"""诊断 6：直接看指定 DICOM 实例号区间的肺窗图像（原始 1024 分辨率，不经过 NIfTI）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pydicom                      # noqa: E402

DICOM_DIR = Path(sys.argv[1])
WORK = Path(sys.argv[2])
i0 = int(sys.argv[3])
i1 = int(sys.argv[4])

rows = []
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows.append((int(ds.InstanceNumber), p))
rows.sort()
byinst = dict(rows)
print("可用 instance 范围: %d .. %d" % (rows[0][0], rows[-1][0]))

insts = [i for i in range(i0, i1 + 1) if i in byinst]
print("绘制 %d 层: %s" % (len(insts), insts))

n = len(insts)
cols = 6
rws = (n + cols - 1) // cols
fig, axes = plt.subplots(rws, cols, figsize=(cols * 4.2, rws * 4.4))
axes = np.atleast_2d(axes)

for ax, inst in zip(axes.ravel(), insts):
    ds = pydicom.dcmread(str(byinst[inst]))
    pa = ds.pixel_array.astype(np.float32)
    sl = float(getattr(ds, "RescaleSlope", 1) or 1)
    ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
    hu = pa * sl + ic
    # 2x2 块平均 -> 512 显示
    d = hu.reshape(512, 2, 512, 2).mean(axis=(1, 3))
    ax.imshow(d.T, cmap="gray", vmin=-1000, vmax=200, origin="upper", aspect=1.0)
    ax.set_title("inst=%d  z=%.1f" % (inst, float(ds.ImagePositionPatient[2])), fontsize=10)
    ax.axis("off")
for ax in axes.ravel()[len(insts):]:
    ax.axis("off")
fig.suptitle("DICOM instances %d-%d  lung window  (image LEFT = patient RIGHT)" % (i0, i1),
             fontsize=15)
fig.tight_layout()
out = WORK / ("dicom_inst_%d_%d.png" % (i0, i1))
fig.savefig(str(out), dpi=88)
plt.close(fig)
print("saved", out)
