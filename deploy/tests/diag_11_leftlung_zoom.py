# -*- coding: utf-8 -*-
"""诊断 11：左肺区域放大裁剪显示（原始分辨率，2x 放大），用于直看结节。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pydicom                      # noqa: E402

DICOM_DIR = Path(sys.argv[1])
WORK = Path(sys.argv[2])
i0, i1, step = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
r0, r1, c0, c1 = (int(v) for v in sys.argv[6:10])
vmin, vmax = float(sys.argv[10]), float(sys.argv[11])

rows = {}
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows[int(ds.InstanceNumber)] = p

insts = [i for i in range(i0, i1 + 1, step) if i in rows]
print("实例:", insts, " 裁剪 row[%d:%d] col[%d:%d] 窗[%.0f,%.0f]" % (r0, r1, c0, c1, vmin, vmax))

cols = 4
rws = (len(insts) + cols - 1) // cols
fig, axes = plt.subplots(rws, cols, figsize=(cols * 5.2, rws * 5.6))
axes = np.atleast_2d(axes)
for ax, inst in zip(axes.ravel(), insts):
    ds = pydicom.dcmread(str(rows[inst]))
    hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    crop = hu[r0:r1, c0:c1]
    ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, origin="upper",
              aspect=0.7 / 0.3786 * 1.0, interpolation="nearest")
    ax.set_title("inst=%d  z=%.1f" % (inst, float(ds.ImagePositionPatient[2])), fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.ravel()[len(insts):]:
    ax.axis("off")
fig.suptitle("LEFT lung zoom  inst %d-%d step %d\nrow[%d:%d] col[%d:%d]  window %.0f..%.0f HU"
             % (i0, i1, step, r0, r1, c0, c1, vmin, vmax), fontsize=15)
fig.tight_layout()
out = WORK / ("leftlung_%d_%d_%d_%d.png" % (i0, i1, int(vmin), int(vmax)))
fig.savefig(str(out), dpi=100)
plt.close(fig)
print("saved", out)
