# -*- coding: utf-8 -*-
"""诊断 16：左肺上叶（inst 100-120）高倍放大，两种窗宽各出一张图。"""
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

rows = {}
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows[int(ds.InstanceNumber)] = p

insts = [i for i in range(i0, i1 + 1, step) if i in rows]
print("insts", insts, "crop row[%d:%d] col[%d:%d]" % (r0, r1, c0, c1))

WINDOWS = [("lung", -1350, 150), ("ggo", -1000, -350)]
for wname, vmin, vmax in WINDOWS:
    fig, axes = plt.subplots(3, 4, figsize=(22, 17))
    for ax, inst in zip(axes.ravel(), insts):
        ds = pydicom.dcmread(str(rows[inst]))
        hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
              + float(ds.RescaleIntercept))
        ax.imshow(hu[r0:r1, c0:c1], cmap="gray", vmin=vmin, vmax=vmax,
                  origin="upper", interpolation="nearest", aspect=1.0)
        ax.set_title("inst=%d" % inst, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(insts):]:
        ax.axis("off")
    fig.suptitle("LEFT lung  inst %d-%d step %d   window %s [%.0f, %.0f] HU   crop row[%d:%d] col[%d:%d]"
                 % (i0, i1, step, wname, vmin, vmax, r0, r1, c0, c1), fontsize=16)
    fig.tight_layout()
    out = WORK / ("lul_%s_%d_%d.png" % (wname, i0, i1))
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    print("saved", out)
