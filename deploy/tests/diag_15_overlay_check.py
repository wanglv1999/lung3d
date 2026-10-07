# -*- coding: utf-8 -*-
"""诊断 15：检查 DICOM 图像四角是否有烧录(burned-in)文字/标尺（层面号、mAs 等）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pydicom                      # noqa: E402

DICOM_DIR = Path(sys.argv[1])
WORK = Path(sys.argv[2])
inst = int(sys.argv[3])

target = None
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    if int(ds.InstanceNumber) == inst:
        target = p
        break
ds = pydicom.dcmread(str(target))
hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
print("inst", inst, "shape", hu.shape, "HU range", hu.min(), hu.max())

# 检查 overlay 相关 tag
for t in ("OverlayRows", "OverlayData", "BurnedInAnnotation", "ImageType",
          "AcquisitionNumber", "Exposure", "ExposureTime", "XRayTubeCurrent",
          "ExposureInmAs", "StudyInstanceUID", "SeriesInstanceUID"):
    print("  %-22s = %s" % (t, getattr(ds, t, "-")))
print("  (0028,2110) LossyImageCompression =", ds.get((0x0028, 0x2110), "-"))

# 四角裁剪放大
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
crops = [("左上", 0, 140, 0, 520), ("右上", 0, 140, 504, 1024),
         ("左下", 884, 1024, 0, 520), ("右下", 884, 1024, 504, 1024)]
for j, (name, r0, r1, c0, c1) in enumerate(crops):
    ax = axes[0, j]
    ax.imshow(hu[r0:r1, c0:c1], cmap="gray", vmin=-1000, vmax=1500, origin="upper",
              interpolation="nearest")
    ax.set_title("%s  row[%d:%d] col[%d:%d]" % (name, r0, r1, c0, c1), fontsize=12)
    ax.axis("off")
    # 直方图看是否有非解剖的极端值（文字通常是最亮或最暗）
    ax = axes[1, j]
    sub = hu[r0:r1, c0:c1]
    ax.hist(sub.ravel(), bins=80, log=True)
    ax.set_title("HU hist min=%.0f max=%.0f" % (sub.min(), sub.max()), fontsize=11)
fig.suptitle("Burned-in annotation check  inst=%d" % inst, fontsize=16)
fig.tight_layout()
out = WORK / ("corners_inst_%d.png" % inst)
fig.savefig(str(out), dpi=90)
plt.close(fig)
print("saved", out)
