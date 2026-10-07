# -*- coding: utf-8 -*-
"""诊断 14：把检出的结节映射回原始 DICOM，测量 HU 并出图验证。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import nibabel as nib               # noqa: E402
import numpy as np                  # noqa: E402
import pydicom                      # noqa: E402
from scipy import ndimage           # noqa: E402

DICOM_DIR = Path(sys.argv[1])
COMB = Path(sys.argv[2])
WORK = Path(sys.argv[3])

comb = nib.load(str(COMB))
arr = np.asanyarray(comb.dataobj)
zooms = comb.header.get_zooms()
print("combined shape", arr.shape, "zooms", zooms)

m = arr == 5
lab, n = ndimage.label(m)
print("结节连通域:", n)
nodes = []
for i in range(1, n + 1):
    ii = np.nonzero(lab == i)
    ctr = [float(ii[j].mean()) for j in range(3)]
    # 512 网格 -> DICOM 1024 网格: 索引 i 对应原始 col=2i+1, j->row=2j+1, k 不变
    row = 2 * ctr[1] + 1
    col = 2 * ctr[0] + 1
    inst = int(round(ctr[2])) + 1        # k = inst-1
    nodes.append(dict(i=i, size=len(ii[0]), ctr=ctr, row=row, col=col, inst=inst))
    print("  结节%d 体素中心 %.0f,%.0f,%.0f -> DICOM row=%.0f col=%.0f inst=%d"
          % (i, ctr[0], ctr[1], ctr[2], row, col, inst))

# 读 DICOM
rows = {}
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows[int(ds.InstanceNumber)] = p


def hu_of(inst):
    ds = pydicom.dcmread(str(rows[inst]))
    return (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
            + float(ds.RescaleIntercept))


print("\n在各结节中心的 HU 剖面:")
for nd in nodes:
    inst = nd["inst"]
    inst = max(1, min(454, inst))
    hu = hu_of(inst)
    r, c = int(round(nd["row"])), int(round(nd["col"]))
    print("\n  结节%d @ inst=%d (row=%d, col=%d):" % (nd["i"], inst, r, c))
    for name, (dr, dc) in [("中心", (0, 0)), ("上1", (-13, 0)), ("下1", (13, 0)),
                           ("左1", (0, -13)), ("右1", (0, 13))]:
        rr = min(1023, max(0, r + dr)); cc = min(1023, max(0, c + dc))
        print("    %s HU=%.0f" % (name, hu[rr, cc]))
    patch = hu[max(0, r - 10):r + 11, max(0, c - 10):c + 11]
    print("    9x9mm 邻域: 均值=%.1f 最小=%.0f 最大=%.0f 标准差=%.1f"
          % (patch.mean(), patch.min(), patch.max(), patch.std()))

# ---- 出图：三个结节所在层面，全图 + 左肺放大，标注红框 ----
fig, axes = plt.subplots(2, 3, figsize=(20, 15))
for j, nd in enumerate(nodes[:3]):
    inst = max(1, min(454, nd["inst"]))
    hu = hu_of(inst)
    r, c = int(round(nd["row"])), int(round(nd["col"]))
    ax = axes[0, j]
    ax.imshow(hu[::2, ::2], cmap="gray", vmin=-1000, vmax=300, origin="upper")
    ax.add_patch(plt.Rectangle((c / 2 - 14, r / 2 - 14), 28, 28, fill=False, ec="red", lw=2))
    ax.set_title("inst=%d  full slice  (red=nodule%d)" % (inst, nd["i"]), fontsize=12)
    ax.axis("off")
    ax = axes[1, j]
    rr0, rr1 = max(0, r - 100), r + 100
    cc0, cc1 = max(0, c - 100), c + 100
    ax.imshow(hu[rr0:rr1, cc0:cc1], cmap="gray", vmin=-1000, vmax=300, origin="upper",
              interpolation="nearest")
    ax.set_title("zoom 4x  HU at center = %.0f" % hu[r, c], fontsize=12)
    ax.axis("off")
fig.suptitle("Detected nodules mapped back to ORIGINAL DICOM (1024 grid)", fontsize=16)
fig.tight_layout()
out = WORK / "nodules_on_dicom.png"
fig.savefig(str(out), dpi=105)
plt.close(fig)
print("\nsaved", out)
