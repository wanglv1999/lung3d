# -*- coding: utf-8 -*-
"""诊断 19：生成交付用证据图 —— 左肺上叶病灶 + 检出框，两窗对照。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
from matplotlib import font_manager  # noqa: E402
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans CN"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np                  # noqa: E402
import pydicom                      # noqa: E402

DICOM_DIR = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

rows = {}
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows[int(ds.InstanceNumber)] = p

INSTS = [104, 108, 110, 112, 113, 114, 116]
# 检出框（来自用户上一轮输出 combined.nii.gz 的 label5 第 1 个连通域）
BOX = dict(inst=113, row=544, col=687, w_mm=6.9, h_mm=6.9, d_mm=6.9, score=0.984)
PIX = 0.378634765625
R = int(BOX["w_mm"] / PIX / 2)

r0, r1, c0, c1 = 400, 780, 520, 900

fig, axes = plt.subplots(2, len(INSTS), figsize=(3.9 * len(INSTS), 13))
for j, inst in enumerate(INSTS):
    ds = pydicom.dcmread(str(rows[inst]))
    hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
          + float(ds.RescaleIntercept))
    crop = hu[r0:r1, c0:c1]
    for i, (vmin, vmax, ttl) in enumerate([(-1350, 150, "肺窗"), (-1000, -350, "磨玻璃窗")]):
        ax = axes[i, j]
        ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, origin="upper",
                  interpolation="nearest", aspect=1.0)
        # 检出框只在 inst 113 画（框是 3D 的，只有该层有中心）
        if inst == BOX["inst"]:
            ax.add_patch(plt.Circle((BOX["col"] - c0, BOX["row"] - r0), R + 12,
                                    fill=False, ec="red", lw=2.2))
            ax.add_patch(plt.Rectangle((BOX["col"] - c0 - R, BOX["row"] - r0 - R),
                                       2 * R, 2 * R, fill=False, ec="yellow", lw=2.0))
        ax.set_title("inst=%d  %s" % (inst, ttl), fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("左肺上叶病灶（混合磨玻璃结节，inst 106-116）与检出框\n"
             "黄框 = 检出框尺寸(6.9mm)；红圈 = 检出框中心所在层面(inst 113, score 0.984)\n"
             "图像左侧 = 患者右侧", fontsize=15)
fig.tight_layout()
out = OUT / "证据_左上叶病灶与检出框.png"
fig.savefig(str(out), dpi=100)
plt.close(fig)
print("saved", out)
