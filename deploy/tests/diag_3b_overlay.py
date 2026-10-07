# -*- coding: utf-8 -*-
"""诊断 3b：整卷肺窗拼图（含肺野轮廓），用于确证头脚方向。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import nibabel as nib               # noqa: E402
import numpy as np                  # noqa: E402

WORK = Path(sys.argv[1])
img = nib.load(str(WORK / "ct.nii.gz"))
arr = np.asanyarray(img.dataobj).astype(np.int16)
field = np.load(str(WORK / "lungfield.npy"))
nz = arr.shape[2]

# 4x 下采样仅用于显示
disp = arr[::2, ::2, :]
fld = field[::2, ::2, :]

zs = list(range(10, nz, 20))
fig, axes = plt.subplots(6, 4, figsize=(17, 24))
for ax, k in zip(axes.ravel(), zs):
    sl = disp[:, :, k].T
    ax.imshow(np.clip(sl, -1000, 400), cmap="gray", vmin=-1000, vmax=400,
              origin="upper", aspect=1.0)
    m = np.ma.masked_where(~fld[:, :, k].T, fld[:, :, k].T)
    ax.imshow(m, cmap="autumn", alpha=0.35, origin="upper", aspect=1.0)
    ax.set_title("z=%d" % k, fontsize=15)
    ax.axis("off")
fig.suptitle("Whole volume - lung window + lung field overlay (red)\n"
             "image LEFT = patient RIGHT", fontsize=20)
fig.tight_layout()
out = WORK / "axial_whole_overlay.png"
fig.savefig(str(out), dpi=80)
plt.close(fig)
print("saved", out, " slices:", zs)
