# -*- coding: utf-8 -*-
"""诊断 3：生成左肺（尤其上叶/肺尖）的轴位拼图与冠状位 MIP。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import nibabel as nib               # noqa: E402
import numpy as np                  # noqa: E402

WORK = Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "apex"

nifti = nib.load(str(WORK / "ct.nii.gz"))
arr = np.asanyarray(nifti.dataobj)
sp = np.array([abs(nifti.affine[i, i]) for i in range(3)])
print("shape", arr.shape, "sp", sp)

if mode == "apex":
    zs = np.arange(330, 434, 4)
    j = "_apex"
elif mode == "upper":
    zs = np.arange(270, 434, 7)
    j = "_upper"
else:
    zs = np.arange(60, 434, 16)
    j = "_whole"

# 轴位拼图（全胸横断，显示左右对比）
fig, axes = plt.subplots(5, 6, figsize=(26, 22))
ax_list = list(axes.ravel())
for ax, k in zip(ax_list, zs):
    sl = arr[:, :, k].T
    ax.imshow(sl, cmap="gray", vmin=-1000, vmax=200, origin="upper", aspect=1.0)
    ax.set_title("z=%d (%s)" % (k, "S" if True else ""), fontsize=13)
    ax.axis("off")
for ax in ax_list[len(zs):]:
    ax.axis("off")
fig.suptitle("AXIAL lung window  -  patient RIGHT is on image LEFT", fontsize=20)
fig.tight_layout()
fig.savefig(str(WORK / ("axial" + j + ".png")), dpi=90)
plt.close(fig)
print("saved axial%s.png  z=%s" % (j, list(zs)))

# 左上肺矢状位 MIP（固定 x，看 y-z）
# 左肺 x>228
fig, axes = plt.subplots(1, 4, figsize=(24, 11))
xs = [280, 300, 320, 340]
for ax, xv in zip(axes, xs):
    slab = arr[max(0, xv - 12):xv + 12, :, :]
    mip = slab.max(axis=0)     # (y, z)
    ax.imshow(mip.T, cmap="gray", vmin=-1000, vmax=200, origin="lower",
              aspect=sp[0] / sp[2])
    ax.set_title("sagittal MIP x=%d (patient LEFT side)" % xv, fontsize=13)
    ax.set_xlabel("voxel y (-> posterior)")
    ax.set_ylabel("voxel z (-> superior)")
fig.tight_layout()
fig.savefig(str(WORK / ("sagittal_L" + j + ".png")), dpi=95)
plt.close(fig)
print("saved sagittal_L%s.png" % j)
