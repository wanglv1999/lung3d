# -*- coding: utf-8 -*-
"""诊断 18：对比「模型坐标（当前 affine）」与「真实解剖坐标」下的冠状位投影，
直观显示 3D 模型是否上下颠倒、结节落在何处。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
from matplotlib import font_manager  # noqa: E402
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
import nibabel as nib               # noqa: E402
import numpy as np                  # noqa: E402

p = Path(sys.argv[1])
WORK = Path(sys.argv[2])
img = nib.load(str(p))
a = np.asanyarray(img.dataobj)
aff = img.affine
nz = a.shape[2]

COL = {1: (1.0, 0.2, 0.2), 2: (0.35, 0.35, 1.0), 3: (0.2, 1.0, 0.3), 5: (1.0, 1.0, 0.1)}
NAME = {1: "pulmonary artery", 2: "pulmonary vein", 3: "trachea/airway", 5: "NODULE"}

fig, axes = plt.subplots(1, 2, figsize=(22, 11))
for ax, mode in zip(axes, ["true", "model"]):
    # 冠状投影：对 y 取投影，画在 (x, z) 平面
    proj = np.zeros((a.shape[0], nz, 4))
    for L, c in COL.items():
        m = (a == L)
        proj[:, :, :3] = np.where(m.any(axis=1)[:, :, None], np.array(c), proj[:, :, :3])
        proj[:, :, 3] = np.maximum(proj[:, :, 3], m.any(axis=1) * 0.9)
    im = proj.transpose(1, 0, 2)          # (k, x)
    if mode == "model":
        # 模型坐标：world_z = -1174.118 + 0.7k -> k 越大 z 越大 = 显示越靠上
        pass                              # 保持 k 递增向下 -> 等价于 z 递减向下（倒置）
    # 真实解剖：k=0 为头顶侧，应画在最上方 -> 需要翻转
    if mode == "true":
        im = im[::-1]
    ax.imshow(im, origin="upper", aspect=0.7 / 0.7573)
    ax.set_title(("真实解剖（头侧在上）—— 查看端本应显示的样子" if mode == "true"
                  else "当前 affine 导出的模型实际朝向（头侧在下 = 上下颠倒）"), fontsize=14)
    ax.set_xlabel("voxel x (-> patient LEFT)")
    ax.set_ylabel("k=0 在顶部" if mode == "model" else "k=0 在底部")
    ax.grid(alpha=0.15)
handles = [plt.Line2D([], [], color=c, lw=8, label=NAME[L]) for L, c in COL.items()]
axes[0].legend(handles=handles, loc="upper right", fontsize=12, framealpha=0.9)
fig.suptitle("combined.nii.gz 冠状投影：真实解剖 vs 当前 affine 导出的模型朝向\n"
             "判据：真实解剖中气管顶端应高于肺动脉顶端；模型坐标里恰好相反 -> 模型上下颠倒",
             fontsize=16)
fig.tight_layout()
out = WORK / "model_orientation.png"
fig.savefig(str(out), dpi=90)
plt.close(fig)
print("saved", out)

# 数值判据
print("\n判据（真实解剖中气管顶端应最高）：")
for L in (3, 1, 2, 5):
    idx = np.argwhere(a == L)
    if not len(idx):
        continue
    ks = idx[:, 2]
    print("  %-18s k=%d..%d  模型 z=%.1f..%.1f  真实 z=%.1f..%.1f"
          % (NAME[L], ks.min(), ks.max(),
             aff[2, 2] * ks.min() + aff[2, 3], aff[2, 2] * ks.max() + aff[2, 3],
             -1174.118 - 0.7 * ks.max(), -1174.118 - 0.7 * ks.min()))
