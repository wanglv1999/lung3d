# -*- coding: utf-8 -*-
"""修复前后对比图：combined.nii.gz 的冠状投影，纵向用「世界 z (mm)」作图。

关键：纵向一律以 affine 给出的世界 z 为坐标（上=头侧），
这样「上下镜像」才会如实显现 —— 修复前气管会跑到下方，修复后在上方。

用法：
  ./.venv/Scripts/python.exe deploy/tests/fig_fix_compare.py <修复前combined> <修复后combined> <输出目录>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import nibabel as nib                    # noqa: E402
from matplotlib import font_manager      # noqa: E402

for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

BEFORE = Path(sys.argv[1])
AFTER = Path(sys.argv[2])
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)

COL = {1: (0.20, 0.40, 1.00), 2: (0.85, 0.17, 0.17), 3: (0.20, 1.00, 0.30), 5: (1.00, 1.00, 0.10)}
NAME = {1: "肺动脉", 2: "肺静脉", 3: "气管", 5: "肺结节"}


def panel_data(path):
    img = nib.load(str(path))
    a = np.asanyarray(img.dataobj).astype(np.uint8)
    aff = img.affine
    nz, nx = a.shape[2], a.shape[0]
    im = np.zeros((nz, nx, 4), dtype=np.float32)          # [k, x]
    for L, c in COL.items():
        m = (a == L).any(axis=1)                          # (x, k)
        m = m.T                                           # (k, x)
        im[:, :, :3] = np.where(m[:, :, None], np.array(c, dtype=np.float32), im[:, :, :3])
        im[:, :, 3] = np.maximum(im[:, :, 3], m * 0.95)
    z0 = float(aff[2, 3])
    z1 = float(aff[2, 2] * (nz - 1) + aff[2, 3])
    if z1 >= z0:
        disp, ylo, yhi = im, z0, z1
    else:
        disp, ylo, yhi = im[::-1], z1, z0
    tops = {}
    for L in (1, 3):
        idx = np.argwhere(a == L)
        tops[L] = float(aff[2, 2] * idx[:, 2].max() + aff[2, 3]) if len(idx) else None
    return disp, ylo, yhi, tops, float(aff[2, 2])


fig, axes = plt.subplots(1, 2, figsize=(20, 10.5))
for ax, (path, title, note) in zip(axes, [
        (BEFORE, "修复前", "未按物理层位置排序 -> 体素层序与 affine 声明相反"),
        (AFTER, "修复后", "按物理层位置升序排列 -> 几何自洽")]):
    disp, ylo, yhi, tops, dz = panel_data(path)
    ax.imshow(disp, origin="lower", aspect=1.0, extent=[0, disp.shape[1], ylo, yhi])
    ax.set_title("%s：%s\naffine z 斜率 = %+.3f   （纵轴 = 世界 z，向上为头侧）" % (title, note, dz), fontsize=13)
    ax.set_xlabel("体素 x  (→ 患者左侧)")
    ax.set_ylabel("世界 z (mm)   ↑ 头侧   ↓ 足侧")
    if tops[1] is not None and tops[3] is not None:
        d = tops[3] - tops[1]
        ok = d > 0
        ax.text(0.02, 0.03,
                "气管顶端 z = %.0f mm\n肺动脉顶端 z = %.0f mm\n差 = %+.0f mm  ->  %s"
                % (tops[3], tops[1], d, "朝向正确（气管在上）" if ok else "上下颠倒（气管在下）"),
                transform=ax.transAxes, fontsize=13, va="bottom",
                bbox=dict(fc="white", alpha=0.88, ec="#888"),
                color="#0a7a0a" if ok else "#b00000")
handles = [plt.Line2D([], [], marker="s", ls="", ms=14, color=c, label=NAME[L]) for L, c in COL.items()]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=13, frameon=False)
fig.suptitle("光子计数 CT（李玉英，西门子 NAEOTOM Alpha）：层序朝向修复前后对比", fontsize=16)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
out = OUT / "修复前后_模型朝向对比.png"
fig.savefig(out, dpi=115)
print("已写出", out)
