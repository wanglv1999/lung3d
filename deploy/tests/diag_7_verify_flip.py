# -*- coding: utf-8 -*-
"""诊断 7：决定性验证 —— NIfTI 层索引到底对应哪张 DICOM。

并排渲染 NIfTI k=0 / 10 / 453 与 DICOM instance 1 / 11 / 454，
同时做像素级 max 差比对。
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import nibabel as nib               # noqa: E402
import numpy as np                  # noqa: E402
import pydicom                      # noqa: E402

DICOM_DIR = Path(sys.argv[1])
WORK = Path(sys.argv[2])

img = nib.load(str(WORK / "ct.nii.gz"))
arr = np.asanyarray(img.dataobj).astype(np.int32)

rows = []
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows.append((int(ds.InstanceNumber), float(ds.ImagePositionPatient[2]), p))
rows.sort()
byinst = {i: (z, p) for i, z, p in rows}
N = len(rows)
print("DICOM 实例数", N, " NIfTI 层数", arr.shape[2])

# --- 像素级比对：k 对 inst，以及 k 对 (N+1-inst) ---
diffs_ident, diffs_flip = [], []
for k in [0, 10, 100, 200, 300, 440, 453]:
    sl = arr[:, :, k]
    inst_i = k + 1
    inst_f = N - k
    def dcm_half(inst):
        ds = pydicom.dcmread(str(byinst[inst][1]))
        pa = ds.pixel_array.astype(np.int32)
        return (pa * float(ds.RescaleSlope) + float(ds.RescaleIntercept))[1::2, 1::2]
    di = int(np.abs(sl - dcm_half(inst_i)).max())
    df = int(np.abs(sl - dcm_half(inst_f)).max())
    diffs_ident.append(di); diffs_flip.append(df)
    print("NIfTI k=%3d | 与 inst=%3d 的最大像素差=%6d | 与 inst=%3d 的最大像素差=%6d"
          % (k, inst_i, di, inst_f, df))

print("\n>>> 结论: %s" % (
    "k 与 inst-1 同序（无翻转）" if sum(diffs_ident) < sum(diffs_flip)
    else "k 与 inst 反向（上下翻转）"))

# --- 并排渲染 ---
ks = [0, 10, 453]
insts = [1, 11, 454]
fig, axes = plt.subplots(2, 3, figsize=(17, 12))
for c, (k, inst) in enumerate(zip(ks, insts)):
    ax = axes[0, c]
    ax.imshow(arr[:, :, k].T, cmap="gray", vmin=-1000, vmax=300, origin="upper", aspect=1.0)
    ax.set_title("NIfTI  k=%d" % k, fontsize=15); ax.axis("off")
    ds = pydicom.dcmread(str(byinst[inst][1]))
    hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    ax = axes[1, c]
    ax.imshow(hu[::2, ::2], cmap="gray", vmin=-1000, vmax=300, origin="upper", aspect=1.0)
    ax.set_title("DICOM inst=%d (z=%.1f)" % (inst, byinst[inst][0]), fontsize=15)
    ax.axis("off")
fig.suptitle("TOP row = NIfTI slices / BOTTOM row = DICOM instances\n"
             "if column1 matches -> no flip;  if NIfTI k=0 looks like DICOM 454 -> flip",
             fontsize=16)
fig.tight_layout()
out = WORK / "verify_flip.png"
fig.savefig(str(out), dpi=85)
plt.close(fig)
print("saved", out)
