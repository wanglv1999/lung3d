# -*- coding: utf-8 -*-
"""诊断 2：肺野分割 + 左肺上叶肺窗图像，用于人工定位结节。

用法:
  python diag_2_lungfield.py <工作目录(ASCII)>
输出:
  lungfield.npy          肺野 bool 体 (与降采样 ct.nii.gz 同网格)
  lung_geom.json         肺野几何参数
  montage_axial_LUL.png  左肺上叶轴位拼图
  coronal_L.png          左肺冠状位 MIP
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import nibabel as nib               # noqa: E402
import numpy as np                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lung3d_reconstruct import build_lung_field   # noqa: E402

WORK = Path(sys.argv[1])
nii_path = WORK / "ct.nii.gz"

nifti = nib.load(str(nii_path))
arr = np.asanyarray(nifti.dataobj)
print("体数据 shape(x,y,z):", arr.shape, " zooms:", nifti.header.get_zooms())

field = build_lung_field(nifti)
if field is None:
    sys.exit("肺野分割失败")
np.save(str(WORK / "lungfield.npy"), field)
print("肺野体素数:", int(field.sum()))

idx = np.argwhere(field)
(x0, y0, z0), (x1, y1, z1) = idx.min(0), idx.max(0)
print("肺野包围盒 体素 x[%d,%d] y[%d,%d] z[%d,%d]" % (x0, x1, y0, y1, z0, z1))

sp = np.array([abs(nifti.affine[i, i]) for i in range(3)])
print("体素间距 mm:", sp)

# 肺野在 z 方向的逐层面积，用于划分上/中/下
area = field.sum(axis=(0, 1))
nz = arr.shape[2]
zz = np.arange(nz)

# 左右肺：以肺野 x 中位数为界（x 增大 = 患者左侧）
xmid = int(np.median(idx[:, 0]))
print("中线体素 x =", xmid)

# 解剖：RAS x 越大越靠患者右侧 -> 体素 x 越小越靠右
whole = nib.load(str(WORK / "ct_full.nii.gz")) if (WORK / "ct_full.nii.gz").exists() else nifti
full_arr = np.asanyarray(whole.dataobj)
print("全分辨率 shape:", full_arr.shape)

geom = {
    "shape": list(arr.shape),
    "zooms": [float(v) for v in nifti.header.get_zooms()],
    "bbox_vox": [int(x0), int(x1), int(y0), int(y1), int(z0), int(z1)],
    "xmid_vox": int(xmid),
    "lung_z0": int(z0), "lung_z1": int(z1),
    "z_first_slice_with_lung": int(np.argmax(area > 0)),
    "z_last_slice_with_lung": int(nz - 1 - np.argmax(area[::-1] > 0)),
}
print(json.dumps(geom, indent=2))
(WORK / "lung_geom.json").write_text(json.dumps(geom, indent=2), encoding="utf-8")

# ---------------- 轴位拼图（左肺上叶） ----------------
# 上叶：肺野最上缘往下 45% 范围（粗略）
ztop = geom["z_first_slice_with_lung"]
zbot = geom["z_last_slice_with_lung"]
print("肺野 z 覆盖: %d .. %d (%d 层)" % (ztop, zbot, zbot - ztop + 1))
zu0 = ztop
zu1 = int(ztop + 0.42 * (zbot - ztop))

sel = np.linspace(zu0, zu1, 24).astype(int)
# 裁剪：左肺一侧 (x > xmid) 加一点余量
cx0 = max(0, xmid - 20)
cy0, cy1 = max(0, y0 - 30), min(arr.shape[1] - 1, y1 + 30)
cx1 = min(arr.shape[0] - 1, x1 + 20)

fig, axes = plt.subplots(4, 6, figsize=(24, 17))
for ax, k in zip(axes.ravel(), sel):
    sl = arr[cx0:cx1 + 1, cy0:cy1 + 1, k].T      # (y, x)
    ax.imshow(sl, cmap="gray", vmin=-1000, vmax=200, origin="upper",
              aspect=sp[0] / sp[1])
    ax.set_title("z=%d" % k, fontsize=11)
    ax.axis("off")
fig.suptitle("Axial lung-window montage  LEFT lung upper lobe  (patient LEFT = image RIGHT)",
             fontsize=16)
fig.tight_layout()
fig.savefig(str(WORK / "montage_axial_LUL.png"), dpi=95)
plt.close(fig)
print("已保存 montage_axial_LUL.png")

# ---------------- 冠状位 MIP（左肺） ----------------
# 取左肺 y 中心，做 x-z 平面 MIP
yj = int(np.median(idx[idx[:, 0] > xmid][:, 1]))
slab = arr[cx0:cx1 + 1, max(0, yj - 25):yj + 25, :]
mip = slab.max(axis=1)
fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(mip, cmap="gray", vmin=-1000, vmax=200, origin="lower",
          aspect=sp[2] / sp[0])
ax.set_title("Coronal MIP  LEFT lung  (y=%d +-25)  patient LEFT side" % yj, fontsize=13)
ax.set_xlabel("voxel x (-> patient left)")
ax.set_ylabel("voxel z (-> superior)")
fig.tight_layout()
fig.savefig(str(WORK / "coronal_L.png"), dpi=110)
plt.close(fig)
print("已保存 coronal_L.png  (切片 y=%d)" % yj)

# 逐层肺野左右面积
left_area = field[xmid:, :, :].sum(axis=(0, 1))
right_area = field[:xmid, :, :].sum(axis=(0, 1))
np.save(str(WORK / "z_area.npy"), np.stack([left_area, right_area]))
print("\nz 层  |  左肺面积  右肺面积   (每 20 层采样)")
for k in range(0, nz, 20):
    print("  %4d | %8d %8d" % (k, left_area[k], right_area[k]))
