# -*- coding: utf-8 -*-
"""修复后验证：层序/朝向是否自洽、左上叶磨玻璃病灶是否被检出。

用法：
  ./.venv/Scripts/python.exe deploy/tests/verify_fix_liyuying.py \
      <DICOM目录> <重建输出case目录>
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lung3d_reconstruct import build_lung_field          # noqa: E402

DICOM_DIR = Path(sys.argv[1])
CASE_DIR = Path(sys.argv[2])
LESION_INST = range(102, 120)

# ---- 真实的 DICOM 层序（按物理位置升序，即修复后 NIfTI 的 k 序）----
rows = []
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    try:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    except Exception:
        continue
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows.append((float(ds.ImagePositionPatient[2]), int(ds.InstanceNumber), p))
rows.sort()                                    # z 升序 -> 与修复后体数据 k 序一致
n_slices = len(rows)
print("DICOM 层数 =", n_slices)
print("物理 z 升序首层: inst=%d z=%.2f ; 末层: inst=%d z=%.2f"
      % (rows[0][1], rows[0][0], rows[-1][1], rows[-1][0]))

ct = nib.load(str(CASE_DIR / "ct.nii.gz"))
comb = nib.load(str(CASE_DIR / "combined.nii.gz"))
aff = comb.affine
a = np.asanyarray(comb.dataobj).astype(np.uint8)
ctarr = np.asanyarray(ct.dataobj).astype(np.int16)
print("\nct.nii.gz   shape=%s spacing=%s" % (ct.shape, tuple(round(float(v), 4) for v in ct.header.get_zooms())))
print("combined    shape=%s" % (comb.shape,))
print("affine z 行 =", np.array2string(aff[2], precision=4))
print("axcodes     =", nib.aff2axcodes(aff))
dz = float(ct.header.get_zooms()[2])

# ---- 检查 1：几何自洽（k=0 是否真的对应最靠足侧那层）----
zf = aff[2, 3]
zl = aff[2, 2] * (comb.shape[2] - 1) + aff[2, 3]
print("\n[检查1] 几何自洽")
print("  NIfTI 第0层世界 z = %.2f  vs DICOM 最靠足侧层 z = %.2f" % (zf, rows[0][0]))
print("  NIfTI 末层世界 z = %.2f  vs DICOM 最靠头侧层 z = %.2f" % (zl, rows[-1][0]))
c1 = abs(zf - rows[0][0]) < 1.0 and abs(zl - rows[-1][0]) < 1.0
print("  >>>", "自洽 ✓" if c1 else "不自洽 ✗")

# ---- 检查 2：模型朝向（气管顶端应高于肺动脉顶端）----
print("\n[检查2] 模型朝向（真实解剖：气管顶端 z 应 > 肺动脉顶端 z）")
NAME = {1: "肺动脉", 2: "肺静脉", 3: "气管", 5: "肺结节"}
tops = {}
for L, nm in NAME.items():
    idx = np.argwhere(a == L)
    if len(idx) == 0:
        print("  %-6s 无voxel" % nm)
        continue
    ks = idx[:, 2]
    tops[nm] = (float(aff[2, 2] * ks.max() + aff[2, 3]), float(aff[2, 2] * ks.min() + aff[2, 3]),
                int(ks.max()), int(ks.min()))
    print("  %-6s 体积 %8d vox | 世界 z %.1f .. %.1f mm | k %.1f .. %.1f"
          % (nm, len(idx), tops[nm][1], tops[nm][0], tops[nm][3], tops[nm][2]))
c2 = None
if "气管" in tops and "肺动脉" in tops:
    c2 = tops["气管"][0] > tops["肺动脉"][0]
    print("  气管顶端 - 肺动脉顶端 = %.1f mm" % (tops["气管"][0] - tops["肺动脉"][0]))
print("  >>>", ("朝向正确(气管在上) ✓" if c2 else "仍上下颠倒 ✗") if c2 is not None else "无法判定")

# ---- 检查 3：检出结节定位（k -> DICOM instance）----
print("\n[检查3] 检出结节（label 5 连通域）定位")
lung = build_lung_field(ct)
if lung is None:
    print("  肺野计算失败")
else:
    lab, n = ndimage.label(a == 5)
    print("  共 %d 个连通域 | 肺野体素占比 %.1f%%" % (n, 100.0 * lung.mean()))
    idxs = np.argwhere(lung)
    xmid = int(np.median(idxs[:, 0]))
    hit = False
    print("  # | voxels | 尺寸 mm (x,y,z)     | 侧 | k 中心 | DICOM inst | 中心 HU | 在肺野")
    for i in range(1, n + 1):
        m = lab == i
        nv = int(m.sum())
        if nv < 3:
            continue
        ii = np.argwhere(m)
        kk, rr, cc = ii[:, 2], ii[:, 0], ii[:, 1]
        sz = ((np.ptp(cc) + 1) * 0.7572, (np.ptp(rr) + 1) * 0.7572, (np.ptp(kk) + 1) * dz)
        kc = int(round(float(kk.mean())))
        # 修复后：k 序 == 物理 z 升序，直接取 DICOM 侧
        inst = rows[min(max(kc, 0), n_slices - 1)][1]
        side = "左" if cc.mean() > xmid else "右"
        hu = float(ctarr[rr, cc, kk].mean())
        inl = bool(lung[int(rr.mean()), int(cc.mean()), kc])
        flag = ""
        if inst in LESION_INST:
            hit = True
            flag = "  <== 用户所指层面区间"
        print("  %d | %6d | %5.1f x %5.1f x %5.1f | %s | %5d | %6d | %7.0f | %s%s"
              % (i, nv, sz[0], sz[1], sz[2], side, kc, inst, hu, "是" if inl else "否", flag))
    print("  >>> 左肺上叶 102-116 层面区间内检出:", "有 ✓" if hit else "无 ✗")

# ---- 检查 4：该层面的实性/磨玻璃密度（复核病灶本身）----
print("\n[检查4] 102-120 层左肺病灶区密度复核")
for z, inst, _ in rows:
    if inst not in LESION_INST:
        continue
    ds = pydicom.dcmread(str(rows[[r[1] for r in rows].index(inst)][2]))
    hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    reg = hu[470:640, 620:790]
    print("  inst %3d | >-300(实性) %5d | -720..-300(磨玻璃) %5d | 均值 %7.0f"
          % (inst, int((reg > -300).sum()), int(((reg > -720) & (reg <= -300)).sum()), float(reg.mean())))

print("\n完成。")
