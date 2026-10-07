# -*- coding: utf-8 -*-
"""诊断 10：在指定实例区间内自动寻找左肺内的实性团块（候选结节）并测量。"""
import sys
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage

DICOM_DIR = Path(sys.argv[1])
i0, i1 = int(sys.argv[2]), int(sys.argv[3])

rows = {}
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows[int(ds.InstanceNumber)] = p

insts = [i for i in range(i0, i1 + 1) if i in rows]
vol = []
for inst in insts:
    ds = pydicom.dcmread(str(rows[inst]))
    hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    vol.append(hu)
vol = np.stack(vol)                      # (n, row, col)
print("体块 shape (n,row,col):", vol.shape, " 实例", insts[0], "..", insts[-1])

VOX_MM3 = 0.378634765625 ** 2 * 0.7      # 0.10036 mm^3
SLICE_DZ = 0.7

# 肺野：先取身体轮廓（>-400 填孔），轮廓内的低密度区为含气肺
body = np.zeros_like(vol, dtype=bool)
for k in range(vol.shape[0]):
    body[k] = ndimage.binary_fill_holes(vol[k] > -400)
lung = body & (vol < -400)
lung = ndimage.binary_fill_holes(lung)   # 把肺内血管/结节纳入肺野
print("肺野体素占比 %.1f%%" % (100.0 * lung.mean()))

# 实性结构：>-300 且在肺野内
solid = (vol > -300) & lung
# 去掉与肺野边界相接的大块（胸壁/纵隔）——用 3D 连通域，剔除超大与超小
lab, n = ndimage.label(solid)
sizes = np.bincount(lab.ravel())
print("实性连通域个数:", n)
cands = []
for i in range(1, n + 1):
    s = int(sizes[i])
    if s < 30 or s > 30000:
        continue
    m = lab == i
    zz, rr, cc = np.nonzero(m)
    if cc.mean() < 512:          # col<512 是患者右肺
        continue
    # 紧致度：包围盒填充率
    bb = (np.ptp(zz) + 1) * (np.ptp(rr) + 1) * (np.ptp(cc) + 1)
    fill = s / bb
    ext_mm = [(np.ptp(zz) + 1) * SLICE_DZ,
              (np.ptp(rr) + 1) * 0.378634765625,
              (np.ptp(cc) + 1) * 0.378634765625]
    mean_hu = float(vol[m].mean())
    cands.append(dict(n=s, vol_mm3=round(s * VOX_MM3, 1),
                      ext_mm=[round(e, 2) for e in ext_mm],
                      center_inst=int(insts[0] + zz.mean()),
                      center_row=float(rr.mean()), center_col=float(cc.mean()),
                      mean_hu=round(mean_hu, 1), fill=round(fill, 3),
                      inst_span=[int(insts[0] + zz.min()), int(insts[0] + zz.max())]))
cands.sort(key=lambda c: -c["vol_mm3"])
print("\n左肺候选实性团块（按体积降序，前 12 个）：")
for c in cands[:12]:
    print("  vol=%7.1f mm3  ext=%s mm  HU=%6.1f  inst %d..%d  中心inst=%.1f row=%.0f col=%.0f 填充=%.2f"
          % (c["vol_mm3"], c["ext_mm"], c["mean_hu"], c["inst_span"][0], c["inst_span"][1],
             c["center_inst"], c["center_row"], c["center_col"], c["fill"]))
