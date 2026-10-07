# -*- coding: utf-8 -*-
"""诊断 12：左肺内多阈值团块搜索（可发现磨玻璃/混合密度结节）。"""
import sys
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage

DICOM_DIR = Path(sys.argv[1])
i0, i1 = int(sys.argv[2]), int(sys.argv[3])
VOX = 0.378634765625 ** 2 * 0.7
PIX = 0.378634765625
DZ = 0.7

rows = {}
for p in sorted(DICOM_DIR.iterdir()):
    if not p.is_file():
        continue
    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    if int(getattr(ds, "Rows", 0) or 0) != 1024:
        continue
    rows[int(ds.InstanceNumber)] = p

insts = [i for i in range(i0, i1 + 1) if i in rows]
vol = np.stack([pydicom.dcmread(str(rows[i])).pixel_array.astype(np.float32)
                * float(pydicom.dcmread(str(rows[i])).RescaleSlope)
                + float(pydicom.dcmread(str(rows[i])).RescaleIntercept) for i in insts])
print("block", vol.shape, "inst", insts[0], "..", insts[-1])

body = np.stack([ndimage.binary_fill_holes(vol[k] > -400) for k in range(vol.shape[0])])
lung = body & (vol < -400)
lung = ndimage.binary_fill_holes(lung)

# 只保留患者左肺（col>512，且是主要连通分量）
lab, n = ndimage.label(lung)
if n:
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = np.argmax(sizes)
    lung = lab == keep
print("肺野体素:", int(lung.sum()))

found = {}
for t in (-700, -600, -500, -400, -300, -200, -100, 0, 100):
    m = (vol > t) & lung
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        idx = np.nonzero(lab == i)
        s = len(idx[0])
        if s < 40 or s > 40000:
            continue
        zz, rr, cc = idx
        if cc.mean() < 512:
            continue
        bb = (np.ptp(zz) + 1) * (np.ptp(rr) + 1) * (np.ptp(cc) + 1)
        fill = s / bb
        if fill < 0.25:
            continue
        ext = [(np.ptp(zz) + 1) * DZ, (np.ptp(rr) + 1) * PIX, (np.ptp(cc) + 1) * PIX]
        if min(ext) < 2.0 or max(ext) > 32:
            continue
        out.append((round(s * VOX, 1), [round(e, 1) for e in ext],
                    int(insts[0] + zz.mean()), round(fill, 2),
                    round(float(vol[lab == i].mean()), 1),
                    float(rr.mean()), float(cc.mean())))
    out.sort(key=lambda x: -x[0])
    found[t] = out
    print("\n>>> 阈值 HU > %4d : 合格团块 %d 个" % (t, len(out)))
    for o in out[:6]:
        print("     vol=%8.1f mm3  ext=%s mm  ~inst %d  填充=%.2f  平均HU=%7.1f  row=%.0f col=%.0f"
              % (o[0], o[1], o[2], o[3], o[4], o[5], o[6]))
