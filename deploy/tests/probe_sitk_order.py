# -*- coding: utf-8 -*-
"""探查 SimpleITK ImageSeriesReader 对"给定文件顺序"的真实处理方式。

目的：确定修复方案。要回答三个问题：
  Q1 显式 SetFileNames 时，ITK 是否保留我们给的顺序（还是自己按 IPP 重排）？
  Q2 ITK 算出的 direction z 符号由什么决定（IOP 叉乘？还是实际层间距）？
  Q3 若把文件列表改成 z 升序，几何是否自然自洽（无需再手工翻 direction）？
只读，不改任何数据。
"""
import sys
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

DICOM_DIR = Path(sys.argv[1])
MAX_N = int(sys.argv[2]) if len(sys.argv) > 2 else 454


def collect():
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
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None:
            continue
        rows.append({
            "path": p,
            "inst": int(getattr(ds, "InstanceNumber", 0) or 0),
            "z": float(ipp[2]),
            "iop": [float(v) for v in getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])],
        })
    rows.sort(key=lambda r: r["inst"])
    return rows


rows = collect()
print("文件数(1024 Rows):", len(rows))
z_first_inst = rows[0]["z"]
z_last_inst = rows[-1]["z"]
print("InstanceNumber 1 的 IPP z = %.3f   最大 InstanceNumber(%d) 的 z = %.3f"
      % (z_first_inst, rows[-1]["inst"], z_last_inst))
print("=> 编号方向:", "自头向足(编号增大→z 减小)" if z_first_inst > z_last_inst
      else "自足向头(编号增大→z 增大)")
print("IOP:", rows[0]["iop"])
r_, c_ = np.array(rows[0]["iop"][:3]), np.array(rows[0]["iop"][3:])
n_ = np.cross(r_, c_)
print("叉乘法向 =", n_)
print()


def probe(tag, files):
    r = sitk.ImageSeriesReader()
    r.MetaDataDictionaryArrayUpdateOn()
    r.LoadPrivateTagsOn()
    r.SetFileNames([str(p) for p in files])
    img = r.Execute()
    arr = sitk.GetArrayFromImage(img)          # (k, row, col)
    sp = img.GetSpacing()
    d = img.GetDirection()
    p0 = img.TransformIndexToPhysicalPoint((0, 0, 0))
    pN = img.TransformIndexToPhysicalPoint((0, 0, img.GetSize()[2] - 1))

    def ippz(p):
        return float(pydicom.dcmread(str(p), stop_before_pixels=True).ImagePositionPatient[2])

    # 像素比对：arr 第 0 层 是否等于 files[0]（同为 1024 网格，无抽取）
    ds0 = pydicom.dcmread(str(files[0]))
    px0 = ds0.pixel_array.astype(np.int32)
    same_as_first = int(np.abs(arr[0][1::2, 1::2].astype(np.int32) - px0[1::2, 1::2]).max()) if False else None
    d_full = int(np.abs(arr[0].astype(np.int32) - px0).max())
    dsN = pydicom.dcmread(str(files[-1]))
    d_full_N = int(np.abs(arr[-1].astype(np.int32) - dsN.pixel_array.astype(np.int32)).max())

    print("[%s]" % tag)
    print("  给出顺序: 首=%s(z=%.1f, inst=%d)  末=%s(z=%.1f, inst=%d)"
          % (files[0].name, ippz(files[0]), pydicom.dcmread(str(files[0]), stop_before_pixels=True).InstanceNumber,
             files[-1].name, ippz(files[-1]), pydicom.dcmread(str(files[-1]), stop_before_pixels=True).InstanceNumber))
    print("  spacing =", tuple(round(v, 4) for v in sp))
    print("  direction =", tuple(round(v, 3) for v in d))
    print("  idx0 物理点 z = %.3f   idxN 物理点 z = %.3f" % (p0[2], pN[2]))
    print("  arr[0] 与 给定首文件 逐像素最大差 = %d ; arr[-1] 与 给定末文件 最大差 = %d" % (d_full, d_full_N))
    ok0 = abs(p0[2] - ippz(files[0])) < 0.01
    okN = abs(pN[2] - ippz(files[-1])) < 0.01
    print("  几何自洽? idx0↔首文件 %s ; idxN↔末文件 %s" % (ok0, okN))
    print("  >>> 结论: %s" % ("自洽 ✓" if (ok0 and okN and d_full == 0 and d_full_N == 0)
                              else "不自洽 ✗（像素顺序与几何声明不符）"))
    print()


files_inst = [r["path"] for r in rows]                                   # 现状：按 InstanceNumber 升序
files_zasc = [r["path"] for r in sorted(rows, key=lambda x: x["z"])]      # 按 z 升序
files_zdesc = [r["path"] for r in sorted(rows, key=lambda x: -x["z"])]    # 按 z 降序

probe("A 现状：按 InstanceNumber 升序（= z 降序）", files_inst)
probe("B 按 IPP z 升序（期望：几何自然自洽）", files_zasc)
probe("C 按 IPP z 降序", files_zdesc)
