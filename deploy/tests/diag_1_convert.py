# -*- coding: utf-8 -*-
"""诊断 1：DICOM -> NIfTI 转换与几何检查（只读源数据，输出到 ASCII 临时目录）。

用法:
  python diag_1_convert.py <DICOM目录> <工作目录(ASCII)>
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import nibabel as nib   # noqa: E402

from lung3d_reconstruct import dicom_to_nifti, _pick_best_series_file  # noqa: E402

DICOM_DIR = Path(sys.argv[1])
WORK = Path(sys.argv[2])


def describe(tag, p):
    img = nib.load(str(p))
    arr = np.asanyarray(img.dataobj)
    zooms = img.header.get_zooms()
    print("=" * 90)
    print("[%s] %s" % (tag, p))
    print("  shape (x,y,z)  : %s" % (img.shape,))
    print("  zooms (x,y,z)  : %.5f, %.5f, %.5f mm" % (zooms[0], zooms[1], zooms[2]))
    print("  dtype          : %s" % arr.dtype)
    print("  值域           : min=%s max=%s" % (arr.min(), arr.max()))
    print("  方向 affine    :\n%s" % np.array2string(img.affine, precision=3))
    print("  axcodes        : %s" % (nib.aff2axcodes(img.affine),))
    # 物理范围
    corners = np.array([[0, 0, 0, 1], [arr.shape[0] - 1, 0, 0, 1],
                        [0, arr.shape[1] - 1, 0, 1], [0, 0, arr.shape[2] - 1, 1]])
    w = (img.affine @ corners.T).T[:, :3]
    print("  物理范围 RAS X : %.1f .. %.1f  (跨度 %.1f mm)" % (w[:, 0].min(), w[:, 0].max(),
                                                              w[:, 0].max() - w[:, 0].min()))
    print("  物理范围 RAS Y : %.1f .. %.1f  (跨度 %.1f mm)" % (w[:, 1].min(), w[:, 1].max(),
                                                              w[:, 1].max() - w[:, 1].min()))
    print("  物理范围 RAS Z : %.1f .. %.1f  (跨度 %.1f mm)" % (w[:, 2].min(), w[:, 2].max(),
                                                              w[:, 2].max() - w[:, 2].min()))
    return arr, img


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    # 先确认选中的是哪个序列
    fl = _pick_best_series_file(DICOM_DIR)
    print(">>> _pick_best_series_file 选中文件数: %d" % len(fl))
    import pydicom
    ds0 = pydicom.dcmread(str(fl[0]), stop_before_pixels=True)
    print(">>> 选中序列 SeriesNumber=%s Desc=%s Rows=%s UID=%s" % (
        getattr(ds0, "SeriesNumber", "?"), getattr(ds0, "SeriesDescription", "?"),
        getattr(ds0, "Rows", "?"), getattr(ds0, "SeriesInstanceUID", "?")))
    dsN = pydicom.dcmread(str(fl[-1]), stop_before_pixels=True)
    print(">>> 首层 IPP=%s  InstanceNumber=%s" % (
        getattr(ds0, "ImagePositionPatient", "?"), getattr(ds0, "InstanceNumber", "?")))
    print(">>> 末层 IPP=%s  InstanceNumber=%s" % (
        getattr(dsN, "ImagePositionPatient", "?"), getattr(dsN, "InstanceNumber", "?")))

    # 检查层间 IPP 是否严格等距、是否有重复 z
    zs = []
    insts = []
    for p in fl:
        d = pydicom.dcmread(str(p), stop_before_pixels=True)
        ipp = getattr(d, "ImagePositionPatient", None)
        zs.append(float(ipp[2]))
        insts.append(int(getattr(d, "InstanceNumber", 0) or 0))
    zs = np.array(zs)
    dz = np.diff(zs)
    print(">>> 层间距: min=%.4f max=%.4f 唯一值=%s" % (
        dz.min(), dz.max(), np.unique(np.round(dz, 4))[:6]))
    print(">>> 重复 z 层数: %d" % (len(zs) - len(np.unique(np.round(zs, 4)))))
    print(">>> InstanceNumber 单调递增: %s" % bool(np.all(np.diff(insts) > 0)))

    nii_path = WORK / "ct.nii.gz"
    full_nii = WORK / "ct_full.nii.gz"
    if nii_path.exists():
        nii_path.unlink()
    if full_nii.exists():
        full_nii.unlink()
    dicom_to_nifti(DICOM_DIR, nii_path, out_full=full_nii)
    print("\n>>> full_nii 是否生成: %s" % full_nii.exists())

    if full_nii.exists():
        describe("全分辨率(检测用)", full_nii)
    describe("降采样(分割/网格用)", nii_path)

    # 对全分辨率体数据做肺部 HU 直方图，便于判断是否为 HU
    src = full_nii if full_nii.exists() else nii_path
    arr = np.asanyarray(nib.load(str(src)).dataobj)
    print("=" * 90)
    print("[全分辨率 HU 直方图] (采样)")
    flat = arr.ravel()[::37]
    for lo, hi in [(-1200, -900), (-900, -700), (-700, -500), (-500, -200),
                   (-200, 100), (100, 500), (500, 1500)]:
        n = int(((flat >= lo) & (flat < hi)).sum())
        print("  [%6d, %6d) : %9d (%.2f%%)" % (lo, hi, n, 100.0 * n / flat.size))


if __name__ == "__main__":
    main()
