# -*- coding: utf-8 -*-
"""DICOM 层序 / 体数据朝向 的回归测试。

背景（真实缺陷）：光子计数 CT（西门子 NAEOTOM Alpha）采用「自头向足编号」
—— InstanceNumber 1 在最上方（z 最大），编号增大 z 反而减小。而 SimpleITK/ITK 的
ImageSeriesReader 保留给定文件顺序、但 direction 第 3 列恒取 IOP 叉乘（轴位即 +z），
于是按 InstanceNumber 排序喂进去会得到「像素层序与 affine 声明相反」的体数据：
affine 的 z 符号错误 -> 分割/结节检测/网格导出整体上下镜像。
常规 CT（自足向头编号）不会触发，所以长期未暴露。

修复：一律按物理层位置(IPP 在层法向上的投影)升序排列切片。
本测试用合成 DICOM 覆盖两种编号方向，确保：
  A 自头向足编号（缺陷场景）-> 排序后 z 单调递增，且转换出的 NIfTI 几何自洽
  B 自足向头编号（常规场景）-> 排序结果与按 InstanceNumber 完全一致（不回归）
  C 无 IOP 时 -> 退回按 InstanceNumber 排序
  D 无 IPP 时 -> 不崩溃，退回 InstanceNumber
  E _verify_slice_geometry 对错误方向能检出并纠正

用法（项目根目录下）：
  ./.venv/Scripts/python.exe deploy/tests/test_slice_orientation.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lung3d_reconstruct import (          # noqa: E402
    _layer_order_note,
    _slice_normal,
    _sort_series_files,
    _verify_slice_geometry,
    dicom_to_nifti,
)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, "" if ok else "  <- " + str(detail)))


def make_series(directory, n_slices=20, rows=64, cols=64, z0=-100.0, dz=2.5,
                head_to_foot=True, with_ipp=True, with_iop=True):
    """写一套合成 CT 序列。

    head_to_foot=True  -> InstanceNumber 1 在 z 最大处（光子 CT 式编号，缺陷场景）
    head_to_foot=False -> InstanceNumber 1 在 z 最小处（常规 CT 编号）
    每层像素值编码其层号，便于校验顺序。
    """
    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    paths = []
    for i in range(n_slices):
        inst = i + 1
        # 真实的物理位置：自头向足时，编号越大 z 越小
        k = (n_slices - 1 - i) if head_to_foot else i
        z = z0 + dz * k
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = CTImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset("", Dataset(), file_meta=fm, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.PatientName = "TEST^ORIENT"
        ds.PatientID = "ORIENT001"
        ds.InstanceNumber = inst
        ds.Rows, ds.Columns = rows, cols
        ds.PixelSpacing = [0.7, 0.7]
        ds.SliceThickness = dz
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = 1
        ds.RescaleIntercept = -1024
        if with_ipp:
            ds.ImagePositionPatient = [-22.0, -22.0, float(z)]
        if with_iop:
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        # 像素：全 -1000HU(空气) 打一个随层号移动的方块，便于识别
        arr = np.full((rows, cols), -1000, dtype=np.int16)
        arr[10:20, (inst * 2) % (cols - 12):(inst * 2) % (cols - 12) + 10] = 100
        ds.PixelData = (arr - (-1024)).astype(np.uint16).tobytes() if False else arr.tobytes()
        ds.BitsAllocated = 16
        ds.PixelRepresentation = 1
        p = directory / ("%06d.dcm" % inst)
        ds.save_as(str(p), write_like_original=False)
        paths.append(p)
    return paths


TMP = Path(tempfile.mkdtemp(prefix="lung3d_orient_test_"))
try:
    # ---------------- A 自头向足编号（缺陷场景） ----------------
    print("\nA 自头向足编号（光子计数 CT 式）：InstanceNumber 1 在 z 最大处")
    da = TMP / "head_to_foot"
    make_series(da, n_slices=20, z0=-100.0, dz=2.5, head_to_foot=True)
    fa = sorted(da.iterdir())
    oa = _sort_series_files(fa)
    za = [float(pydicom.dcmread(str(p), stop_before_pixels=True).ImagePositionPatient[2]) for p in oa]
    check("A1 排序后 z 单调递增", all(za[i] <= za[i + 1] for i in range(len(za) - 1)), za)
    check("A2 首层为 InstanceNumber 最大的（最靠足侧）",
          int(pydicom.dcmread(str(oa[0]), stop_before_pixels=True).InstanceNumber) == 20,
          oa[0].name)
    check("A3 末层为 InstanceNumber 1（最靠头侧）",
          int(pydicom.dcmread(str(oa[-1]), stop_before_pixels=True).InstanceNumber) == 1,
          oa[-1].name)

    nii_a = TMP / "out_head" / "ct.nii.gz"
    dicom_to_nifti(da, nii_a)

    def nifti_consistency(nii_path, files_sorted):
        import nibabel as nib
        img = nib.load(str(nii_path))
        aff = img.affine
        nz = img.shape[2]
        z_of = lambda p: float(pydicom.dcmread(str(p), stop_before_pixels=True).ImagePositionPatient[2])
        # NIfTI 世界坐标(RAS) z 与 DICOM(LPS) z 同号
        z_first = aff[2, 2] * 0 + aff[2, 3]
        z_last = aff[2, 2] * (nz - 1) + aff[2, 3]
        return z_first, z_last, z_of(files_sorted[0]), z_of(files_sorted[-1]), aff

    zf, zl, ef, el, aff = nifti_consistency(nii_a, oa)
    check("A4 NIfTI 首层世界 z 与首文件 IPP 一致 (%.2f vs %.2f)" % (zf, ef), abs(zf - ef) < 0.5)
    check("A5 NIfTI 末层世界 z 与末文件 IPP 一致 (%.2f vs %.2f)" % (zl, el), abs(zl - el) < 0.5)
    check("A6 affine z 斜率为正(索引增大朝头侧)", aff[2, 2] > 0, aff[2, 2])
    check("A7 层序自检描述含「自头向足」",
          "自头向足" in (_layer_order_note(oa) or ""), _layer_order_note(oa))

    # ---------------- B 自足向头编号（常规场景，不得回归） ----------------
    print("\nB 自足向头编号（常规 CT）：InstanceNumber 1 在 z 最小处")
    db = TMP / "foot_to_head"
    make_series(db, n_slices=20, z0=-100.0, dz=2.5, head_to_foot=False)
    fb = sorted(db.iterdir())
    ob = _sort_series_files(fb)
    inst_b = [int(pydicom.dcmread(str(p), stop_before_pixels=True).InstanceNumber) for p in ob]
    check("B1 排序结果 = 按 InstanceNumber 升序（与修复前一致）",
          inst_b == list(range(1, 21)), inst_b)
    nii_b = TMP / "out_foot" / "ct.nii.gz"
    dicom_to_nifti(db, nii_b)
    zf, zl, ef, el, aff = nifti_consistency(nii_b, ob)
    check("B2 NIfTI 几何自洽 (首 %.2f/%.2f 末 %.2f/%.2f)" % (zf, ef, zl, el),
          abs(zf - ef) < 0.5 and abs(zl - el) < 0.5)
    check("B3 affine z 斜率为正", aff[2, 2] > 0, aff[2, 2])

    # ---------------- C 无 IOP ----------------
    print("\nC 无 ImageOrientationPatient")
    dc = TMP / "no_iop"
    make_series(dc, n_slices=12, with_iop=False, head_to_foot=True)
    oc = _sort_series_files(sorted(dc.iterdir()))
    inst_c = [int(pydicom.dcmread(str(p), stop_before_pixels=True).InstanceNumber) for p in oc]
    check("C1 无 IOP 时退回按 InstanceNumber 升序", inst_c == list(range(1, 13)), inst_c)
    check("C2 _slice_normal(None) 返回 None", _slice_normal(None) is None)

    # ---------------- D 无 IPP ----------------
    print("\nD 无 ImagePositionPatient")
    dd = TMP / "no_ipp"
    make_series(dd, n_slices=12, with_ipp=False, head_to_foot=True)
    od = _sort_series_files(sorted(dd.iterdir()))
    inst_d = [int(pydicom.dcmread(str(p), stop_before_pixels=True).InstanceNumber) for p in od]
    check("D1 无 IPP 时不崩溃且按 InstanceNumber 升序", inst_d == list(range(1, 13)), inst_d)

    # ---------------- E _verify_slice_geometry 能检出并纠正 ----------------
    print("\nE _verify_slice_geometry 对错误层序的检出与纠正")
    import SimpleITK as sitk
    f_list = sorted(da.iterdir())          # 故意按 InstanceNumber 升序（= 错误层序）
    r = sitk.ImageSeriesReader()
    r.SetFileNames([str(p) for p in f_list])
    bad = r.Execute()
    note = _verify_slice_geometry(bad, f_list)
    check("E1 错误层序被检出", note is not None and "已翻转" in (note or ""), note)
    # 自检后 k 轴应指向 z 递减方向；末层物理 z 应等于末文件 IPP
    pN = bad.TransformIndexToPhysicalPoint((0, 0, bad.GetSize()[2] - 1))
    z_last_file = float(pydicom.dcmread(str(f_list[-1]), stop_before_pixels=True).ImagePositionPatient[2])
    check("E2 纠正后末层物理 z 与末文件一致 (%.2f vs %.2f)" % (pN[2], z_last_file),
          abs(pN[2] - z_last_file) < 0.5, (pN[2], z_last_file))
    # 正确层序不应触发纠正
    r2 = sitk.ImageSeriesReader()
    r2.SetFileNames([str(p) for p in oa])
    good = r2.Execute()
    check("E3 正确层序不触发纠正", _verify_slice_geometry(good, oa) is None)

finally:
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print()
    print("=" * 74)
    print("结果：%d/%d 通过" % (passed, len(RESULTS)))
    for n, ok, d in RESULTS:
        if not ok:
            print("  未通过:", n, d)
    print("=" * 74)
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(0 if passed == len(RESULTS) else 1)
