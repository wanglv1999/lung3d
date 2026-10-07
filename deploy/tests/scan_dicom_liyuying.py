# -*- coding: utf-8 -*-
"""扫描 DICOM 目录，按 SeriesInstanceUID 分组，输出序列构成与几何信息。
只读操作，不做任何修改。
"""
import os
import sys
from collections import defaultdict

import pydicom

DICOM_DIR = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\wangl\Desktop\李玉英 63151440"


def main():
    files = [f for f in os.listdir(DICOM_DIR) if os.path.isfile(os.path.join(DICOM_DIR, f))]
    print("目录: %s" % DICOM_DIR)
    print("文件总数: %d" % len(files))

    groups = defaultdict(list)
    bad = []
    for i, fn in enumerate(files):
        p = os.path.join(DICOM_DIR, fn)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception as e:
            bad.append((fn, str(e)))
            continue
        key = (
            str(getattr(ds, "SeriesInstanceUID", "NO_UID")),
            str(getattr(ds, "SeriesDescription", "NO_DESC")),
            str(getattr(ds, "Modality", "NO_MOD")),
        )
        groups[key].append((fn, ds))

    print("解析失败: %d" % len(bad))
    for fn, err in bad[:5]:
        print("   ", fn, err)

    print("\n序列数: %d" % len(groups))
    print("=" * 100)
    info = []
    for (uid, desc, mod), items in groups.items():
        ds0 = items[0][1]
        rows = getattr(ds0, "Rows", None)
        cols = getattr(ds0, "Columns", None)
        thick = getattr(ds0, "SliceThickness", None)
        spacing = getattr(ds0, "PixelSpacing", None)
        # 层位置
        zs = []
        for fn, ds in items:
            z = getattr(ds, "ImagePositionPatient", None)
            if z is not None and len(z) == 3:
                zs.append(float(z[2]))
        zs.sort()
        span = (zs[-1] - zs[0]) if len(zs) > 1 else 0.0
        step = (span / (len(zs) - 1)) if len(zs) > 1 else 0.0
        kvp = getattr(ds0, "KVP", None)
        recon = getattr(ds0, "ConvolutionKernel", None)
        info.append(dict(
            uid=uid, desc=desc, mod=mod, n=len(items), rows=rows, cols=cols,
            thick=thick, spacing=spacing, zspan=span, zstep=step,
            kvp=kvp, recon=recon, series_no=getattr(ds0, "SeriesNumber", None),
        ))

    info.sort(key=lambda x: (str(x["mod"]), str(x["desc"])))
    for it in info:
        print("Modality      : %s" % it["mod"])
        print("  SeriesNumber: %s" % it["series_no"])
        print("  Description : %s" % it["desc"])
        print("  层数        : %d" % it["n"])
        print("  Rows x Cols : %s x %s" % (it["rows"], it["cols"]))
        print("  PixelSpacing: %s" % (it["spacing"],))
        print("  SliceThick  : %s" % (it["thick"],))
        print("  实际层间距  : %.4f mm  (z 跨度 %.2f mm)" % (it["zstep"], it["zspan"]))
        print("  KVP         : %s   Kernel: %s" % (it["kvp"], it["recon"]))
        print("  UID         : %s" % it["uid"])
        print("-" * 100)

    # 病人层面信息
    dsf = files and [g for g in groups.values()][0][0][1]
    for tag in ("PatientName", "PatientID", "PatientSex", "PatientAge",
                "StudyDate", "StudyDescription", "BodyPartExamined",
                "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"):
        print("%-24s: %s" % (tag, getattr(dsf, tag, "-")))


if __name__ == "__main__":
    main()
