# -*- coding: utf-8 -*-
"""滑窗推理参数调优：在 8GB 显存上找一个「不 OOM 且快」的 roi 配置。

同一份数据、同一个网络，只改滑窗 roi / 拼接设备，比较耗时与峰值显存。
预处理只做一次（各配置共享），因此差异只来自推理本身。

用法：
  ./.venv/Scripts/python.exe deploy/tests/tune_sw.py <DICOM目录> <工作目录> 512x512x192:cpu 512x512x96:cpu ...
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lung3d_reconstruct import NoduleDetector, dicom_to_nifti   # noqa: E402

DICOM_DIR = Path(sys.argv[1])
WORK = Path(sys.argv[2])
SPECS = sys.argv[3:] or ["512x512x192:cpu", "512x512x96:cpu"]
WORK.mkdir(parents=True, exist_ok=True)

full = WORK / "ct_full.nii.gz"
if not full.exists():
    print("转换 DICOM -> 全分辨率 NIfTI …", flush=True)
    dicom_to_nifti(DICOM_DIR, WORK / "ct_ds.nii.gz", out_full=full)
    if not full.exists():
        import shutil
        shutil.copyfile(WORK / "ct_ds.nii.gz", full)

os.environ["NODULE_SLIDING_WINDOW"] = "1"
det = NoduleDetector(str(ROOT / "monai_nodule"), device="cuda", score_thresh=0.05)

# 预处理只做一次
t0 = time.time()
img = NoduleDetector._preprocess(full)
print("预处理(LoadImage/Orientation/Spacing/Scale) %.1fs, shape=%s" % (time.time() - t0, tuple(img.shape)), flush=True)
batch = img.unsqueeze(0).to(det.device)


def run(roi, stitch):
    from monai.apps.detection.transforms.dictionary import (
        AffineBoxToWorldCoordinated, ClipBoxToImaged, ConvertBoxModed)
    from monai.transforms import Compose

    det.sw_roi_size = list(roi)
    det.detector.set_sliding_window_inferer(
        roi_size=list(roi), overlap=det.sw_overlap, sw_batch_size=1,
        mode="constant", device=stitch)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            preds = det.detector([batch[0]], use_inferer=True)
    dt = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3

    pred = preds[0]
    data = {"box": pred["box"].to("cpu"), "label": pred["label"].to("cpu"),
            "label_scores": pred["label_scores"].to("cpu"), "image": img.to("cpu")}
    post = Compose([
        ClipBoxToImaged(box_keys="box", label_keys="label",
                        box_ref_image_keys="image", remove_empty=True),
        AffineBoxToWorldCoordinated(box_keys="box", box_ref_image_keys="image",
                                    affine_lps_to_ras=False),
        ConvertBoxModed(box_keys="box", src_mode="xyzxyz", dst_mode="cccwhd"),
    ])
    out = post(data)
    bx = out["box"].numpy()
    sc = out["label_scores"].numpy()
    keep = sc >= det.score_thresh
    return dt, peak, bx[keep], sc[keep]


rows = []
for spec in SPECS:
    roi_s, _, stitch = spec.partition(":")
    roi = [int(v) for v in roi_s.lower().split("x")]
    stitch = stitch or "cpu"
    print("\n" + "=" * 90, flush=True)
    print(">>> roi=%s stitch=%s" % (roi, stitch), flush=True)
    try:
        dt, peak, bx, sc = run(roi, stitch)
        print("<<< 用时 %.1fs  峰值显存 %.2fGB  检出 %d 个" % (dt, peak, len(bx)), flush=True)
        for b, s in sorted(zip(bx.tolist(), sc.tolist()), key=lambda t: -t[1])[:8]:
            print("    %.4f  c=(%.0f,%.0f,%.0f) whd=(%.1f,%.1f,%.1f)" % (s, b[0], b[1], b[2], b[3], b[4], b[5]),
                  flush=True)
        rows.append({"roi": roi, "stitch": stitch, "sec": round(dt, 1), "peak_gb": round(peak, 2),
                     "n": int(len(bx)), "boxes": bx.tolist(), "scores": sc.tolist()})
    except Exception as e:
        print("<<< 失败: %s" % str(e)[:300], flush=True)
        rows.append({"roi": roi, "stitch": stitch, "error": str(e)[:300]})
    gc.collect()
    torch.cuda.empty_cache()

(WORK / "tune_sw.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n已写出 tune_sw.json", flush=True)
