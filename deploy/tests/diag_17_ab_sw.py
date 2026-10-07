# -*- coding: utf-8 -*-
"""诊断 17：决定性 A/B
A = 现状：错误 affine（z 反向） + 无滑窗（整卷一次前向）
B = 修正：正确 affine      + 官方滑窗（roi 512x512x192, overlap 0.25）
输出低阈值下全部检出框及其位置/大小/HU。
"""
import gc
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lung3d_reconstruct import NoduleDetector   # noqa: E402

WORK = Path(sys.argv[1])
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
USE_SW = (sys.argv[3] if len(sys.argv) > 3 else "1") == "1"

full = WORK / "ct_full.nii.gz"
ds_nii = WORK / "ct.nii.gz"
img = nib.load(str(full))
full_arr = np.asanyarray(img.dataobj).astype(np.int32)

# 修正 affine 的副本
fix = WORK / "ct_full_fixed.nii.gz"
if not fix.exists():
    aff = img.affine.copy()
    aff[2, 2] = -aff[2, 2]
    nib.save(nib.Nifti1Image(full_arr, aff, img.header), str(fix))
print("fixed affine z:", nib.load(str(fix)).affine[2])

field = np.load(str(WORK / "lungfield.npy"))
idx = np.argwhere(field)
z_lung = (int(idx[:, 2].min()), int(idx[:, 2].max()))
xmid = int(np.median(idx[:, 0]))
print("肺野 z 体素 %s  xmid=%d" % (z_lung, xmid))


def predict(det, nii_path):
    import monai
    from monai.apps.detection.transforms.dictionary import (
        AffineBoxToWorldCoordinated, ClipBoxToImaged, ConvertBoxModed)
    from monai.transforms import Compose

    if USE_SW:
        # 官方 bundle 的滑窗设置
        det.detector.set_sliding_window_inferer(
            roi_size=[512, 512, 192], overlap=0.25, sw_batch_size=1,
            mode="constant", device="cpu")

    m = NoduleDetector.__new__(NoduleDetector)
    m.torch = torch
    m.detector = det.detector
    m.network = det.network
    m.device = det.device
    m.score_thresh = 0.0

    img_t = NoduleDetector._preprocess(nii_path)
    batch = img_t.unsqueeze(0).to(m.device)
    with torch.no_grad():
        if m.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                preds = m.detector([batch[0]], use_inferer=USE_SW)
        else:
            preds = m.detector([batch[0]], use_inferer=USE_SW)
    pred = preds[0]
    data = {"box": pred["box"].to("cpu"), "label": pred["label"].to("cpu"),
            "label_scores": pred["label_scores"].to("cpu"), "image": img_t.to("cpu")}
    post = Compose([
        ClipBoxToImaged(box_keys="box", label_keys="label",
                        box_ref_image_keys="image", remove_empty=True),
        AffineBoxToWorldCoordinated(box_keys="box", box_ref_image_keys="image",
                                    affine_lps_to_ras=False),
        ConvertBoxModed(box_keys="box", src_mode="xyzxyz", dst_mode="cccwhd"),
    ])
    out = post(data)
    return out["box"].numpy(), out["label_scores"].numpy()


def report(tag, boxes, scores):
    print("\n" + "=" * 100)
    print("[%s] 检出 %d 个 (全部原始)" % (tag, len(boxes)))
    nii_ds = nib.load(str(ds_nii))
    inv = np.linalg.inv(nii_ds.affine)
    print("  score  | 尺寸 mm (w,h,d)      | 侧别 叶  | 在肺野 | DICOM(inst,row,col) | HU@中心 | zfrac")
    for b, s in sorted(zip(boxes.tolist(), scores.tolist()), key=lambda t: -t[1]):
        c = inv @ np.array([b[0], b[1], b[2], 1.0])
        v = np.rint(c[:3]).astype(int)
        ok = bool(np.all(v >= 0) and np.all(v < np.array(field.shape)))
        inl = bool(ok and field[tuple(v)])
        side = "左肺" if v[0] > xmid else "右肺"
        zf = (v[2] - z_lung[0]) / max(1, z_lung[1] - z_lung[0])
        lobe = "上叶" if zf < 0.38 else ("中/舌叶" if zf < 0.66 else "下叶")
        # DICOM 对应位置
        inst = int(v[2]) + 1
        row = int(2 * v[1] + 1)
        col = int(2 * v[0] + 1)
        hu = float(full_arr[min(1023, max(0, 2 * v[0] + 1)),
                            min(1023, max(0, 2 * v[1] + 1)),
                            int(v[2])]) if 0 <= v[2] < full_arr.shape[2] else float("nan")
        print("  %.4f | %5.1f %5.1f %5.1f | %s %s | %s | (%d,%d,%d) | %7.0f | %.2f"
              % (s, b[3], b[4], b[5], side, lobe, "是" if inl else "否", inst, row, col, hu, zf))


res = {}
for tag, path in [("A 现状(错误affine+无滑窗)", str(full)),
                  ("B 修正affine+官方滑窗", str(fix))]:
    t0 = time.time()
    det = NoduleDetector(str(ROOT / "monai_nodule"), device="cuda", score_thresh=0.0)
    try:
        bx, sc = predict(det, path)
        print("\n>>> %s 用时 %.1fs" % (tag, time.time() - t0))
        report(tag, bx, sc)
        res[tag] = dict(boxes=bx.tolist(), scores=sc.tolist(), sec=time.time() - t0)
    except Exception as e:
        print("\n>>> %s 失败: %s" % (tag, e))
        res[tag] = dict(error=str(e))
    del det
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

Path(WORK / "detect_ab_sw.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
print("\n已写出 detect_ab_sw.json")
