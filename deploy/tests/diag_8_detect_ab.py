# -*- coding: utf-8 -*-
"""诊断 8：A/B 对比检测 —— 当前(错误) affine  vs  修正后的 affine。

输出两组的原始检出框（低阈值 0.02），并换算到体素/解剖位置。
"""
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lung3d_reconstruct import NoduleDetector, build_lung_field   # noqa: E402

WORK = Path(sys.argv[1])
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
WHICH = sys.argv[3] if len(sys.argv) > 3 else "both"

full = WORK / "ct_full.nii.gz"
img = nib.load(str(full))
print("全分辨率 shape", img.shape, "zooms", img.header.get_zooms())
print("affine z 行:", img.affine[2])

# 肺野（降采样网格，用于解剖定位）
field = np.load(str(WORK / "lungfield.npy"))
nii_ds = nib.load(str(WORK / "ct.nii.gz"))
lung_bbox = None
idx = np.argwhere(field)
lung_bbox = (idx.min(0), idx.max(0))
print("肺野包围盒(x,y,z):", lung_bbox[0], lung_bbox[1])

# 解剖参照：k = inst-1，k 越大越靠下
# 肺野 z 范围
z_lung = (int(idx[:, 2].min()), int(idx[:, 2].max()))
xmid = int(np.median(idx[:, 0]))
print("肺野 z 体素范围 %s  中线 x=%d" % (z_lung, xmid))


def run(tag, nii_path):
    det = NoduleDetector(str(ROOT / "monai_nodule"), device="cuda", score_thresh=THRESH)
    t0 = time.time()
    boxes, scores, _ = det.predict(nii_path)
    dt = time.time() - t0
    print("\n" + "=" * 96)
    print("[%s] 检出 %d 个 (阈值 %.3f)  用时 %.1fs" % (tag, len(boxes), THRESH, dt))
    im = nib.load(str(nii_path))
    aff = im.affine
    inv = np.linalg.inv(aff)
    rows = []
    for b, s in sorted(zip(boxes.tolist(), scores.tolist()), key=lambda t: -t[1]):
        c = inv @ np.array([b[0], b[1], b[2], 1.0])
        v = c[:3]
        # 映射到降采样网格
        c2 = np.linalg.inv(nii_ds.affine) @ np.array([b[0], b[1], b[2], 1.0])
        v2 = np.rint(c2[:3]).astype(int)
        # 左右：x 体素 > xmid 为患者左
        side = "左" if v[0] > 0.3786 * 0 + 0 else ""
        # 用降采样网格判断左右（同一世界坐标）
        side = "左肺" if v2[0] > xmid else "右肺"
        inlung = bool(np.all(v2 >= 0) and np.all(v2 < np.array(field.shape))
                      and field[tuple(v2)])
        zfrac = (v2[2] - z_lung[0]) / max(1, (z_lung[1] - z_lung[0]))
        # 注意：NIfTI k=0 为最上方(肺尖)，k 增大向下(肺底)，故 zfrac 小=上叶
        lobe = ("上叶区" if zfrac < 0.38 else ("中/舌叶区" if zfrac < 0.66 else "下叶区"))
        rows.append(dict(score=s, world=[round(x, 1) for x in b[:3]],
                         size=[round(x, 1) for x in b[3:]],
                         vox_full=[round(float(x), 1) for x in v],
                         vox_ds=[int(x) for x in v2],
                         side=side, lobe=lobe, in_lung_field=inlung,
                         zfrac_of_lung=round(float(zfrac), 3)))
    for r in rows:
        print("  score=%.4f  size=%s mm  %s%s  in肺野=%s  zfrac=%.2f  vox_full=%s"
              % (r["score"], r["size"], r["side"], r["lobe"],
                 r["in_lung_field"], r["zfrac_of_lung"], r["vox_full"]))
    del det
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return rows


out = {}
if WHICH in ("both", "A"):
    out["A_current_affine"] = run("A 当前(错误affine)", str(full))

if WHICH in ("both", "B"):
    # 修正 affine：z 方向取反（k=0 为最上方、k 增大向下）
    fix = full.parent / "ct_full_fixed.nii.gz"
    aff = img.affine.copy()
    aff[2, 2] = -aff[2, 2]
    nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj), aff, img.header), str(fix))
    im2 = nib.load(str(fix))
    print("\n修正后 affine z 行:", im2.affine[2], " axcodes:", nib.aff2axcodes(im2.affine))
    out["B_fixed_affine"] = run("B 修正affine", str(fix))

Path(WORK / "detect_ab.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
print("\n已写出 detect_ab.json")
