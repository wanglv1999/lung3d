# -*- coding: utf-8 -*-
"""诊断 13：检查用户遗留的 李玉英.nii.gz（疑似 combined.nii.gz），
解析各标签结构、内置 labels.json、以及结节(label 5)的位置。"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

p = Path(sys.argv[1])
img = nib.load(str(p))
print("shape", img.shape, "zooms", img.header.get_zooms())
print("affine:\n", np.array2string(img.affine, precision=3))
print("axcodes", nib.aff2axcodes(img.affine))

# 头扩展里的 labels.json
hdr = img.header
exts = hdr.extensions
print("\nNIfTI 头扩展数:", len(exts))
for e in exts:
    code = getattr(e, "get_code", lambda: "?")()
    content = e.get_content()
    print("--- ext code=%s len=%d ---" % (code, len(content)))
    try:
        obj = json.loads(content.decode("utf-8"))
        print(json.dumps(obj, ensure_ascii=False, indent=1)[:2000])
    except Exception as ex:
        print("  非 JSON:", content[:200], ex)

arr = np.asanyarray(img.dataobj)
u, c = np.unique(arr, return_counts=True)
print("\n标签值分布:")
for v, n in zip(u.tolist(), c.tolist()):
    print("  label %3d : %10d 体素  (%.3f mm3)" % (v, n, n * 0.7572695016860962 ** 2 * 0.7))

# 结节 label 5 的位置
if 5 in u:
    m = arr == 5
    idx = np.argwhere(m)
    print("\n结节(label5) 体素包围盒 x[%d,%d] y[%d,%d] z[%d,%d]"
          % (idx[:, 0].min(), idx[:, 0].max(), idx[:, 1].min(), idx[:, 1].max(),
             idx[:, 2].min(), idx[:, 2].max()))
    # 连通域（可能多个结节）
    from scipy import ndimage
    lab, n = ndimage.label(m)
    print("结节连通域数:", n)
    for i in range(1, n + 1):
        ii = np.nonzero(lab == i)
        s = len(ii[0])
        ctr = [float(ii[j].mean()) for j in range(3)]
        world = img.affine @ np.array([ctr[0], ctr[1], ctr[2], 1.0])
        ext = [(np.ptp(ii[j]) + 1) * img.header.get_zooms()[j] for j in range(3)]
        print("  结节%d: %d 体素 (%.1f mm3)  体素中心=%.0f,%.0f,%.0f  尺寸≈%s mm  世界=%.1f,%.1f,%.1f"
              % (i, s, s * 0.7572695016860962 ** 2 * 0.7, ctr[0], ctr[1], ctr[2],
                 [round(e, 1) for e in ext], world[0], world[1], world[2]))
