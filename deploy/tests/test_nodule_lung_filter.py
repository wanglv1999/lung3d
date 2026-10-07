# -*- coding: utf-8 -*-
"""肺外检测过滤 (build_lung_field / filter_nodules_outside_lung) 合成用例。

合成 CT：身体轮廓内两团"肺"空气区 + 若干实性结构，
验证：肺内框保留、体外/纵隔远框剔除、肺野填孔包含实性结节、
      胸膜旁框在容差内保留、禁用开关与失败安全。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lung3d_reconstruct import build_lung_field, filter_nodules_outside_lung

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


def make_synthetic():
    """返回 (nifti_like, lung_field_expected_centers)。

    体素 1mm 各向同性，64x64x32。
    身体 = 中央圆柱(HU=30)；左肺/右肺 = 两块矩形空气(HU=-800)；
    肺内实性结节 = 肺中心小球(HU=50，应被填孔纳入肺野)；
    纵隔 = 两肺之间的组织柱(HU=40，肺野外)。
    """
    import nibabel as nib

    shape = (64, 64, 32)
    hu = np.full(shape, -1000.0, dtype=np.float32)          # 体外空气
    xx, yy, zz = np.mgrid[: shape[0], : shape[1], : shape[2]]
    body = ((xx - 32) ** 2 + (yy - 32) ** 2) <= 26 ** 2     # 身体轮廓
    hu[body] = 30.0                                          # 组织
    # 左右肺（空气）：向内收缩 2 体素，保证肺与体外空气之间有组织壁相隔
    # （否则体外空气与肺空气连成一个连通域，真实 CT 中由胸壁分隔）
    from scipy import ndimage
    core = ndimage.binary_erosion(body, iterations=2)
    lung_l = core & (xx < 28) & (yy > 14) & (yy < 50) & (zz > 6) & (zz < 26)
    lung_r = core & (xx > 36) & (yy > 14) & (yy < 50) & (zz > 6) & (zz < 26)
    hu[lung_l | lung_r] = -800.0
    # 肺内实性小球（结节），位于左肺中心 -> 应被填孔算入肺野
    nodule = (xx - 20) ** 2 + (yy - 32) ** 2 + (zz - 16) ** 2 <= 3 ** 2
    hu[nodule] = 50.0
    affine = np.eye(4)
    affine[:3, 3] = [-32.0, -32.0, -16.0]   # 世界坐标居中
    nii = nib.Nifti1Image(hu, affine)
    return nii, affine


def main():
    print("=" * 70)
    print("肺外检测过滤 合成用例")
    print("=" * 70)
    nii, aff = make_synthetic()

    # ---- 1. build_lung_field ----
    field = build_lung_field(nii)
    check("build_lung_field 返回非 None", field is not None)
    if field is None:
        print("FAIL: lung_field 为 None，后续无法测试")
        sys.exit(1)

    # 肺空气区算入
    check("左肺空气算入肺野", field[20, 32, 10])
    check("右肺空气算入肺野", field[44, 32, 16])
    # 肺内实性结节（被空气包围 -> 填孔）算入
    check("肺内实心结节(填孔)算入肺野", field[20, 32, 16])
    # 纵隔（两肺之间组织）不在肺野
    check("纵隔不在肺野", not field[32, 32, 16])
    # 体外空气不在肺野
    check("体外空气不在肺野", not field[2, 2, 16])
    # 身体组织不在肺野
    check("体壁组织不在肺野", not field[32, 58, 16])

    def vox_to_world(ijk):
        h = np.ones(4)
        h[:3] = ijk
        return (aff @ h)[:3]

    # ---- 2. 过滤：肺内框保留 / 体外框剔除 / 纵隔远处框剔除 ----
    boxes = np.array([
        [*vox_to_world((20, 32, 16)), 8, 8, 8],    # 肺内结节中心 -> 保留
        [*vox_to_world((44, 32, 16)), 6, 6, 6],    # 右肺内 -> 保留
        [*vox_to_world((2, 2, 16)), 10, 10, 10],   # 体外空气 -> 剔除
        [*vox_to_world((32, 58, 16)), 8, 8, 8],    # 体壁 -> 剔除
        [*vox_to_world((32, 32, 16)), 6, 6, 6],    # 纵隔中心(离肺野 > 容差) -> 剔除
    ], dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    kept, ks, removed = filter_nodules_outside_lung(boxes, scores, nii, field, margin_mm=3.0)
    check("肺内/体外过滤: 剔除数=3", removed == 3, f"removed={removed}")
    check("保留数=2", len(kept) == 2, f"kept={len(kept)}")
    check("保留分数正确", list(np.round(ks, 2)) == [0.9, 0.8], f"scores={ks}")

    # ---- 3. 容差：胸膜旁(紧贴肺野外)框在 margin 内保留 ----
    # 肺边界 x=28，取 x=30(纵隔侧 2 体素)中心，margin=3 应保留
    box_edge = np.array([[*vox_to_world((30, 32, 16)), 4, 4, 4]], dtype=np.float64)
    kept2, _, removed2 = filter_nodules_outside_lung(
        box_edge, np.array([0.9]), nii, field, margin_mm=3.0)
    check("容差内胸膜旁框保留(margin=3)", removed2 == 0, f"removed={removed2}")
    kept3, _, removed3 = filter_nodules_outside_lung(
        box_edge, np.array([0.9]), nii, field, margin_mm=0.5)
    check("容差外框剔除(margin=0.5)", removed3 == 1, f"removed={removed3}")

    # ---- 4. 失败安全：lung_field=None 原样返回；空框 ----
    b0, s0, r0 = filter_nodules_outside_lung(boxes, scores, nii, None)
    check("lung_field=None 原样返回", r0 == 0 and len(b0) == len(boxes))
    b0, s0, r0 = filter_nodules_outside_lung(np.zeros((0, 6)), np.zeros(0), nii, field)
    check("空框安全", r0 == 0 and len(b0) == 0)

    # ---- 5. 图像外中心剔除 ----
    box_far = np.array([[10000.0, 10000.0, 10000.0, 5, 5, 5]])
    _, _, r5 = filter_nodules_outside_lung(box_far, np.array([0.9]), nii, field)
    check("世界坐标远框(图像外)剔除", r5 == 1, f"removed={r5}")

    # ---- 6. 4D 数据降维 ----
    import nibabel as nib
    hu4 = np.stack([np.asanyarray(nii.dataobj)], axis=-1)
    nii4 = nib.Nifti1Image(hu4, aff)
    f4 = build_lung_field(nii4)
    check("4D 数据可处理", f4 is not None and f4.shape == (64, 64, 32))

    print("-" * 70)
    print(f"通过: {PASS}, 失败: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
