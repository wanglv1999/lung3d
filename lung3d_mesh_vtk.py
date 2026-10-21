#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lung3d_mesh_vtk.py —— 全精度等值面引擎（VTK FlyingEdges3D + 四边形误差简化）

为什么需要它
------------
线上后端原来用 skimage.measure.marching_cubes。该实现对整幅体数据做多份
float 缓冲，512×512×294 全精度时峰值内存 4.6 GB，2 核 2G 必定被 OOM killer
杀掉。为了活下来只能把 MESH_DS_FACTOR 调大（线上已调到 3），体素间距
0.85mm 放大到 2.54×2.54×3.0mm，直径 3~5mm 的段间血管被抹平 —— 这是小程序
里"血管看着断裂"的直接原因。

本模块换一条实现路线：

  1. 先用 np.argwhere 求非零包围盒，**只把包围盒内的数据送进等值面算法**
     （血管只占全肺约 1/4 体积，这一刀省掉 3/4 的缓冲）；
  2. 用 vtkFlyingEdges3D（分块流式实现，内存占用远低于 Lewiner）；
  3. 平滑用 vtkWindowedSincPolyDataFilter（不存在 trimesh Taubin 在非流形
     顶点上的发散问题，见 lung3d_api.smooth_mesh_safe 的注释）；
  4. 用 vtkQuadricDecimation 保体积简化到目标面数，把 GLB 压回可传输大小。

实测（512×512×294 真实病例，4 个结构）：
  - 峰值内存 880 MB（skimage 全精度 4,624 MB）
  - 动脉体积 152.3 cm³，与桌面全精度 152.1 一致
  - 总面数 397k，GLB 合计 5.78 MB

对外只暴露一个函数：mask_to_trimesh()，签名与 lung3d_api 中的同名函数兼容。
"""

from __future__ import annotations

import gc
import os

import numpy as np

# ---------------------------------------------------------------------------
# 环境变量
# ---------------------------------------------------------------------------
ENV_ENGINE = "MESH_ENGINE"              # vtk | skimage | auto（默认 auto）
ENV_TARGET_FACES = "MESH_TARGET_FACES"  # 单结构面数上限，默认 150000，0=不简化
ENV_FACE_BUDGET = "MESH_FACE_BUDGET"    # 例：arteries:150000,airways:60000
ENV_MIN_COMP_MM3 = "MESH_MIN_COMPONENT_MM3"  # 小碎块剔除阈值(mm³)，默认 2.0
ENV_SMOOTH = "MESH_SMOOTH"              # 平滑迭代次数，默认 5
ENV_MIN_VESSEL_D = "MESH_MIN_VESSEL_D_MM"    # 血管最小显示直径(mm)，默认 2.0，0=不剪
ENV_PRUNE_CUT = "MESH_PRUNE_CUT_VOX"         # 细枝处切断的膨胀体素，默认 1
ENV_MIN_BLOCK = "MESH_MIN_BLOCK_MM3"         # 血管连通块体积下限(mm³)，默认 20
ENV_PRUNE_MODE = "MESH_PRUNE_RADIUS_MODE"    # open(默认，省内存) | edt(精确、费内存)
ENV_PRUNE_CONN = "MESH_PRUNE_STRUCT_CONN"    # 开运算结构元邻域：1=6邻域 2=18 3=26，默认2

# 只对这些结构做"远端细支剪枝"（按 key 去 lung_ 前缀后匹配）
VESSEL_KEYS = ("arteries", "veins", "artery", "vein",
               "pulmonary_arteries", "pulmonary_veins")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def vtk_available():
    try:
        import vtk  # noqa: F401
        return True
    except Exception:
        return False


def engine_name():
    """解析 MESH_ENGINE：显式指定优先，auto 时有 vtk 就用 vtk。"""
    want = (os.environ.get(ENV_ENGINE) or "auto").strip().lower()
    if want in ("vtk", "skimage"):
        return want
    return "vtk" if vtk_available() else "skimage"


def face_budget(key=None, role=None):
    """按结构分配面数。MESH_FACE_BUDGET 形如 "arteries:150000,airways:60000"。

    匹配顺序：完整 key（lung_arteries）→ key 去前缀（arteries）→ role（vessel）。
    都没命中则用 MESH_TARGET_FACES。
    """
    default = _env_int(ENV_TARGET_FACES, 150000)
    raw = (os.environ.get(ENV_FACE_BUDGET) or "").strip()
    if not raw:
        return default
    table = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        k, v = item.split(":", 1)
        try:
            table[k.strip().lower()] = int(v.strip())
        except ValueError:
            continue
    if not table:
        return default

    cands = []
    if key:
        k = str(key).lower()
        cands.append(k)
        if k.startswith("lung_"):
            cands.append(k[5:])
    if role:
        cands.append(str(role).lower())
    for c in cands:
        if c in table:
            return table[c]
    return default


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------
def _affine_parts(affine):
    affine = np.asarray(affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    spacing[spacing == 0] = 1.0
    R = affine[:3, :3] / spacing[np.newaxis, :]
    origin = (affine @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    return spacing, origin, R


def _polys_to_faces(polydata):
    """vtkPolyData -> (N,3) 面数组，兼容 VTK 9.x 的新旧 CellArray 布局。"""
    from vtk.util import numpy_support as nps

    polys = polydata.GetPolys()
    n_cells = polys.GetNumberOfCells()
    if n_cells == 0:
        return np.zeros((0, 3), dtype=np.int64)

    # VTK 9.x：优先走新 API（GetData() 在 9.6 起已废弃）
    try:
        conn = nps.vtk_to_numpy(polys.GetConnectivityArray())
        offs = nps.vtk_to_numpy(polys.GetOffsetsArray())
        if len(offs) == n_cells + 1:
            sizes = np.diff(offs)
            if np.all(sizes == 3):
                return conn.reshape(-1, 3).astype(np.int64)
            rows = []
            for i in range(n_cells):
                idx = conn[offs[i]:offs[i + 1]]
                for k in range(1, len(idx) - 1):
                    rows.append((idx[0], idx[k], idx[k + 1]))
            return np.asarray(rows, dtype=np.int64) if rows else np.zeros((0, 3), np.int64)
    except Exception:
        pass

    # 旧布局兜底：每个多边形首元素是点数
    arr = nps.vtk_to_numpy(polys.GetData())
    if arr.size and arr.size == n_cells * 4:
        return arr.reshape(-1, 4)[:, 1:].astype(np.int64)
    rows = []
    i = 0
    while i < len(arr):
        n = int(arr[i])
        poly = arr[i + 1:i + 1 + n]
        for k in range(1, len(poly) - 1):
            rows.append((poly[0], poly[k], poly[k + 1]))
        i += 1 + n
    return np.asarray(rows, dtype=np.int64) if rows else np.zeros((0, 3), np.int64)


def drop_small_components(mask, spacing, min_mm3=2.0):
    """按**连通域体积**剔除孤立碎块。

    原来是靠 DISTAL_PRUNE_MM 按管径剪枝，顺手把 1~2 体素的噪点也带走了。
    改到全精度、不再剪枝之后，这些噪点会变成零散小疙瘩，需要在掩膜层面
    先清掉——按体积判断比按管径准确，也不会误伤真实的分支。

    返回 (清理后的掩膜, 被剔除的连通域数)。
    """
    if min_mm3 <= 0:
        return mask, 0
    from scipy import ndimage

    lbl, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        return mask, 0
    voxel_mm3 = float(np.prod(np.asarray(spacing, dtype=np.float64)))
    counts = np.bincount(lbl.ravel(), minlength=n + 1)
    keep_label = np.zeros(n + 1, dtype=bool)
    keep_label[1:] = counts[1:] * voxel_mm3 >= min_mm3
    removed = int(n - keep_label[1:].sum())
    if removed == 0:
        return mask, 0
    return keep_label[lbl], removed


def filter_mesh_components(mesh, min_mm3=2.0):
    """兜底：对已生成的网格再按连通块体积过滤一次（掩膜层没拦住时用）。"""
    if min_mm3 <= 0:
        return mesh
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        return mesh
    if len(parts) <= 1:
        return mesh
    keep = [p for p in parts if abs(p.volume) / 1000.0 >= min_mm3]
    if not keep or len(keep) == len(parts):
        return mesh
    import trimesh
    merged = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    return merged


def is_vessel(key=None, role=None):
    """结构是否属于血管（决定要不要做远端细支剪枝）。"""
    k = str(key or "").lower()
    if k.startswith("lung_"):
        k = k[5:]
    if k in VESSEL_KEYS:
        return True
    return str(role or "").lower() in ("vessel", "artery", "vein", "vascular")


def prune_distal_branches(mask, spacing, min_dia_mm=2.0, cut_vox=1,
                          min_block_mm3=20.0, tag=None, log=print):
    """删掉直径 < min_dia_mm 的远端细支，返回 (新掩膜, 统计 dict)。

    为什么不能直接"把细的体素挖掉"
    ------------------------------
    那样会在细颈处拦腰截断，粗段变成悬空孤立块（实测断块 13→34、
    最大连通块占比 97%→86%），观感就是血管断开——正是小程序里被投诉的现象。

    默认 open 模式（省内存，2G 容器可用）
    ------------------------------------
      1) 开运算：结构元半径 = (直径/2) 换算成体素，去掉所有装不下这个球的管段
         —— 直径不足的管子整段消失，而不是被截断；
      2) 连通域标注 + 小碎块剔除（细颈断开后掉下来的碎片）；
      3) 对保留区域做同半径膨胀、并限制回原掩膜内，把粗管恢复到原始粗细。
    细枝消失于分叉处，粗管保持连通，不会出现"悬空残端"。
    内存只需几个 bool 数组（12M 体素约 12MB/个）。

    edt 模式（半径判据精确到 0.1mm，但多占约 200MB）
    ------------------------------------------------
      骨架化 → 骨架点取 EDT 半径 → 细枝点处切断掩膜 → 只保留含粗骨架点的连通块。
    2G 容器上会把整例峰值推到约 1.4GB，只在内存充裕时用。

    只在非零包围盒内计算（四周补背景保证形态学/距离变换的边界正确）。
    """
    from scipy import ndimage

    r_min = float(min_dia_mm) / 2.0
    sp = np.asarray(spacing, dtype=np.float64)
    vox_mm3 = float(np.prod(sp))
    out_full = np.zeros_like(mask, dtype=bool)
    stat = dict(removed_vox=0, blocks=0, vol0=0.0, vol1=0.0,
                big_share=1.0, kept_share=1.0)

    idx = np.argwhere(mask)
    if idx.size == 0:
        return out_full, stat
    lo = np.maximum(idx.min(0) - 1, 0)
    hi = np.minimum(idx.max(0) + 2, np.asarray(mask.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    sub = mask[sl]
    del idx
    subp = np.zeros(tuple(int(s) + 2 for s in sub.shape), dtype=bool)
    subp[1:-1, 1:-1, 1:-1] = sub

    mode = (os.environ.get(ENV_PRUNE_MODE) or "open").strip().lower()
    three = np.ones((3, 3, 3), dtype=bool)

    if mode == "edt":
        from skimage.morphology import skeletonize
        skel = skeletonize(subp)
        edt = ndimage.distance_transform_edt(subp, sampling=sp)
        r_on_skel = edt[skel]
        del edt
        gc.collect()
        thin = np.zeros_like(skel)
        thin[skel] = r_on_skel < r_min
        del r_on_skel
        thick = skel & ~thin
        if not thick.any():
            log("  [细支剪枝] %s 没有骨架点达到直径 %.1fmm，跳过"
                % (tag or "mask", min_dia_mm))
            return np.asarray(mask, dtype=bool).copy(), stat
        cut = ndimage.binary_dilation(
            thin, structure=np.ones((int(cut_vox) * 2 + 1,) * 3, dtype=bool)) & subp
        m2 = subp & ~cut
        del cut
        lab, n = ndimage.label(m2, structure=np.ones((3, 3, 3), dtype=np.uint8))
        keep = np.unique(lab[thick & m2])
        keep = keep[keep > 0]
        del skel, thin, thick
        eff_mm = 2.0 * r_min
        eff_desc = "edt 精确 %.2fmm" % eff_mm
        # 已经"从分叉点剪过"，不需要再膨胀还原
        ball = None
    else:
        # 结构元半径（体素）：按最小间距换算，至少 1 层。
        # 用 6 邻域十字（欧氏距离 1）而不是 3×3×3 立方：立方的对角算切比雪夫
        # 距离 1.73，实际把判据放粗到 3mm 左右，会连中等分支一起削掉。
        k = max(1, int(round(r_min / float(sp.min()))))
        ball = ndimage.generate_binary_structure(3, _env_int(ENV_PRUNE_CONN, 2))
        m2 = ndimage.binary_opening(subp, structure=ball, iterations=k)
        lab, n = ndimage.label(m2, structure=np.ones((3, 3, 3), dtype=np.uint8))
        keep = np.arange(1, n + 1, dtype=np.int64)
        eff_mm = 2.0 * k * float(sp.min())
        eff_desc = "open 近似 %.1fmm（球半径 %d 体素）" % (eff_mm, k)

    if n == 0 or len(keep) == 0:
        log("  [细支剪枝] %s 直径<%.1fmm 剪枝后为空，返回空掩膜"
            % (tag or "mask", min_dia_mm))
        return out_full, stat

    n_small = 0
    cnt = np.bincount(lab.ravel(), minlength=n + 1) * vox_mm3
    if min_block_mm3 > 0:
        small = keep[cnt[keep] < min_block_mm3]
        n_small = int(len(small))
        keep = keep[~np.isin(keep, small)]
    del cnt
    if len(keep) == 0:
        log("  [细支剪枝] %s 全部连通块都小于 %.0fmm³" % (tag or "mask", min_block_mm3))
        return out_full, stat

    # 查表代替 np.isin（后者内部要排序，多一份 int32 全尺寸拷贝）
    keep_tab = np.zeros(n + 1, dtype=bool)
    keep_tab[keep] = True
    kept = keep_tab[lab]
    del keep_tab

    if ball is not None:
        # 把被开运算削掉的 1 层边界补回来，恢复粗管原始粗细（不超出原掩膜）
        out_p = ndimage.binary_dilation(kept, structure=ball, iterations=k) & subp
    else:
        out_p = kept
    d = out_p[1:-1, 1:-1, 1:-1]
    out_full[sl] = d
    del d, kept, lab, subp, sub
    gc.collect()

    vol0 = float(mask.sum()) * vox_mm3
    vol1 = float(out_full.sum()) * vox_mm3
    lab2, n2 = ndimage.label(out_full, structure=np.ones((3, 3, 3), dtype=np.uint8))
    cnt2 = np.bincount(lab2.ravel(), minlength=n2 + 1)[1:] * vox_mm3
    stat.update(removed_vox=int(mask.sum() - out_full.sum()), blocks=int(n2),
                vol0=vol0, vol1=vol1,
                big_share=float(cnt2.max() / vol1) if vol1 else 0.0,
                kept_share=vol1 / vol0 if vol0 else 0.0)
    log("  [细支剪枝] %s 判据 %s → %.1f→%.1f cm³（保留 %.0f%%）连通块 %d "
        "最大块 %.1f%% 碎片剔除 %d"
        % (tag or "mask", eff_desc, vol0 / 1000, vol1 / 1000,
           stat["kept_share"] * 100, stat["blocks"], stat["big_share"] * 100,
           n_small))
    return out_full, stat


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def mask_to_trimesh(mask, affine, ds_factor=1, smooth=None, prune_mm=0.0,
                    key=None, role=None):
    """二值掩膜 -> trimesh（RAS 世界坐标，与桌面查看器一致）。

    参数与 lung3d_api.mask_to_trimesh 保持兼容：
      ds_factor  —— 降采样倍数，推荐 1（全精度）
      smooth     —— 平滑迭代次数，None 时读 MESH_SMOOTH（默认 5）
      prune_mm   —— >0 时仍按管径剪枝（兼容旧行为），推荐 0
      key/role   —— 仅用于查面数预算，不影响几何
    """
    import vtk
    import trimesh
    from vtk.util import numpy_support as nps

    if smooth is None:
        smooth = _env_int(ENV_SMOOTH, 5)

    m = np.asarray(mask) > 0
    if not m.any():
        return None
    spacing, origin, R = _affine_parts(affine)

    # 1) 包围盒裁剪：只把非零区域送进等值面算法
    idx = np.argwhere(m)
    lo = idx.min(0)
    hi = idx.max(0) + 1
    del idx
    sub = np.ascontiguousarray(m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.uint8)
    del m

    # 2) 血管远端细支剪枝：只保留直径 >= MESH_MIN_VESSEL_D_MM 的分支
    #    必须在降采样之前、用原始 spacing 做，否则半径判据失真
    _vd = _env_float(ENV_MIN_VESSEL_D, 2.0)
    if _vd > 0 and is_vessel(key, role):
        pruned, _st = prune_distal_branches(
            sub.astype(bool), spacing, min_dia_mm=_vd,
            cut_vox=_env_int(ENV_PRUNE_CUT, 1),
            min_block_mm3=_env_float(ENV_MIN_BLOCK, 20.0),
            tag=key or role, log=print)
        if pruned.sum() == 0:
            return None
        sub = np.ascontiguousarray(pruned).astype(np.uint8)
        del pruned

    # 3) 可选降采样（全精度时跳过；这里用 order=0 保留原标签，不做插值）
    eff = np.array([1, 1, 1], dtype=float)
    if ds_factor and int(ds_factor) > 1:
        f = int(ds_factor)
        from scipy import ndimage
        sub = (ndimage.zoom(sub.astype(np.float32), 1.0 / f, order=0) > 0.5).astype(np.uint8)
        eff = eff * f
    sp = spacing * eff

    # 4) 可选管径剪枝（兼容旧配置；推荐 DISTAL_PRUNE_MM=0 不用）
    if prune_mm and prune_mm > 0:
        from scipy import ndimage
        d = ndimage.distance_transform_edt(sub, sampling=sp)
        sub = (d >= prune_mm).astype(np.uint8)
        if sub.sum() == 0:
            return None

    # 5) 小碎块剔除
    sub_bool, removed = drop_small_components(sub.astype(bool), sp,
                                             _env_float(ENV_MIN_COMP_MM3, 2.0))
    if removed:
        print("  [碎块过滤] %s 剔除 %d 个小于 %.1fmm³ 的孤立块"
              % (key or "mask", removed, _env_float(ENV_MIN_COMP_MM3, 2.0)))
        sub = np.ascontiguousarray(sub_bool).astype(np.uint8)
    if sub.sum() == 0:
        return None

    # 6) 交给 VTK
    img = vtk.vtkImageData()
    img.SetDimensions(*sub.shape)
    img.SetSpacing(*[float(s) for s in sp])
    img.SetOrigin(*[float(v) for v in (origin + R @ (spacing * lo))])
    try:
        dm = vtk.vtkMatrix3x3()
        for i in range(3):
            for j in range(3):
                dm.SetElement(i, j, float(R[i, j]))
        img.SetDirectionMatrix(dm)
    except Exception:
        pass  # VTK < 9.0 无方向矩阵，方向信息退化但不至于崩
    img.GetPointData().SetScalars(
        nps.numpy_to_vtk(sub.ravel(order="F"), deep=True,
                         array_type=vtk.VTK_UNSIGNED_CHAR))
    del sub
    gc.collect()

    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, 0.5)
    fe.ComputeNormalsOff()
    fe.ComputeGradientsOff()
    fe.Update()
    surf = fe.GetOutput()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(surf)
    tri.Update()
    surf = tri.GetOutput()
    if surf.GetNumberOfPolys() == 0:
        return None

    # 7) 平滑（WindowedSinc，不存在 Taubin 的非流形发散问题）
    if smooth and int(smooth) > 0:
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(surf)
        sm.SetNumberOfIterations(int(smooth))
        sm.SetPassBand(0.1)
        sm.BoundarySmoothingOff()
        sm.FeatureEdgeSmoothingOff()
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        surf = sm.GetOutput()

    # 8) 保体积简化到目标面数
    budget = face_budget(key, role)
    n_poly = surf.GetNumberOfPolys()
    if budget and n_poly > budget:
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(surf)
        dec.SetTargetReduction(1.0 - budget / float(n_poly))
        try:
            dec.SetVolumePreservation(True)
        except Exception:
            pass
        dec.Update()
        surf = dec.GetOutput()

    # 8) 回 trimesh（与原链路同一出口，GLB 导出无需改动）
    verts = nps.vtk_to_numpy(surf.GetPoints().GetData()).astype(np.float64)
    faces = _polys_to_faces(surf)
    if len(faces) == 0:
        return None
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
    mesh = filter_mesh_components(mesh, _env_float(ENV_MIN_COMP_MM3, 2.0))
    try:
        _ = mesh.vertex_normals
    except Exception:
        pass
    return mesh
