# -*- coding: utf-8 -*-
"""网格平滑发散（放射状毛刺）的回归测试。

2026-10-19 现象：小程序/网页端气管结构出现很长的放射状毛刺，本机 viewer3d.py
正常。根因是 trimesh.filter_taubin 在 marching_cubes 产生的**非流形顶点**上
数值发散（邻接矩阵行和 >1，放大因子 >1），MESH_DS_FACTOR=3 时气管有 19 个顶点
被推出 280mm。修复为平滑后按位移钳制回滚（smooth_mesh_safe）。

  A 薄壁管发散       -> 模拟气管的薄壁空心管，ds=3 平滑后不得出现超长边
  B 正常实体不误伤   -> 实心球/粗管平滑后几何与旧实现一致，回滚 0 个顶点
  C smooth=0         -> 不平滑时几何与 marching_cubes 原始输出一致
  D 退化输入         -> 面数 <=4 时直接返回 0，不抛异常
  E 位移钳制阈值     -> 钳制上限随中位边长自适应，极粗网格不会误判
  F mesh_max_edge    -> 自检函数能正确识别超长边（导出前的告警依据）
  G 端到端           -> mask_to_trimesh 在 ds=2/3/4 下均无超长边

用法（项目根目录下）：
  ./.venv/Scripts/python.exe deploy/tests/test_mesh_smooth.py
"""
import sys
import traceback
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

import lung3d_api as api  # noqa: E402
import skimage.measure  # noqa: E402
import trimesh  # noqa: E402
from scipy import ndimage  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  [OK]   %s" % name)
    else:
        FAIL += 1
        print("  [FAIL] %s  %s" % (name, detail))


# 单位仿射（1mm 各向同性）
AFFINE = np.array([
    [-1.0, 0.0, 0.0, 100.0],
    [0.0, -1.0, 0.0, 100.0],
    [0.0, 0.0, 1.0, -300.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def thin_wall_tube(shape=(96, 96, 160), r_out=18.0, wall=1.6):
    """薄壁空心管 + 侧支：模拟气管（壁厚 ~1.5 体素，抽点后易产生非流形）。"""
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    tube = (r <= r_out) & (r >= r_out - wall)
    # 一条细侧支（半径 1.2 体素，模拟细支气管）
    branch = ((np.abs(yy - (cy + 30)) <= 1.2) & (np.abs(xx - cx) <= 1.2)
              & (zz > shape[2] * 0.6))
    return (tube | branch).astype(bool)


def solid_ball(shape=(80, 80, 80), radius=22.0):
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    c = np.array(shape, dtype=float) / 2.0
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= radius ** 2


def max_edge_ratio(mesh):
    mx, med, nlong = api.mesh_max_edge(mesh)
    return mx, med, nlong


print("=" * 74)
print("A 薄壁管：ds=3 平滑后不得出现放射状毛刺")
print("=" * 74)
tube = thin_wall_tube()
print("  人造气管 mask 体素=%d shape=%s" % (int(tube.sum()), tube.shape))
m_raw = api.mask_to_trimesh(tube, AFFINE, ds_factor=3, smooth=0)
mx0, med0, nl0 = max_edge_ratio(m_raw)
print("  未平滑: verts=%d max_edge=%.2fmm median=%.2f >10x=%d"
      % (len(m_raw.vertices), mx0, med0, nl0))
check("A1 未平滑时几何干净", nl0 == 0, "超长边 %d 条" % nl0)

m_sm = api.mask_to_trimesh(tube, AFFINE, ds_factor=3, smooth=15)
mx1, med1, nl1 = max_edge_ratio(m_sm)
print("  已平滑: verts=%d max_edge=%.2fmm median=%.2f >10x=%d"
      % (len(m_sm.vertices), mx1, med1, nl1))
check("A2 平滑后无超长边（毛刺已消除）", nl1 == 0, "超长边 %d 条，最长 %.2fmm" % (nl1, mx1))
check("A3 平滑后最长边不超过 10 倍中位边长", mx1 <= med1 * 10,
      "max=%.2f median=%.2f" % (mx1, med1))

# 最小复现样本：两个四面体共享一个顶点（非流形，行和 = 0.6667）
# 真实气管 ds=3 时那 19 个出问题的顶点，行和取值正是 0.6667~0.875
SHARED_V = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
SHARED_F = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3],
            [0, 4, 5], [0, 5, 6], [0, 6, 4], [4, 5, 6]]


def make_shared():
    return trimesh.Trimesh(vertices=np.array(SHARED_V, dtype=float),
                           faces=np.array(SHARED_F, dtype=np.int64))


m_non = make_shared()
L = trimesh.smoothing.laplacian_calculation(m_non)
rowsum = np.asarray(L.sum(axis=1)).ravel()
n_bad = int((np.abs(rowsum - 1.0) > 1e-6).sum())
print("  最小复现样本: 行和≠1 的顶点 %d 个, 行和范围 [%.4f, %.4f]"
      % (n_bad, rowsum.min(), rowsum.max()))
check("A4 最小复现样本确有非流形顶点（行和≠1）", n_bad > 0, "%d 个" % n_bad)
check("A5 行和偏离值落在真实数据的区间内",
      bool(np.any(np.abs(rowsum - 1.0) > 1e-6) and rowsum.min() > 0.6),
      "min=%.4f" % rowsum.min())

# 修复版应把共享顶点固定住（该顶点在 Taubin 迭代中原地不动）
m_fix = make_shared()
v_before = np.array(m_fix.vertices, copy=True)
i_shared = int(np.argmin(np.linalg.norm(v_before, axis=1)))  # 原点处的共享顶点
rolled = api.smooth_mesh_safe(m_fix, iterations=15)
v_after = np.asarray(m_fix.vertices)
d_shared = float(np.linalg.norm(v_after[i_shared] - v_before[i_shared]))
print("  修复版: 回滚=%d 个, 共享顶点位移=%.6fmm" % (rolled, d_shared))
check("A6 共享顶点被固定（不平滑）", d_shared < 1e-6, "位移 %.6fmm" % d_shared)

print()
print("=" * 74)
print("B 正常实体网格：钳制不应误伤")
print("=" * 74)
ball = solid_ball()
m_b = api.mask_to_trimesh(ball, AFFINE, ds_factor=2, smooth=15)
mxb, medb, nlb = max_edge_ratio(m_b)
print("  实心球: verts=%d max_edge=%.2fmm median=%.2f >10x=%d"
      % (len(m_b.vertices), mxb, medb, nlb))
check("B1 实心球平滑后无超长边", nlb == 0, "超长边 %d" % nlb)

# 与旧实现逐点对比：正常网格上二者应完全一致
m_old2 = trimesh.Trimesh(vertices=np.array(m_b.vertices, copy=True),
                         faces=np.array(m_b.faces, copy=True))
# 反向：先还原未平滑再分别跑
m_plain = api.mask_to_trimesh(ball, AFFINE, ds_factor=2, smooth=0)
v0 = np.array(m_plain.vertices, copy=True)
f0 = np.array(m_plain.faces, copy=True)
m_new = trimesh.Trimesh(vertices=v0.copy(), faces=f0.copy())
rolled = api.smooth_mesh_safe(m_new, iterations=15)
m_ref = trimesh.Trimesh(vertices=v0.copy(), faces=f0.copy())
trimesh.smoothing.filter_taubin(m_ref, lamb=0.5, nu=0.5, iterations=15)
diff = float(np.abs(np.asarray(m_new.vertices) - np.asarray(m_ref.vertices)).max())
print("  回滚顶点数=%d  与旧实现最大坐标差=%.6fmm" % (rolled, diff))
check("B2 正常网格回滚 0 个顶点", rolled == 0, "回滚了 %d 个" % rolled)
check("B3 正常网格几何与旧实现完全一致", diff < 1e-6, "差 %.6fmm" % diff)

print()
print("=" * 74)
print("C smooth=0 不做平滑")
print("=" * 74)
m_c = api.mask_to_trimesh(tube, AFFINE, ds_factor=3, smooth=0)
check("C1 smooth=0 与未平滑结果一致",
      np.allclose(np.asarray(m_c.vertices), np.asarray(m_raw.vertices), atol=1e-5),
      "顶点不一致")

print()
print("=" * 74)
print("D 退化输入")
print("=" * 74)
tiny = trimesh.Trimesh(vertices=np.zeros((3, 3), dtype=np.float32),
                       faces=np.array([[0, 1, 2]], dtype=np.int64))
r = api.smooth_mesh_safe(tiny, iterations=15)
check("D1 单面网格返回 0 且不抛异常", r == 0, "返回 %r" % r)
r2 = api.smooth_mesh_safe(tiny, iterations=0)
check("D2 iterations=0 返回 0", r2 == 0, "返回 %r" % r2)
empty = trimesh.Trimesh(vertices=np.zeros((0, 3), dtype=np.float32),
                        faces=np.zeros((0, 3), dtype=np.int64))
check("D3 空网格不抛异常", api.smooth_mesh_safe(empty, iterations=5) == 0)

print()
print("=" * 74)
print("E 钳制阈值随网格尺度自适应")
print("=" * 74)
# 粗网格：边长很大时，阈值应放宽（不误伤），但仍能挡住飞出模型的顶点
coarse = trimesh.creation.icosphere(subdivisions=2, radius=50.0)
med_coarse = float(np.median(coarse.edges_unique_length))
v_before = np.array(coarse.vertices, copy=True)
rolled_c = api.smooth_mesh_safe(coarse, iterations=15)
limit_expect = max(3.0 * med_coarse, 1.5)
print("  粗网格中位边长=%.2fmm 期望阈值=%.2fmm 回滚=%d" % (med_coarse, limit_expect, rolled_c))
check("E1 粗网格正常平滑不触发钳制", rolled_c == 0, "回滚了 %d 个" % rolled_c)

# 人为把一个顶点推飞，验证会被拉回
v_far = np.array(coarse.vertices, copy=True)
v_far[0] += np.array([0.0, 0.0, 500.0])
coarse2 = trimesh.Trimesh(vertices=v_far, faces=np.array(coarse.faces, copy=True))
rolled_d = api.smooth_mesh_safe(coarse2, iterations=15)
check("E2 飞出模型的顶点被回滚", rolled_d >= 1, "回滚了 %d 个" % rolled_d)

print()
print("=" * 74)
print("F mesh_max_edge 自检函数")
print("=" * 74)
good = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
mxg, medg, nlg = api.mesh_max_edge(good)
check("F1 正常网格超长边为 0", nlg == 0, "%d 条" % nlg)
bad_v = np.array(good.vertices, copy=True)
bad_v[0] += np.array([0.0, 0.0, 200.0])
bad_m = trimesh.Trimesh(vertices=bad_v, faces=np.array(good.faces, copy=True))
mxb2, medb2, nlb2 = api.mesh_max_edge(bad_m)
print("  人为毛刺: max_edge=%.1fmm median=%.2f >10x=%d" % (mxb2, medb2, nlb2))
check("F2 能识别出人为毛刺", nlb2 > 0 and mxb2 > 100, "max=%.1f nlong=%d" % (mxb2, nlb2))
check("F3 空网格返回 (0,0,0)", api.mesh_max_edge(empty) == (0.0, 0.0, 0))

print()
print("=" * 74)
print("G 端到端：ds=2/3/4 全部无超长边")
print("=" * 74)
for ds in (2, 3, 4):
    m = api.mask_to_trimesh(tube, AFFINE, ds_factor=ds, smooth=15)
    mx_, med_, nl_ = api.mesh_max_edge(m)
    print("  ds=%d: verts=%-6d max_edge=%7.2fmm median=%.2f >10x=%d"
          % (ds, len(m.vertices), mx_, med_, nl_))
    check("G%d ds=%d 无超长边" % (ds, ds), nl_ == 0, "%d 条，最长 %.2fmm" % (nl_, mx_))

print()
print("=" * 74)
print("结果: %d 通过, %d 失败" % (PASS, FAIL))
print("=" * 74)
sys.exit(0 if FAIL == 0 else 1)
