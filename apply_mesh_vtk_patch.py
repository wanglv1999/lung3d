#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apply_mesh_vtk_patch.py —— 把"全精度等值面引擎"打进 lung3d_api.py

用法（在目标机器上，后端代码所在目录执行）：

    python apply_mesh_vtk_patch.py --check     # 只检查锚点，不改文件
    python apply_mesh_vtk_patch.py --apply      # 应用补丁（自动备份）
    python apply_mesh_vtk_patch.py --revert     # 回滚到最近一次备份

设计原则
--------
1. **锚点式替换，不用行号**：另一台机器的代码可能有别的改动，按行号会错位。
   本脚本按"唯一特征串"定位，找不到就整体中止，绝不半途修改。
2. **原实现完整保留**：把原 mask_to_trimesh 改名为 _mask_to_trimesh_skimage，
   新函数只是**分发器**。把 MESH_ENGINE 改回 skimage 立刻退回老行为。
3. **幂等**：已经打过补丁会直接提示并退出，不会重复插入。
4. **前后校验**：应用后立即做 py_compile 语法检查，失败自动回滚。
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_NAME = "lung3d_api.py"
MODULE_NAME = "lung3d_mesh_vtk.py"
MARKER = "_mask_to_trimesh_skimage"

# ---------------------------------------------------------------------------
# 锚点：(说明, 原文, 替换文)
# ---------------------------------------------------------------------------
ANCHOR_IMPORT = (
    "导入区（挂上新模块，失败不影响启动）",
    "ROOT = Path(__file__).resolve().parent\n",
    """ROOT = Path(__file__).resolve().parent

# [mesh-vtk-patch] 全精度等值面引擎（可选）。导入失败时自动退回 skimage 老路径，
# 这样即使容器里没装 vtk，后端也能照常启动。
try:
    import lung3d_mesh_vtk as _mesh_vtk
except Exception as _e:  # pragma: no cover
    _mesh_vtk = None
    print("  [mesh] lung3d_mesh_vtk 不可用（%s），网格仍走 skimage" % _e)
""",
)

ANCHOR_DEF = (
    "mask_to_trimesh 定义（原名改为 skimage 实现 + 前置分发器）",
    '''def mask_to_trimesh(mask, affine, ds_factor=2, smooth=15, prune_mm=0.0):
    """二值掩膜 -> trimesh（RAS 世界坐标，与桌面查看器一致）。
    prune_mm: >0 时按管径阈值剪除细血管（在降采样后计算，避免内存爆槓）。"""''',
    '''def _mask_to_trimesh_skimage(mask, affine, ds_factor=2, smooth=15, prune_mm=0.0):
    """二值掩膜 -> trimesh（RAS 世界坐标，与桌面查看器一致）。
    prune_mm: >0 时按管径阈值剪除细血管（在降采样后计算，避免内存爆槓）。"""''',
)

ANCHOR_DISPATCH = (
    "在 skimage 实现之前插入分发器",
    "def _mask_to_trimesh_skimage(mask, affine, ds_factor=2, smooth=15, prune_mm=0.0):",
    '''# [mesh-vtk-patch] 引擎分发
# MESH_ENGINE=vtk      -> 全精度 VTK 引擎（推荐，2 核 2G 实测峰值 880MB）
# MESH_ENGINE=skimage  -> 退回原来的 skimage 实现（默认行为不变）
# MESH_ENGINE=auto      -> 装了 vtk 就用 vtk（本补丁的默认）
MESH_ENGINE_ENV = "MESH_ENGINE"


def mesh_engine():
    """当前生效的网格引擎名：vtk / skimage。

    安全阀：如果显式要求 vtk 但模块不可用，**直接抛错**而不是静默退回
    skimage —— 因为退回后会带着 MESH_DS_FACTOR=1 走 skimage，峰值约 4.6GB，
    2G 容器必被 OOM killer 杀掉，而且现象是"上传一直失败"，极难排查。
    """
    want = (os.environ.get(MESH_ENGINE_ENV) or "auto").strip().lower()
    if _mesh_vtk is None:
        if want == "vtk":
            raise RuntimeError(
                "MESH_ENGINE=vtk 但 vtk 模块不可用。请先 pip install vtk，"
                "或把 MESH_ENGINE 改回 skimage。注意 skimage 不能跑全精度"
                "（MESH_DS_FACTOR=1 约需 4.6GB 内存）。")
        return "skimage"
    if want in ("vtk", "skimage"):
        return want
    return _mesh_vtk.engine_name()


def mask_to_trimesh(mask, affine, ds_factor=2, smooth=None, prune_mm=0.0,
                    key=None, role=None):
    """按 MESH_ENGINE 分发到具体实现。

    smooth=None 时：vtk 走 MESH_SMOOTH（默认 5），skimage 走原来的 15。
    key/role 只用来查面数预算（MESH_FACE_BUDGET），不影响几何。
    """
    eng = mesh_engine()
    if eng == "vtk":
        return _mesh_vtk.mask_to_trimesh(
            mask, affine, ds_factor=ds_factor, smooth=smooth,
            prune_mm=prune_mm, key=key, role=role)
    # skimage 兜底：它撑不住全精度，硬性钳到 2 档，避免整机 OOM
    if ds_factor is None or int(ds_factor) < 2:
        print("  [WARN] skimage 引擎不支持全精度（峰值约 4.6GB），已回退 ds_factor=2")
        ds_factor = 2
    s = 15 if smooth is None else int(smooth)
    return _mask_to_trimesh_skimage(
        mask, affine, ds_factor=ds_factor, smooth=s, prune_mm=prune_mm)


def _mask_to_trimesh_skimage(mask, affine, ds_factor=2, smooth=15, prune_mm=0.0):''',
)

ANCHOR_CALL_WALL = (
    "build_case 中气管壁的调用（补 key/role）",
    "            mesh = mask_to_trimesh(m, a, ds_factor=ds_wall, smooth=0)",
    "            mesh = mask_to_trimesh(m, a, ds_factor=ds_wall, smooth=0,\n"
    "                                   key=key, role=role)",
)

ANCHOR_CALL_MAIN = (
    "build_case 中血管/其他结构的调用（补 key/role）",
    "            mesh = mask_to_trimesh(m, a, ds_factor=ds_env, prune_mm=prune_mm)",
    "            mesh = mask_to_trimesh(m, a, ds_factor=ds_env, prune_mm=prune_mm,\n"
    "                                   key=key, role=role)",
)

ANCHORS = [ANCHOR_IMPORT, ANCHOR_DEF, ANCHOR_DISPATCH, ANCHOR_CALL_WALL, ANCHOR_CALL_MAIN]


def find_target(explicit=None):
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            p = p / TARGET_NAME
        return p
    for cand in (Path.cwd() / TARGET_NAME, HERE / TARGET_NAME,
                 HERE.parent / "base" / TARGET_NAME):
        if cand.exists():
            return cand
    return None


def count_occ(text, sub):
    return text.count(sub)


def check(text):
    ok = True
    print("=" * 68)
    print("锚点检查：%d 处" % len(ANCHORS))
    print("=" * 68)
    for i, (label, old, _new) in enumerate(ANCHORS, 1):
        n = count_occ(text, old)
        if i == 3:
            # 第 3 处锚点的原文（改过名的 def 行）在第 2 处替换后才存在，
            # 这里按"替换后文本"校验
            n = count_occ(text.replace(ANCHOR_DEF[1], ANCHOR_DEF[2]), old)
        flag = "OK " if n == 1 else "!! "
        print("  %s[%d/%d] %-42s 命中 %d 次" % (flag, i, len(ANCHORS), label, n))
        if n != 1:
            ok = False
    print("-" * 68)
    if MARKER in text:
        print("  !! 文件里已有 %s，说明补丁可能已经打过了" % MARKER)
        ok = False
    print("  结论：%s" % ("可以打补丁" if ok else "锚点不匹配，已中止（未修改任何文件）"))
    print("=" * 68)
    return ok


def apply(path: Path, with_module=True):
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("已经打过补丁（检测到 %s），无需重复应用。" % MARKER)
        return 0
    if not check(text):
        return 2

    if with_module:
        src = HERE / MODULE_NAME
        if src.exists() and src.resolve() != (path.parent / MODULE_NAME).resolve():
            shutil.copy2(src, path.parent / MODULE_NAME)
            print("已复制模块 -> %s" % (path.parent / MODULE_NAME))

    bak = path.with_suffix(path.suffix + ".bak_%s" % time.strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(path, bak)
    print("已备份 -> %s" % bak.name)

    new = text
    for label, old, rep in ANCHORS:
        assert new.count(old) == 1, "锚点异常：%s" % label
        new = new.replace(old, rep, 1)

    path.write_text(new, encoding="utf-8")
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        path.write_text(text, encoding="utf-8")
        print("!! 语法检查失败，已自动回滚：%s" % e)
        return 3

    print("-" * 68)
    print("补丁应用成功，语法检查通过。")
    print("下一步：")
    print("  1) 设置环境变量 MESH_ENGINE=vtk / MESH_DS_FACTOR=1 / DISTAL_PRUNE_MM=0")
    print("  2) 运行 python verify_mesh_vtk.py --engine vtk  做一次实测")
    print("  3) 重建镜像并重启服务")
    return 0


def revert(path: Path):
    baks = sorted(path.parent.glob(path.name + ".bak_*"))
    if not baks:
        print("没有找到备份文件。")
        return 1
    latest = baks[-1]
    shutil.copy2(latest, path)
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        print("!! 回滚后语法检查失败：%s" % e)
        return 3
    print("已回滚到 %s" % latest.name)
    return 0


def main():
    ap = argparse.ArgumentParser(description="给 lung3d_api.py 打全精度网格补丁")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="只检查锚点")
    g.add_argument("--apply", action="store_true", help="应用补丁")
    g.add_argument("--revert", action="store_true", help="回滚到最近备份")
    ap.add_argument("--dir", default=None, help="后端代码目录，默认当前目录")
    ap.add_argument("--no-module", action="store_true", help="不复制 lung3d_mesh_vtk.py")
    a = ap.parse_args()

    path = find_target(a.dir)
    if path is None or not path.exists():
        print("找不到 %s，请用 --dir 指定后端代码目录。" % TARGET_NAME)
        return 1
    print("目标文件：%s\n" % path)

    if a.check:
        return 0 if check(path.read_text(encoding="utf-8")) else 2
    if a.apply:
        return apply(path, with_module=not a.no_module)
    return revert(path)


if __name__ == "__main__":
    sys.exit(main())
