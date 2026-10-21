#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_mesh_vtk.py —— 打完补丁后的一次性实测：面数 / 体积 / 内存 / GLB

用法：
    python verify_mesh_vtk.py --input combined.nii.gz
    python verify_mesh_vtk.py --input combined.nii.gz --engine skimage   # 对照组
    python verify_mesh_vtk.py --input combined.nii.gz --engine vtk --env MESH_TARGET_FACES=200000

输出：每个结构的面数、体积、耗时、GLB 大小，以及整例峰值内存。
Windows / Linux 都能跑（峰值内存分别用 psapi / getrusage）。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
def _mem_probe_setup():
    """返回 mem() -> (当前MB, 峰值MB)。"""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(PMC), ctypes.c_uint32]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int

        def _mem():
            p = PMC()
            p.cb = ctypes.sizeof(PMC)
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                              ctypes.byref(p), p.cb):
                return -1.0, -1.0
            return p.WorkingSetSize / 1e6, p.PeakWorkingSetSize / 1e6
        return _mem

    import resource

    def _mem():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        cur = 0.0
        try:
            with open("/proc/self/statm") as f:
                cur = int(f.read().split()[1]) * 4096 / 1e6
        except Exception:
            cur = peak
        return cur, peak
    return _mem


def glb_stats(path):
    """从 GLB 的 JSON 块读出 (三角形数, 顶点数, 字节数)，不需要 trimesh。"""
    try:
        raw = Path(path).read_bytes()
    except Exception:
        return 0, 0, 0
    n = len(raw)
    if n < 20 or raw[:4] != b"glTF":
        return 0, 0, n
    import json as _json
    import struct
    try:
        jlen = struct.unpack_from("<I", raw, 12)[0]
        doc = _json.loads(raw[20:20 + jlen].decode("utf-8"))
    except Exception:
        return 0, 0, n
    acc = doc.get("accessors") or []
    tris = verts = 0
    for mesh in doc.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            idx = prim.get("indices")
            if idx is not None and idx < len(acc):
                tris += int(acc[idx].get("count", 0)) // 3
            pos = (prim.get("attributes") or {}).get("POSITION")
            if pos is not None and pos < len(acc):
                verts += int(acc[pos].get("count", 0))
    return tris, verts, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="combined.nii.gz 或其它分割文件")
    ap.add_argument("--labels", default=None, help="labels.json，缺省用同目录")
    ap.add_argument("--engine", default=None, choices=["vtk", "skimage"])
    ap.add_argument("--app-dir", default=None, help="lung3d_api.py 所在目录，默认本脚本目录")
    ap.add_argument("--out", default=None, help="GLB 输出目录，缺省临时目录")
    ap.add_argument("--env", action="append", default=[],
                    help="追加环境变量，形如 KEY=VALUE，可多次")
    ap.add_argument("--no-ct", action="store_true", help="跳过 CT 外壳（省内存）")
    a = ap.parse_args()

    app_dir = Path(a.app_dir).resolve() if a.app_dir else Path(__file__).resolve().parent
    sys.path.insert(0, str(app_dir))
    if a.engine:
        os.environ["MESH_ENGINE"] = a.engine
    for kv in a.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k.strip()] = v.strip()

    mem = _mem_probe_setup()

    t_import = time.time()
    import lung3d_api as api
    print("导入 lung3d_api 用时 %.1fs，引擎 = %s"
          % (time.time() - t_import, api.mesh_engine()))
    print("关键参数：MESH_DS_FACTOR=%s  DISTAL_PRUNE_MM=%s  MESH_SMOOTH=%s"
          % (os.environ.get("MESH_DS_FACTOR", "(默认2)"),
             os.environ.get("DISTAL_PRUNE_MM", "(默认0.75)"),
             os.environ.get("MESH_SMOOTH", "(默认5)")))
    print("面数预算：%s" % (os.environ.get("MESH_FACE_BUDGET")
                          or ("MESH_TARGET_FACES=%s"
                              % os.environ.get("MESH_TARGET_FACES", "150000"))))

    src = Path(a.input)
    lab = Path(a.labels) if a.labels else src.parent / "labels.json"
    cfg = api.parse_label_config(str(lab)) if lab.exists() else None
    masks = api.masks_from_nifti(str(src), src.name.split(".")[0], cfg)
    print("载入 %d 个结构，起始峰值 %.0f MB" % (len(masks), mem()[1]))

    ct_data, ct_affine = None, None
    if not a.no_ct:
        ctp = src.parent / "ct.nii.gz"
        if ctp.exists():
            ct_data, ct_affine = api.load_ct(str(ctp))
            print("已载入 CT，峰值 %.0f MB" % mem()[1])

    out_dir = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="mesh_verify_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out_dir / "case_verify"
    if case_dir.exists():
        import shutil
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    t0 = time.time()
    info = api.build_case(case_dir, masks, ct_data, ct_affine, {"case": "verify"}, cfg)
    dt = time.time() - t0
    del masks, ct_data
    gc.collect()

    print("-" * 78)
    print("%-18s %10s %10s %10s %10s" % ("结构", "面数", "体积cm³", "GLB", "顶点"))
    print("-" * 78)
    total_faces = total_bytes = 0
    for st in info.get("structures", []):
        key = st.get("key")
        g = case_dir / "meshes" / ("%s.glb" % key)
        tris, vs, sz = glb_stats(g)
        total_faces += tris
        total_bytes += sz
        print("%-18s %10s %10.1f %10s %10s"
              % (key, format(tris, ","), st.get("volume_cm3", 0),
                 "%.2f MB" % (sz / 1e6), format(vs, ",")))
    if info.get("ct_mesh"):
        g = case_dir / "meshes" / "ct_surface.glb"
        tris, vs, sz = glb_stats(g)
        if sz:
            total_faces += tris
            total_bytes += sz
            print("%-18s %10s %10s %10s %10s"
                  % ("ct_surface", format(tris, ","), "—",
                     "%.2f MB" % (sz / 1e6), format(vs, ",")))
    print("-" * 78)
    cur, peak = mem()
    print("总耗时 %.1fs   峰值内存 %.0f MB   面数合计 %s   GLB 合计 %.2f MB"
          % (dt, peak, format(total_faces, ","), total_bytes / 1e6))
    print("产物目录：%s" % case_dir)
    print("REPORT " + json.dumps({"engine": api.mesh_engine(), "sec": round(dt, 1),
                                  "peak_mb": round(peak), "faces": total_faces,
                                  "glb_mb": round(total_bytes / 1e6, 2),
                                  "structures": len(info.get("structures", []))},
                                 ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
