#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rebuild_case.py —— 用当前镜像里的网格引擎原地重建某个病例的 GLB

场景
----
网格引擎/参数升级后（例如换 VTK FlyingEngines3D、调整剪枝阈值），旧病例
目录里的 .glb 仍是旧引擎产物，新旧混看会不一致。本脚本按当前环境变量
重新跑一遍 build_case，覆盖 meshes/ 下的 GLB，**case_id 与分享链接不变**。

用法（在容器内执行）
--------------------
    docker cp source.nii.gz lung3d-web:/tmp/rebuild_src.nii.gz
    docker cp deploy/lighthouse/rebuild_case.py lung3d-web:/tmp/rebuild_case.py
    docker exec lung3d-web python /tmp/rebuild_case.py \
        --case /data/cases/20261019_002351_82ed78 \
        --input /tmp/rebuild_src.nii.gz

注意
----
* 病例目录默认只留 meshes/ 与 report.json，原始 nii.gz 已被暂存清理机制
  删除，所以必须从本机重新上传一份。
* 结构名与 role 靠 labels.json 或 NIfTI 头扩展（桌面工具写入）解析。
  **拿不到标注时结构名会退化成 label_1，血管剪枝不会执行** —— 脚本会把
  解析结果打印出来，出现 label_N 就说明缺标注。
* 内存：全精度 512×512×318 单例峰值约 0.9~1.0 GB，容器上限 1536 MB，
  务必串行执行，不要在别的病例处理中跑。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

APP_DIR = "/app"
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import lung3d_api as api  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="病例目录，如 /data/cases/20261019_...")
    ap.add_argument("--input", required=True, help="原始分割 nii.gz 路径")
    ap.add_argument("--labels", default=None, help="labels.json（缺省读 NIfTI 头扩展）")
    ap.add_argument("--keep-backup", action="store_true", help="保留 report.json 备份")
    a = ap.parse_args()

    case_dir = Path(a.case)
    src = Path(a.input)
    if not case_dir.is_dir():
        print("病例目录不存在：%s" % case_dir)
        return 1
    if not src.exists():
        print("输入文件不存在：%s" % src)
        return 1

    print("引擎 = %s" % api.mesh_engine())
    print("参数：MESH_DS_FACTOR=%s  DISTAL_PRUNE_MM=%s  MESH_MIN_VESSEL_D_MM=%s"
          % (os.environ.get("MESH_DS_FACTOR", "(默认)"),
             os.environ.get("DISTAL_PRUNE_MM", "(默认)"),
             os.environ.get("MESH_MIN_VESSEL_D_MM", "(默认)")))

    # 1) 记下原有元信息（case_id / name / created 等）
    rp = case_dir / "report.json"
    old = {}
    if rp.exists():
        try:
            old = json.loads(rp.read_text(encoding="utf-8"))
        except Exception as e:
            print("读取原 report.json 失败（将重建）：%s" % e)
        if a.keep_backup:
            bak = rp.with_suffix(".json.bak")
            shutil.copy2(rp, bak)
            print("已备份 report.json -> %s" % bak.name)

    # 2) 解析标注：labels.json 优先，其次 NIfTI 头扩展
    cfg = None
    if a.labels:
        cfg = api.parse_label_config(a.labels)
    if not cfg:
        cfg = api.load_label_config(src.parent, case_dir)
    if cfg and cfg.get("by_key"):
        print("标注：%s" % ", ".join("%s(%s)" % (k, v.get("role") or "-")
                                   for k, v in cfg["by_key"].items()))
    else:
        print("!! 没解析到标注，结构名会退化成 label_N，血管剪枝不会执行")

    # 3) 拆掩膜并重建
    masks = api.masks_from_nifti(str(src), "combined.nii.gz", cfg)
    if not masks:
        print("没有解析出任何结构")
        return 1
    print("结构：%s" % ", ".join(masks.keys()))

    report = {"case": case_dir.name}
    info = api.build_case(case_dir, masks, None, None, report, cfg)

    # 4) 还原元信息并落盘
    info["case_id"] = old.get("case_id") or case_dir.name
    if old.get("name"):
        info["name"] = old["name"]
    api.write_json_atomic(rp, info)

    print("-" * 60)
    for st in info.get("structures", []):
        g = case_dir / "meshes" / ("%s.glb" % st["key"])
        size = g.stat().st_size if g.exists() else 0
        print("%-18s %10.2f MB  volume=%.1f cm³" % (st["key"], size / 1e6,
                                                    st.get("volume_cm3", 0)))
    total = sum((case_dir / "meshes" / ("%s.glb" % s["key"])).stat().st_size
                for s in info.get("structures", [])
                if (case_dir / "meshes" / ("%s.glb" % s["key"])).exists())
    print("GLB 合计 %.2f MB" % (total / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
