#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重新着色：把已处理数据的结构配色更新为当前结构表定义。

背景
----
颜色只存在于两处：

1. 数据目录的 `labels.json`（随数据上传时带入）
2. 处理后生成的 `report.json` 的 `structures[].color`（查看端实际读取）

而 GLB 网格是纯几何导出（`trimesh.export(file_type="glb")`），**不烘焙顶点色**，
所以改配色**无需重算网格**，直接改 `report.json` / `labels.json` 即可生效。

注意：`report.json` 生成时就把颜色写死了，因此后来修改 `STRUCTURE_LABELS`
**不会回溯**到已处理的数据 —— 本脚本就是用来补这一步的。

用法
----
    # 预览（不写入）
    python lung3d_recolor.py --dir web_output/cases --dry-run

    # 实际写入（自动备份 .bak）
    python lung3d_recolor.py --dir web_output/cases

    # 只处理某一个 case
    python lung3d_recolor.py --dir web_output/cases/20260929_211037_6de160

    # 自定义某些结构的颜色（覆盖结构表）
    python lung3d_recolor.py --dir web_output/cases \
        --set lung_veins=1.0,0.2,0.2 --set lung_arteries=0.2,0.4,1.0

    # 按标签序号指定（结构表匹配不到 key 时用；序号从 1 起）
    python lung3d_recolor.py --dir web_output/cases --by-index 1=0.2,0.4,1.0

配色约定（RGB 各分量 0~1）
-------------------------
    肺动脉 蓝 / 肺静脉 红 / 气管 绿 / 肺结节 黄
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# 结构表：优先用桌面重建工具里的权威定义；取不到时用内置副本兜底。
try:
    from lung3d_reconstruct import STRUCTURE_LABELS  # type: ignore
except Exception:  # pragma: no cover
    STRUCTURE_LABELS = {
        "lung_arteries":     ("肺动脉", [0.20, 0.40, 1.00], "vessel"),
        "lung_veins":        ("肺静脉", [0.85, 0.17, 0.17], "vessel"),
        "lung_airways":      ("气管",   [0.20, 1.00, 0.30], "airway"),
        "lung_airways_wall": ("气道壁", [0.90, 0.60, 0.20], "wall"),
        "lung_nodules":      ("肺结节", [1.00, 1.00, 0.10], "lesion"),
    }


def parse_color(text):
    """解析 'r,g,b' 或 'r g b'，返回 [float, float, float]。"""
    parts = text.replace(";", ",").replace(" ", ",").split(",")
    parts = [p for p in parts if p != ""]
    if len(parts) != 3:
        raise ValueError("颜色需为 3 个分量，如 1.0,0.2,0.2；收到: %r" % text)
    vals = [float(p) for p in parts]
    return vals


def color_of(key):
    """按结构 key 取配色；未登记返回 None。"""
    item = STRUCTURE_LABELS.get(key)
    if not item:
        return None
    _, color, _role = item
    return list(color) if color else None


def hex_of(rgb):
    return "#%02X%02X%02X" % tuple(max(0, min(255, round(c * 255))) for c in rgb)


def recolor_report(path, overrides, by_index, dry_run, backup):
    """更新单个 report.json 的 structures[].color。返回 (改动条数, 明细列表)。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return 0, [], "读取失败: %s" % e

    structs = data.get("structures")
    if not isinstance(structs, list):
        return 0, [], "无 structures 字段"

    changes = []
    for i, st in enumerate(structs, start=1):
        if not isinstance(st, dict):
            continue
        key = st.get("key")
        old = st.get("color")

        new = None
        src = ""
        if key in overrides:
            new, src = overrides[key], "--set"
        elif str(i) in by_index:
            new, src = by_index[str(i)], "--by-index"
        elif key:
            c = color_of(key)
            if c:
                new, src = c, "结构表"

        if new is None:
            continue

        same = (
            isinstance(old, (list, tuple))
            and len(old) == 3
            and all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(old, new))
        )
        if same:
            continue

        changes.append((i, key, old, new, src))
        if not dry_run:
            st["color"] = [float(c) for c in new]

    if changes and not dry_run:
        if backup:
            bak = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, bak)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(changes), changes, None


def recolor_labels(path, overrides, by_index, dry_run, backup):
    """同步更新 labels.json（若存在），让后续重新上传也带新配色。"""
    if not path.exists():
        return 0, [], None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return 0, [], "读取失败: %s" % e

    items = data.get("structures")
    if not isinstance(items, list):
        return 0, [], None

    changes = []
    for it in items:
        if not isinstance(it, dict):
            continue
        key = it.get("key")
        old = it.get("color")
        new = None
        if key in overrides:
            new = overrides[key]
        elif key:
            new = color_of(key)
        if new is None:
            continue
        if (
            isinstance(old, (list, tuple))
            and len(old) == 3
            and all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(old, new))
        ):
            continue
        changes.append((it.get("label"), key, old, new, "labels.json"))
        if not dry_run:
            it["color"] = [float(c) for c in new]

    if changes and not dry_run:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(changes), changes, None


def iter_case_dirs(root):
    """root 本身是 case 目录，或包含若干 case 目录。"""
    root = Path(root)
    if not root.exists():
        return []
    if (root / "report.json").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "report.json").exists())


def main(argv=None):
    ap = argparse.ArgumentParser(description="重新着色：更新已处理数据的结构配色（无需重算网格）")
    ap.add_argument("--dir", default="web_output/cases", help="case 根目录或单个 case 目录")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=R,G,B", help="按结构 key 覆盖配色，可重复")
    ap.add_argument("--by-index", action="append", default=[],
                    metavar="N=R,G,B", help="按结构序号覆盖配色（1 起），可重复")
    ap.add_argument("--dry-run", action="store_true", help="只预览不写入")
    ap.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份")
    args = ap.parse_args(argv)

    overrides, by_index = {}, {}
    for s in args.set:
        if "=" not in s:
            ap.error("--set 需形如 KEY=R,G.B，收到: %r" % s)
        k, v = s.split("=", 1)
        overrides[k.strip()] = parse_color(v)
    for s in args.by_index:
        if "=" not in s:
            ap.error("--by-index 需形如 N=R,G,B，收到: %r" % s)
        k, v = s.split("=", 1)
        by_index[k.strip()] = parse_color(v)

    cases = iter_case_dirs(args.dir)
    if not cases:
        print("[错误] 未找到含 report.json 的目录：%s" % args.dir)
        return 1

    print("=" * 72)
    print("重新着色%s  目录: %s" % ("（预览，不写入）" if args.dry_run else "", args.dir))
    print("结构表配色：")
    for k, v in STRUCTURE_LABELS.items():
        print("    %-20s %-6s %s" % (k, v[0], hex_of(v[1])))
    if overrides or by_index:
        print("覆盖：%s %s" % (overrides or "-", by_index or "-"))
    print("=" * 72)

    total = 0
    for d in cases:
        n, ch, err = recolor_report(d / "report.json", overrides, by_index,
                                    args.dry_run, not args.no_backup)
        nl, chl, errl = recolor_labels(d / "labels.json", overrides, by_index,
                                       args.dry_run, not args.no_backup)
        print("\n[case] %s" % d.name)
        if err:
            print("    %s" % err)
            continue
        if not ch and not chl:
            print("    配色已是最新，无需改动")
            continue
        for i, key, old, new, src in ch:
            print("    #%d %-20s %s -> %s   (%s)" % (
                i, key, hex_of(old) if old else "-", hex_of(new), src))
        for lab, key, old, new, src in chl:
            print("    label %-4s %-16s %s -> %s   (%s)" % (
                lab, key, hex_of(old) if old else "-", hex_of(new), src))
        total += n + nl

    print("\n" + "=" * 72)
    print("合计 %d 处改动%s" % (total, "（预览）" if args.dry_run else "（已写入）"))
    if not args.dry_run and total:
        print("备份：同目录 *.bak")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
