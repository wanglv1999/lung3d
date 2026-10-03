#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lung3d_pack_data.py 回归测试

覆盖：
  1. 多标签体数据 -> 规范目录三件套（combined.nii.gz / labels.json / report.json）
  2. labels.json 内容（label/key/name/role/color）与内置结构表一致
  3. --generic 模式只写 key，不写显示名与配色
  4. 源目录已有 labels.json 时原样复用（不覆盖为内置表）
  5. --zip 产出的压缩包结构正确（根目录即数据文件，不含外层目录名）
  6. 未知标签值退化为 label_<n>（查看端显示「结构 n」）

运行（需项目 .venv，含 numpy/nibabel）：
    ./.venv/Scripts/python.exe deploy/tests/test_pack_data.py
"""
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lung3d_pack_data as pack  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  <- " + detail) if detail and not cond else ""))


def make_multilabel(path, shape=(24, 24, 16), labels=(1, 2, 3, 4, 5)):
    """构造带 5 个标签的合成体数据，仿射用 0.7mm 各向同性。"""
    data = np.zeros(shape, dtype=np.uint8)
    for i, lab in enumerate(labels):
        sl = tuple(slice(2 + i, 5 + i) for _ in range(3))
        data[sl] = lab
    affine = np.diag([0.707, 0.707, 1.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), str(path))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="lung3d_pack_test_"))
    out = tmp / "out"

    print("[1] 多标签体数据 -> 规范目录")
    src_nii = tmp / "combined.nii.gz"
    make_multilabel(src_nii)
    dst, zp, lmap = pack.pack_one(src_nii, out, "合成数据", want_zip=True)
    check("产出 combined.nii.gz", (dst / "combined.nii.gz").is_file())
    check("产出 labels.json", (dst / "labels.json").is_file())
    check("产出 report.json", (dst / "report.json").is_file())
    check("产出 zip", zp is not None and zp.is_file())
    check("5 个标签全部映射", len(lmap) == 5, str(lmap))

    print("[2] labels.json 内容")
    payload = json.loads((dst / "labels.json").read_text(encoding="utf-8"))
    items = {it["label"]: it for it in payload["structures"]}
    check("schema 标记正确", payload.get("schema") == "lung3d.labels/1")
    check("标签 1 = 肺动脉 + vessel", items[1]["key"] == "lung_arteries" and items[1]["role"] == "vessel")
    check("标签 5 = 肺结节 + lesion", items[5]["key"] == "lung_nodules" and items[5]["role"] == "lesion")
    check("配色为 3 元素数组", all(isinstance(it.get("color"), list) and len(it["color"]) == 3 for it in payload["structures"]))
    check("气道壁 role=wall", items[4]["role"] == "wall")

    print("[3] report.json 体积")
    rep = json.loads((dst / "report.json").read_text(encoding="utf-8"))
    check("case 名 = 传入名", rep["case"] == "合成数据")
    check("含 volumes_cm3", set(rep["volumes_cm3"]) == {"lung_arteries", "lung_veins", "lung_airways", "lung_airways_wall", "lung_nodules"})
    check("体积为正数", all(v > 0 for v in rep["volumes_cm3"].values()))

    print("[4] --generic 模式")
    dst_g, _, _ = pack.pack_one(src_nii, out, "中性数据", generic=True)
    pg = json.loads((dst_g / "labels.json").read_text(encoding="utf-8"))
    check("generic 不含 name", all("name" not in it for it in pg["structures"]))
    check("generic 不含 color", all("color" not in it for it in pg["structures"]))
    check("generic 保留 key", pg["structures"][0]["key"] == "lung_arteries")

    print("[5] 源目录已有 labels.json -> 复用")
    src_dir = tmp / "src_case"
    src_dir.mkdir()
    make_multilabel(src_dir / "combined.nii.gz")
    custom = {"schema": "lung3d.labels/1", "structures": [
        {"label": 1, "key": "bone", "name": "骨骼", "color": [0.5, 0.5, 0.5]},
        {"label": 2, "key": "vessel_x", "name": "血管", "color": [1.0, 0.0, 0.0]},
    ]}
    (src_dir / "labels.json").write_text(json.dumps(custom, ensure_ascii=False), encoding="utf-8")
    dst_r, _, lmap_r = pack.pack_one(src_dir, out, "复用数据")
    check("沿用自定义 key", lmap_r.get(1) == "bone" and lmap_r.get(2) == "vessel_x", str(lmap_r))
    pr = json.loads((dst_r / "labels.json").read_text(encoding="utf-8"))
    names = {it["label"]: it.get("name") for it in pr["structures"]}
    check("沿用自定义显示名", names.get(1) != "肺动脉", str(names))

    print("[6] 未知标签退化")
    src_u = tmp / "unknown.nii.gz"
    make_multilabel(src_u, labels=(1, 7))
    _, _, lmap_u = pack.pack_one(src_u, out, "未知标签")
    check("标签 7 -> label_7", lmap_u.get(7) == "label_7", str(lmap_u))

    print("[7] zip 结构")
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
    check("zip 内为扁平结构", all(not n.startswith("合成数据/") for n in names), str(names[:4]))
    check("zip 含三件套", {"combined.nii.gz", "labels.json", "report.json"} <= set(names))

    print()
    print(f"结果: {len(PASS)} 通过, {len(FAIL)} 失败")
    if FAIL:
        print("失败项: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
