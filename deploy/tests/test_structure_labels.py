# -*- coding: utf-8 -*-
"""结构命名外部化改造的回归测试。

验证 lung3d_api.py 在各输入形态下生成的结构名 / 键名 / 配色是否符合预期：
  A 多标签体数据，无 labels.json        -> 中性命名「结构 N」+ 自动调色板
  B 多标签体数据 + labels.json          -> 恢复配置中的名称与配色
  C labels.json 只声明部分标签          -> 其余回退为「结构 N」
  D 二值体数据                          -> 取文件名作为结构名
  E 数据目录 zip（含 labels.json）      -> 恢复配置中的名称
  F 桌面工具 write_labels_json 输出     -> 可被后端解析
  附加：空配置 / 坏 json / 名称回退的健壮性

用法（项目根目录下）：
  ./.venv/Scripts/python.exe deploy/tests/test_structure_labels.py

注意：测试使用独立的临时输出目录，不会改动 web_output。
"""
import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import nibabel as nib

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

import lung3d_api as api  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="lung3d_test_"))
api.OUTPUT_DIR = TMP

MED_LABELS = {
    "schema": "lung3d.labels/1",
    "structures": [
        {"label": 1, "key": "lung_arteries", "name": "肺动脉", "color": [1.0, 0.2, 0.2], "role": "vessel"},
        {"label": 2, "key": "lung_veins", "name": "肺静脉", "color": [0.2, 0.4, 1.0], "role": "vessel"},
        {"label": 3, "key": "lung_airways", "name": "气管支气管", "color": [0.2, 1.0, 0.3], "role": "airway"},
        {"label": 4, "key": "lung_airways_wall", "name": "气道壁", "color": [0.9, 0.6, 0.2], "role": "wall"},
        {"label": 5, "key": "lung_nodules", "name": "肺结节", "color": [1.0, 1.0, 0.1], "role": "lesion"},
    ],
}

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  |  " + detail) if detail else ""))


def make_volume(path, n_labels=5):
    """生成 n_labels 个分离球形标签的合成体数据（1mm 各向同性）。"""
    shape = (48, 48, 48)
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float64)
    data = np.zeros(shape, dtype=np.uint8)
    centers = [(12, 12, 12), (12, 12, 34), (12, 34, 12), (12, 34, 34), (34, 34, 34)]
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    for i, (cz, cy, cx) in enumerate(centers[:n_labels], start=1):
        m = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= 36
        data[m] = i
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def run_case(case_id, files):
    """在临时输出目录下构造 case_dir/upload 并跑真实处理流程。"""
    case_dir = TMP / case_id
    (case_dir / "upload").mkdir(parents=True, exist_ok=True)
    for fn, producer in files.items():
        producer(case_dir / "upload" / fn)
    api._process_upload_worker(case_dir, None)
    err = case_dir / "error.txt"
    if err.exists():
        raise RuntimeError("worker error: " + err.read_text(encoding="utf-8"))
    return json.loads((case_dir / "report.json").read_text(encoding="utf-8"))


def _vol(p):
    make_volume(p, 5)


def _labels(p):
    p.write_text(json.dumps(MED_LABELS, ensure_ascii=False, indent=2), encoding="utf-8")


print("=" * 74)
print("用例 A —— 多标签体数据，无 labels.json  -> 期望中性命名 结构 1..5")
print("=" * 74)
info = run_case("A_no_labels", {"combined.nii.gz": _vol})
names = [s["name"] for s in info["structures"]]
keys = [s["key"] for s in info["structures"]]
check("键名为 label_N", keys == ["label_%d" % i for i in range(1, 6)], str(keys))
check("名称为 结构 N", names == ["结构 %d" % i for i in range(1, 6)], str(names))
check("配色来自自动调色板", all(len(s["color"]) == 3 for s in info["structures"]))
check("网格文件已生成",
      all((TMP / s["mesh"].split("/")[3] / "meshes" / (s["key"] + ".glb")).exists() for s in info["structures"]))

print()
print("=" * 74)
print("用例 B —— 多标签体数据 + labels.json  -> 期望恢复原有命名/配色")
print("=" * 74)
info = run_case("B_with_labels", {"combined.nii.gz": _vol, "labels.json": _labels})
names = [s["name"] for s in info["structures"]]
check("名称恢复为医学结构名", names == ["肺动脉", "肺静脉", "气管支气管", "气道壁", "肺结节"], str(names))
art = [s for s in info["structures"] if s["key"] == "lung_arteries"][0]
check("配色取自配置文件", [round(c, 2) for c in art["color"]] == [1.0, 0.2, 0.2], str(art["color"]))

print()
print("=" * 74)
print("用例 C —— labels.json 只声明前 2 个  -> 期望 3..5 回退为 结构 N")
print("=" * 74)
_partial = {"structures": MED_LABELS["structures"][:2]}


def _partial_file(p):
    p.write_text(json.dumps(_partial, ensure_ascii=False, indent=2), encoding="utf-8")


info = run_case("C_partial", {"combined.nii.gz": _vol, "labels.json": _partial_file})
names = [s["name"] for s in info["structures"]]
check("部分覆盖 + 部分回退", names == ["肺动脉", "肺静脉", "结构 3", "结构 4", "结构 5"], str(names))

print()
print("=" * 74)
print("用例 D —— 二值体数据，文件名作为结构名")
print("=" * 74)
info = run_case("D_binary", {"my_shape.nii.gz": lambda p: make_volume(p, 1)})
check("单结构取文件名", [s["name"] for s in info["structures"]] == ["my_shape"])

print()
print("=" * 74)
print("用例 E —— 数据目录 zip（模拟桌面工具输出）-> 期望恢复命名")
print("=" * 74)


def _make_zip(dest):
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_volume(d / "combined.nii.gz", 5)
        (d / "labels.json").write_text(json.dumps(MED_LABELS, ensure_ascii=False, indent=2), encoding="utf-8")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(d.iterdir()):
                z.write(f, f.name)
    dest.write_bytes(buf.getvalue())


info = run_case("E_zip", {"data.zip": _make_zip})
check("zip 内 labels.json 生效",
      [s["name"] for s in info["structures"]] == ["肺动脉", "肺静脉", "气管支气管", "气道壁", "肺结节"])

print()
print("=" * 74)
print("用例 F —— 桌面工具 write_labels_json 输出格式可被后端解析")
print("=" * 74)
import importlib.util  # noqa: E402

try:
    rc = importlib.import_module("lung3d_reconstruct")
    out = Path(tempfile.mkdtemp(prefix="rc_"))
    p = rc.write_labels_json(out, {1: "lung_arteries", 2: "lung_veins", 3: "lung_airways",
                                   4: "lung_airways_wall", 5: "lung_nodules"})
    parsed = api.parse_label_config(p)
    check("write_labels_json 可被后端解析", parsed is not None and len(parsed["by_label"]) == 5)
    ok_names = [parsed["by_label"][i]["name"] for i in range(1, 6)]
    check("名称与角色正确",
          ok_names == ["肺动脉", "肺静脉", "气管支气管", "气道壁", "肺结节"]
          and parsed["by_label"][1]["role"] == "vessel"
          and parsed["by_label"][4]["role"] == "wall", str(ok_names))
except Exception as e:
    check("write_labels_json 可被后端解析", False, "%s: %s" % (type(e).__name__, e))

print()
print("=" * 74)
print("附加：空配置 / 坏配置的健壮性")
print("=" * 74)
check("空目录 -> 空配置", api.load_label_config(Path(tempfile.mkdtemp()))["by_label"] == {})
_bad = Path(tempfile.mkdtemp()) / "labels.json"
_bad.write_text("{ this is not json", encoding="utf-8")
check("坏 json -> 不抛异常", isinstance(api.load_label_config(_bad.parent), dict))
check("default_structure_name(label_7)", api.default_structure_name("label_7") == "结构 7")
check("default_structure_name(xyz)", api.default_structure_name("xyz") == "xyz")

passed = sum(1 for _, ok, _ in RESULTS if ok)
print()
print("=" * 74)
print("结果：%d/%d 通过" % (passed, len(RESULTS)))
for n, ok, d in RESULTS:
    if not ok:
        print("  未通过:", n, d)
print("=" * 74)

shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if passed == len(RESULTS) else 1)
