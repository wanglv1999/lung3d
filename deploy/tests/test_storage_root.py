# -*- coding: utf-8 -*-
"""数据持久化改造的回归测试。

验证 lung3d_api.py 的存储层改造：
  A 默认值            -> 未设 CASE_ROOT 时仍是 web_output/cases（本地盘，兼容旧行为）
  B 路径解析          -> CASE_ROOT 绝对/相对路径均能正确解析，后端标记为 external
  C 进程级生效        -> 新进程带 CASE_ROOT 启动时，模块级 OUTPUT_DIR 真的指过去
  D 保留期开关        -> CASE_EXPIRE_DAYS 缺省 7、0 表示不删、非法值回退 7
  E 原子写            -> report.json 写完无 .tmp 残留，内容可读
  F 健康检查          -> /api/health 带 storage 段（backend/path/writable/free_gb）
  G 探测缓存          -> storage_info 有 TTL 缓存，refresh=True 才重新探测
  H 自动清理          -> days=0 不删；按 mtime 删过期病例；孤儿暂存目录被清掉
  I 暂存与产物分离    -> 上传落暂存目录、产物落病例目录，处理完暂存自动删除
  J 旧调用兼容        -> 不传 upload_dir 时仍读 case_dir/upload（历史数据/旧测试）
  K 真实上传链路      -> 走 /api/process 的写盘逻辑，上传文件不进病例目录

用法（项目根目录下）：
  ./.venv/Scripts/python.exe deploy/tests/test_storage_root.py

注意：全部在临时目录内进行，不会改动 web_output 与线上数据。
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import nibabel as nib

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

import lung3d_api as api  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="lung3d_storage_test_"))
CASES = TMP / "cases"
STAGE = TMP / "stage"
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  |  " + detail) if detail else ""))


def make_volume(path, n_labels=2):
    """生成 n_labels 个分离球形标签的合成体数据（1mm 各向同性）。"""
    shape = (40, 40, 40)
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float64)
    data = np.zeros(shape, dtype=np.uint8)
    centers = [(10, 10, 10), (10, 10, 28), (10, 28, 10)]
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    for i, (cz, cy, cx) in enumerate(centers[:n_labels], start=1):
        m = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= 25
        data[m] = i
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def age(path, days):
    """把文件/目录的 mtime 往前推 days 天（模拟历史数据）。"""
    t = time.time() - days * 86400
    os.utime(str(path), (t, t))


print("=" * 74)
print("用例 A —— 未设 CASE_ROOT：保持旧行为（本地盘）")
print("=" * 74)
os.environ.pop(api.CASE_ROOT_ENV, None)
os.environ.pop(api.UPLOAD_STAGING_ENV, None)
check("后端标记 local", api.storage_backend() == "local", api.storage_backend())
check("默认病例目录 = <项目>/web_output/cases",
      api.OUTPUT_DIR == api.DEFAULT_CASE_ROOT, str(api.OUTPUT_DIR))
check("默认暂存目录在系统临时盘",
      api.UPLOAD_STAGING == Path(tempfile.gettempdir()) / "lung3d_uploads", str(api.UPLOAD_STAGING))

print()
print("=" * 74)
print("用例 B —— CASE_ROOT 路径解析（绝对 / 相对）")
print("=" * 74)
abs_root = CASES / "abs"
os.environ[api.CASE_ROOT_ENV] = str(abs_root)
check("绝对路径原样解析",
      api._resolve_dir(api.CASE_ROOT_ENV, api.DEFAULT_CASE_ROOT) == abs_root,
      str(api._resolve_dir(api.CASE_ROOT_ENV, api.DEFAULT_CASE_ROOT)))
check("设置后后端标记 external", api.storage_backend() == "external", api.storage_backend())

os.environ[api.CASE_ROOT_ENV] = "rel_cases"
rel = api._resolve_dir(api.CASE_ROOT_ENV, api.DEFAULT_CASE_ROOT)
check("相对路径按项目根解析", rel == (PROJ / "rel_cases").resolve(), str(rel))

os.environ[api.CASE_ROOT_ENV] = "   "
check("空白值视为未设置（回退 local）", api.storage_backend() == "local", api.storage_backend())
os.environ.pop(api.CASE_ROOT_ENV, None)

print()
print("=" * 74)
print("用例 C —— 新进程带 CASE_ROOT 启动时模块级路径生效")
print("=" * 74)
sub_root = CASES / "sub"
env = os.environ.copy()
env[api.CASE_ROOT_ENV] = str(sub_root)
env[api.UPLOAD_STAGING_ENV] = str(STAGE / "sub")
code = "import lung3d_api as a;print(a.OUTPUT_DIR);print(a.UPLOAD_STAGING);print(a.storage_backend())"
r = subprocess.run([sys.executable, "-c", code], cwd=str(PROJ), env=env,
                   capture_output=True, text=True, timeout=120)
lines = [x.strip() for x in r.stdout.strip().splitlines()]
check("子进程 OUTPUT_DIR 指向 CASE_ROOT",
      len(lines) >= 2 and Path(lines[0]) == sub_root, str(lines))
check("子进程暂存目录指向 UPLOAD_STAGING_DIR",
      len(lines) >= 2 and Path(lines[1]) == STAGE / "sub", str(lines))
check("子进程后端标记 external", len(lines) >= 3 and lines[2] == "external", str(lines))

print()
print("=" * 74)
print("用例 D —— CASE_EXPIRE_DAYS 解析")
print("=" * 74)
os.environ.pop("CASE_EXPIRE_DAYS", None)
check("缺省 7 天", api.case_expire_days() == 7, str(api.case_expire_days()))
os.environ["CASE_EXPIRE_DAYS"] = "0"
check("0 表示关闭自动清理", api.case_expire_days() == 0, str(api.case_expire_days()))
os.environ["CASE_EXPIRE_DAYS"] = "30"
check("可设为 30 天", api.case_expire_days() == 30, str(api.case_expire_days()))
os.environ["CASE_EXPIRE_DAYS"] = "abc"
check("非法值回退 7 天", api.case_expire_days() == 7, str(api.case_expire_days()))
os.environ.pop("CASE_EXPIRE_DAYS", None)

print()
print("=" * 74)
print("用例 E —— report.json 原子写")
print("=" * 74)
api.OUTPUT_DIR = CASES
api.UPLOAD_STAGING = STAGE
CASES.mkdir(parents=True, exist_ok=True)
STAGE.mkdir(parents=True, exist_ok=True)
target = CASES / "atomic_case" / "report.json"
payload = {"case_id": "atomic_case", "name": "测试", "structures": [{"key": "s1"}]}
api.write_json_atomic(target, payload)
check("文件已写出", target.exists())
check("内容可解析且一致",
      json.loads(target.read_text(encoding="utf-8")) == payload)
leftover = list(target.parent.glob("*.tmp*"))
check("无 .tmp 残留", leftover == [], str(leftover))

print()
print("=" * 74)
print("用例 F —— /api/health 暴露存储状态")
print("=" * 74)
ok, msg = api.ensure_storage()
check("启动自检通过", ok, msg)
h = api.health()
check("health.status = ok", h.get("status") == "ok", str(h.get("status")))
st = h.get("storage") or {}
for field in ("backend", "path", "staging", "writable", "free_gb", "total_gb", "expire_days"):
    check("storage 含字段 %s" % field, field in st, str(st.get(field)))
check("writable = True", st.get("writable") is True, str(st.get("writable")))
check("path 指向测试目录", st.get("path") == str(CASES), str(st.get("path")))
check("free_gb 为数值", isinstance(st.get("free_gb"), (int, float)), str(st.get("free_gb")))

print()
print("=" * 74)
print("用例 G —— storage_info 的 TTL 缓存")
print("=" * 74)
i1 = api.storage_info(refresh=True)
i2 = api.storage_info()
check("缓存命中返回同一对象", i1 is i2)
i3 = api.storage_info(refresh=True)
check("refresh=True 重新探测", i3 is not i1)

print()
print("=" * 74)
print("用例 H —— 自动清理策略")
print("=" * 74)
fresh = CASES / "fresh_case"
old = CASES / "old_case"
for d in (fresh, old):
    d.mkdir(parents=True, exist_ok=True)
    api.write_json_atomic(d / "report.json", {"case_id": d.name})
age(old, 40)
age(old / "report.json", 40)  # 清理按 report.json 的 mtime 判定，两者一起推老
check("days=0 时一个都不删", api.cleanup_old_cases(0) == 0)
check("days=0 后旧病例仍在", old.exists())
removed = api.cleanup_old_cases(30)
check("days=30 删掉过期病例", removed == 1 and not old.exists(), "removed=%s" % removed)
check("未过期病例保留", fresh.exists())

stale_stage = STAGE / "stale_upload"
stale_stage.mkdir(parents=True, exist_ok=True)
age(stale_stage, 3)
new_stage = STAGE / "new_upload"
new_stage.mkdir(parents=True, exist_ok=True)
check("孤儿暂存被清掉", api.cleanup_stale_uploads(1) == 1 and not stale_stage.exists())
check("新暂存保留", new_stage.exists())

print()
print("=" * 74)
print("用例 I —— 暂存与产物分离（worker 直调）")
print("=" * 74)
case_dir = CASES / "stage_case"
case_dir.mkdir(parents=True, exist_ok=True)
upload_dir = STAGE / "stage_case"
upload_dir.mkdir(parents=True, exist_ok=True)
make_volume(upload_dir / "combined.nii.gz", 2)
api._process_upload_worker(case_dir, "暂存测试", upload_dir)
err = case_dir / "error.txt"
check("处理无错误", not err.exists(), err.read_text(encoding="utf-8") if err.exists() else "")
check("产物 report.json 已生成", (case_dir / "report.json").exists())
info = json.loads((case_dir / "report.json").read_text(encoding="utf-8"))
check("名称取上传时给的名字", info.get("name") == "暂存测试", str(info.get("name")))
check("结构已提取", len(info.get("structures") or []) == 2, str(len(info.get("structures") or [])))
check("暂存目录处理完被删除", not upload_dir.exists())
check("病例目录内不再残留 upload/", not (case_dir / "upload").exists())
check("网格文件落在病例目录", any((case_dir / "meshes").glob("*.glb")))

print()
print("=" * 74)
print("用例 J —— 旧调用兼容（不传 upload_dir，读 case_dir/upload）")
print("=" * 74)
legacy = CASES / "legacy_case"
(legacy / "upload").mkdir(parents=True, exist_ok=True)
make_volume(legacy / "upload" / "combined.nii.gz", 2)
api._process_upload_worker(legacy, None)
check("旧调用仍能处理", (legacy / "report.json").exists())
check("旧调用不误删 upload/", (legacy / "upload").exists())

print()
print("=" * 74)
print("用例 K —— 真实上传链路：文件落暂存、不进病例目录")
print("=" * 74)


class FakeUpload:
    """模拟 fastapi UploadFile 的最小接口：async read(n)。"""

    def __init__(self, data, filename):
        self._buf = data
        self.filename = filename
        self._pos = 0

    async def read(self, n=-1):
        if n is None or n < 0:
            chunk = self._buf[self._pos:]
            self._pos = len(self._buf)
            return chunk
        chunk = self._buf[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk


src = make_volume(TMP / "up_src.nii.gz", 2)
resp = asyncio.run(api.process_files(files=[FakeUpload(src.read_bytes(), "combined.nii.gz")],
                                     name="上传链路测试", filename=None))
case_id = resp.get("case_id")
check("接口返回 processing", resp.get("status") == "processing", str(resp))
check("上传文件落暂存目录", (STAGE / case_id / "combined.nii.gz").exists(), case_id)
check("病例目录无 upload/", not (CASES / case_id / "upload").exists())

deadline = time.time() + 120
while time.time() < deadline and not (CASES / case_id / "report.json").exists():
    time.sleep(0.5)
check("后台处理完成", (CASES / case_id / "report.json").exists(), case_id)
check("处理完暂存已删除", not (STAGE / case_id).exists())
info = json.loads((CASES / case_id / "report.json").read_text(encoding="utf-8"))
check("结构数正确", len(info.get("structures") or []) == 2, str(len(info.get("structures") or [])))

print()
print("=" * 74)
total = len(RESULTS)
passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    if not ok:
        print("  FAILED:", name, detail)
print("结果: %d/%d 通过" % (passed, total))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if passed == total else 1)
