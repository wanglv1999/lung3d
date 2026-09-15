#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lung3D Web 后端服务（FastAPI）
===============================
上传已分割的 nii.gz（或病例目录 zip）→ 提取各结构网格(GLB) → 返回元数据给前端(网页版/小程序)

启动：
  ./.venv/Scripts/python.exe lung3d_api.py --host 0.0.0.0 --port 8000

接口：
  POST /api/process     上传文件(multipart: files)。支持:
                         - 单个多标签分割 nii.gz(自动按标签拆分)
                         - 单个二值 nii.gz(文件名作为结构名)
                         - 病例目录 zip(ct.nii.gz/combined.nii.gz/seg_totalseg/*/report.json)
                         - 多个 nii.gz 同时上传(各自为独立结构)
  GET  /api/case/{id}   读取某病例 report.json
  GET  /api/mesh/{case_id}/{name}.glb   下载网格
  GET  /api/cases       已处理病例列表
  GET  /api/wxacode     生成打开某病例3D页的微信小程序码(PNG)。需配置 wx_config.json(appid/secret)
  GET  /api/qrcode      生成指向网页版深链的普通二维码(PNG，调试用)

小程序码配置(wx_config.json，与 lung3d_api.py 同目录):
  {
    "appid": "wx...",            # 小程序 AppID
    "secret": "...",             # 小程序 AppSecret
    "env_version": "release"     # release/trial/develop
  }
"""
import argparse
import base64
import gc
import io
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "web_output" / "cases"

# 并发处理上限：同时只处理 MAX_CONCURRENT_PROCESSING 个病例，
# 其余排队等待，避免多任务同时占用内存导致容器 OOM。
MAX_CONCURRENT_PROCESSING = max(1, int(os.environ.get("MAX_CONCURRENT_PROCESSING", "1")))
_PROCESS_SEM = threading.BoundedSemaphore(MAX_CONCURRENT_PROCESSING)

STRUCTURES = [
    dict(key="lung_arteries", name="肺动脉", color=[1.00, 0.20, 0.20]),
    dict(key="lung_veins", name="肺静脉", color=[0.20, 0.40, 1.00]),
    dict(key="lung_airways", name="气管支气管", color=[0.20, 1.00, 0.30]),
    dict(key="lung_airways_wall", name="气道壁", color=[0.90, 0.60, 0.20]),
    dict(key="lung_nodules", name="肺结节", color=[1.00, 1.00, 0.10]),
]
LABEL_MAP = {
    1: "lung_arteries",
    2: "lung_veins",
    3: "lung_airways",
    4: "lung_airways_wall",
    5: "lung_nodules",
}
PALETTE = [
    [1.0, 0.5, 0.2], [0.5, 0.2, 1.0], [0.2, 0.9, 0.9],
    [0.9, 0.4, 0.8], [0.6, 0.8, 0.2], [0.4, 0.6, 1.0],
]

app = FastAPI(title="Lung3D API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 网格提取（复用桌面版逻辑，输出 trimesh -> GLB）
# ---------------------------------------------------------------------------
def np_affine_parts(affine):
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    R = affine[:3, :3] / spacing[np.newaxis, :]
    origin = (affine @ np.array([0, 0, 0, 1.0]))[:3]
    return spacing, origin, R


def downsample_binary(mask, factor):
    if factor <= 1 or mask.ndim != 3:
        return mask
    from scipy import ndimage
    return (ndimage.zoom(mask.astype(np.uint8), 1.0 / factor, order=0) > 0.5)


def mask_to_trimesh(mask, affine, ds_factor=2, smooth=15, prune_mm=0.0):
    """二值掩膜 -> trimesh（RAS 世界坐标，与桌面查看器一致）。
    prune_mm: >0 时按管径阈值剪除细血管（在降采样后计算，避免内存爆槓）。"""
    import skimage.measure
    import trimesh

    arr = downsample_binary((mask > 0), ds_factor) if ds_factor > 1 else (mask > 0)
    if arr.sum() == 0:
        return None
    if prune_mm > 0:
        from scipy import ndimage
        spacing_ds = np.linalg.norm(affine[:3, :3], axis=0) * ds_factor
        d = ndimage.distance_transform_edt(arr, sampling=spacing_ds)
        arr = (d >= prune_mm)
        if arr.sum() == 0:
            return None
    verts, faces, _, _ = skimage.measure.marching_cubes(arr, level=0.5)
    spacing, origin, R = np_affine_parts(affine)
    world = verts @ (R @ np.diag(spacing * ds_factor)).T + origin
    mesh = trimesh.Trimesh(vertices=world.astype(np.float32), faces=faces.astype(np.int64))
    mesh.remove_unreferenced_vertices()
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
    try:
        mesh.update_faces(mesh.unique_faces)
    except Exception:
        pass
    if smooth > 0 and len(mesh.faces) > 4:
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.5, iterations=smooth)
        except Exception:
            pass
    try:
        _ = mesh.vertex_normals  # 触发法线计算
    except Exception:
        pass
    return mesh


def prune_thin_vessels(mask, affine, prune_mm=1.5, ds_factor=1):
    """按管径阈值剪除远端细血管：移除到边缘距离 < prune_mm(mm) 的体素，保留主干。
    ds_factor: 堵膜已降采样的倍数（用于校准 sampling）。"""
    if prune_mm <= 0:
        return mask
    from scipy import ndimage
    spacing = np.linalg.norm(affine[:3, :3], axis=0) * ds_factor
    d = ndimage.distance_transform_edt((mask > 0), sampling=spacing)
    return (d >= prune_mm)


def ct_surface_trimesh(ct_data, affine, ds_factor=4, threshold=-400.0):
    """CT 等值面（胸腔轮廓，可选背景）。"""
    import skimage.measure
    import trimesh

    from scipy import ndimage
    if ds_factor > 1:
        ct_data = ndimage.zoom(ct_data.astype(np.float32), 1.0 / ds_factor, order=1)
    arr = (ct_data > threshold).astype(np.uint8)
    if arr.sum() == 0:
        return None
    try:
        verts, faces, _, _ = skimage.measure.marching_cubes(arr, level=0.5)
    except Exception:
        return None
    spacing, origin, R = np_affine_parts(affine)
    world = verts @ (R @ np.diag(spacing * ds_factor)).T + origin
    mesh = trimesh.Trimesh(vertices=world.astype(np.float32), faces=faces.astype(np.int64))
    mesh.remove_unreferenced_vertices()
    try:
        _ = mesh.vertex_normals
    except Exception:
        pass
    return mesh


def export_glb(mesh, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = mesh.export(file_type="glb")
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------
def load_nifti(path):
    import nibabel as nib
    return nib.load(str(path))


def nifti_is_segmentation(img):
    d = np.asarray(img.dataobj)
    if d.dtype.kind == "f" or d.dtype.kind == "i":
        if d.min() < -500:  # HU 数据更像 CT
            return False
    return True


def masks_from_nifti(path, base_name):
    """返回 {key: (bool_mask, affine)}。多标签按 LABEL_MAP/数值拆分，二值作为单结构。"""
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim != 3:
        return None
    labels = np.unique(data)
    labels = labels[labels > 0]
    out = {}
    if len(labels) > 1:
        for lab in labels:
            key = LABEL_MAP.get(int(lab), f"label_{int(lab)}")
            out[key] = ((data == lab), img.affine)
    else:
        key = base_name[:-7] if base_name.lower().endswith(".nii.gz") else Path(base_name).stem
        out[key] = ((data > 0), img.affine)
    return out


def load_ct(path):
    img = load_nifti(path)
    data = np.asarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.dtype.kind in ("u", "i") and data.min() >= 0 and data.max() <= 4095:
        data = data.astype(np.int16) - 1024
    return np.ascontiguousarray(data), img.affine


def load_report_json(case_dir):
    rp = Path(case_dir) / "report.json"
    if not rp.exists():
        return {}
    try:
        return json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_from_case_dir(case_dir):
    """从病例目录收集：masks / ct / report。返回 (masks, ct_or_None, ct_affine, report)。"""
    case_dir = Path(case_dir)
    masks = {}
    affine = None
    ct_data, ct_affine = None, None

    combined = case_dir / "combined.nii.gz"
    if combined.exists():
        res = masks_from_nifti(combined, "combined.nii.gz")
        if res:
            for k, (m, a) in res.items():
                masks[k] = (m, a)
                if affine is None:
                    affine = a

    seg_dir = case_dir / "seg_totalseg"
    if seg_dir.is_dir():
        for p in sorted(seg_dir.glob("*.nii.gz")):
            key = p.name[:-7]
            if key in masks:
                continue
            img = load_nifti(p)
            masks[key] = (np.asarray(img.dataobj) > 0.5, img.affine)
            if affine is None:
                affine = img.affine

    for p in sorted(case_dir.glob("*.nii.gz")):
        key = p.name[:-7]
        if p.name == "combined.nii.gz" or key in masks:
            continue
        img = load_nifti(p)
        d = np.asarray(img.dataobj)
        if nifti_is_segmentation(img):
            masks[key] = ((d > 0.5), img.affine)
            if affine is None:
                affine = img.affine
        else:
            ct_data, ct_affine = np.ascontiguousarray(d), img.affine

    ct = case_dir / "ct.nii.gz"
    if ct.exists() and ct_data is None:
        ct_data, ct_affine = load_ct(ct)

    report = load_report_json(case_dir)
    if affine is None and ct_affine is not None:
        affine = ct_affine
    return masks, ct_data, ct_affine, report


# ---------------------------------------------------------------------------
# 构建病例
# ---------------------------------------------------------------------------
def build_case(case_dir, masks, ct_data, ct_affine, report):
    case_dir = Path(case_dir)
    mesh_dir = case_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    affine = None
    structures = []
    palette_iter = iter(PALETTE)
    try:
        distal_prune_mm = float(os.environ.get("DISTAL_PRUNE_MM", "0.75"))
    except Exception:
        distal_prune_mm = 0.75
    # 网格降采样系数：数值越大内存/耗时越低（质量略降），容器内存紧张时调大
    try:
        ds_env = max(1, int(os.environ.get("MESH_DS_FACTOR", "2")))
    except Exception:
        ds_env = 2
    try:
        ds_wall = max(1, int(os.environ.get("WALL_DS_FACTOR", str(ds_env))))
    except Exception:
        ds_wall = ds_env
    for key, (m, a) in masks.items():
        if not m.any():
            continue
        if affine is None:
            affine = a
        st = next((s for s in STRUCTURES if s["key"] == key), None)
        name = st["name"] if st else key
        color = st["color"] if st else next(palette_iter)
        if key == "lung_airways_wall":
            mesh = mask_to_trimesh(m, a, ds_factor=ds_wall, smooth=0)
        else:
            prune_mm = distal_prune_mm if key in ("lung_arteries", "lung_veins") else 0.0
            mesh = mask_to_trimesh(m, a, ds_factor=ds_env, prune_mm=prune_mm)
        if mesh is None:
            continue
        fname = f"{key}.glb"
        export_glb(mesh, mesh_dir / fname)
        vox = abs(np.linalg.det(a)) / 1000.0  # cm^3
        structures.append({
            "key": key,
            "name": name,
            "color": color,
            "mesh": f"/api/mesh/{case_dir.name}/{fname}",
            "volume_cm3": round(float(m.sum() * vox), 3),
            "voxels": int(m.sum()),
        })
        gc.collect()

    case_info = {
        "case_id": case_dir.name,
        "name": report.get("case", case_dir.name),
        "structures": structures,
        "ct_mesh": None,
    }

    if ct_data is not None:
        m = ct_surface_trimesh(ct_data, ct_affine)
        if m is not None:
            fname = "ct_surface.glb"
            export_glb(m, mesh_dir / fname)
            case_info["ct_mesh"] = f"/api/mesh/{case_dir.name}/{fname}"

    if affine is not None:
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        case_info["spacing_mm"] = [round(float(x), 4) for x in spacing]
        case_info["dimensions"] = list(masks[list(masks.keys())[0]][0].shape) if masks else None

    (case_dir / "report.json").write_text(
        json.dumps(case_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_info


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
def _process_upload_worker(case_dir, case_name):
    """后台异步处理上传的病例（受并发信号量限制，避免同时占用内存）。"""
    with _PROCESS_SEM:
        try:
            tmp = case_dir / "upload"
            masks, ct_data, ct_affine, report = {}, None, None, {}
            for p in sorted(tmp.iterdir()):
                fname = p.name
                if fname.lower().endswith(".zip"):
                    with zipfile.ZipFile(p) as z:
                        z.extractall(tmp / "_zip")
                    masks, ct_data, ct_affine, report = extract_from_case_dir(tmp / "_zip")
                    break
                elif fname.lower().endswith((".nii", ".nii.gz")):
                    img = load_nifti(p)
                    if nifti_is_segmentation(img):
                        res = masks_from_nifti(p, fname)
                        if res:
                            masks.update(res)
                    else:
                        ct_data, ct_affine = load_ct(p)
            if not masks and ct_data is None:
                files_info = []
                for p in sorted(tmp.iterdir()):
                    try:
                        files_info.append("%s (%d bytes)" % (p.name, p.stat().st_size))
                    except Exception:
                        files_info.append(p.name)
                (case_dir / "error.txt").write_text(
                    "无法识别文件：需要分割 nii.gz、多标签 nii.gz 或病例目录 zip。收到: " + "; ".join(files_info),
                    encoding="utf-8")
                return
            case_info = build_case(case_dir, masks, ct_data, ct_affine, report)
            if not case_info["structures"] and case_info["ct_mesh"] is None:
                (case_dir / "error.txt").write_text("未提取到任何结构网格", encoding="utf-8")
                return
            if case_name and case_name.strip():
                case_info["name"] = case_name.strip()
                (case_dir / "report.json").write_text(
                    json.dumps(case_info, ensure_ascii=False, indent=2), encoding="utf-8")
            print("case processed:", case_dir.name)
        except Exception as e:
            try:
                (case_dir / "error.txt").write_text("处理失败: %s" % e, encoding="utf-8")
            except Exception:
                pass
            print("case processing error:", case_dir.name, e)
        finally:
            gc.collect()


@app.post("/api/process")
async def process_files(files: list[UploadFile] = File(...), name: str = Form(None), filename: str = Form(None)):
    if not files:
        return JSONResponse({"error": "未上传文件"}, status_code=400)

    case_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    case_dir = OUTPUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    tmp = case_dir / "upload"
    tmp.mkdir(exist_ok=True)

    use_filename = (filename or "").strip()
    for uf in files:
        fname = (use_filename or Path(uf.filename or "file").name)
        use_filename = ""
        fname = Path(fname).name
        save_path = tmp / fname
        # 异步分块读入，避免大文件同步读阻塞事件循环
        size = 0
        with save_path.open("wb") as f:
            while True:
                chunk = await uf.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)
        if size == 0:
            try:
                save_path.unlink(missing_ok=True)
            except Exception:
                pass

    threading.Thread(target=_process_upload_worker, args=(case_dir, name), daemon=True).start()
    return {"status": "processing", "case_id": case_id}


@app.get("/api/case/{case_id}")
def get_case(case_id: str):
    d = OUTPUT_DIR / case_id
    rp = d / "report.json"
    if not rp.exists():
        ep = d / "error.txt"
        if ep.exists():
            return JSONResponse({"error": ep.read_text(encoding="utf-8")}, status_code=400)
        return JSONResponse({"error": "病例不存在或处理中"}, status_code=404)
    return json.loads(rp.read_text(encoding="utf-8"))

@app.delete("/api/case/{case_id}")
def delete_case(case_id: str):
    d = OUTPUT_DIR / case_id
    rp = d / "report.json"
    if not rp.exists():
        return JSONResponse({"error": "病例不存在"}, status_code=404)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "case_id": case_id}

@app.post("/api/case/{case_id}/rename")
def rename_case(case_id: str, payload: dict):
    rp = OUTPUT_DIR / case_id / "report.json"
    if not rp.exists():
        return JSONResponse({"error": "病例不存在"}, status_code=404)
    new_name = (payload.get("name") or "").strip()
    if not new_name:
        return JSONResponse({"error": "名称不能为空"}, status_code=400)
    info = json.loads(rp.read_text(encoding="utf-8"))
    info["name"] = new_name
    rp.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "case_id": case_id, "name": new_name}


@app.get("/api/mesh/{case_id}/{name}")
def get_mesh(case_id: str, name: str):
    p = OUTPUT_DIR / case_id / "meshes" / name
    if not p.exists() or p.suffix.lower() not in (".glb",):
        return JSONResponse({"error": "网格不存在"}, status_code=404)
    return FileResponse(p, media_type="model/gltf-binary")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/cases")
def list_cases():
    out = []
    if OUTPUT_DIR.is_dir():
        for d in sorted(OUTPUT_DIR.iterdir()):
            rp = d / "report.json"
            if rp.exists():
                try:
                    info = json.loads(rp.read_text(encoding="utf-8"))
                    out.append({"case_id": d.name, "name": info.get("name", d.name),
                                "n": len(info.get("structures", [])),
                                "created": d.stat().st_mtime})
                except Exception:
                    continue
    return out


# ---------------------------------------------------------------------------
# 微信小程序码
# ---------------------------------------------------------------------------
WX_CONFIG_PATH = ROOT / "wx_config.json"
_wx_token = {"token": None, "expires": 0.0}


def _load_wx_config():
    cfg = {}
    if WX_CONFIG_PATH.exists():
        try:
            cfg = json.loads(WX_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    cfg.setdefault("appid", os.environ.get("WX_APPID", ""))
    cfg.setdefault("secret", os.environ.get("WX_SECRET", ""))
    cfg.setdefault("env_version", "release")
    return cfg


def _wx_access_token():
    cfg = _load_wx_config()
    if not cfg["appid"] or not cfg["secret"]:
        raise RuntimeError("未配置 wx_config.json(appid/secret)，无法生成小程序码")
    now = time.time()
    if _wx_token["token"] and _wx_token["expires"] > now + 60:
        return _wx_token["token"]
    url = ("https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential"
           "&appid=%s&secret=%s" % (urllib.parse.quote(cfg["appid"]), urllib.parse.quote(cfg["secret"])))
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    if "access_token" not in data:
        raise RuntimeError("获取 access_token 失败: %s" % data)
    _wx_token["token"] = data["access_token"]
    _wx_token["expires"] = now + float(data.get("expires_in", 7200))
    return data["access_token"]


def _case_exists(case_id):
    return (OUTPUT_DIR / case_id / "report.json").exists()


@app.get("/api/wxacode")
def wxacode(case: str, env: str = None, env_version: str = None):
    """生成打开该病例 3D 页的微信小程序码(PNG)。scene = c=<case_id>。"""
    if not _case_exists(case):
        return JSONResponse({"error": "病例不存在"}, status_code=404)
    cfg = _load_wx_config()
    ver = env_version or env or cfg.get("env_version", "release")
    if ver not in ("release", "trial", "develop"):
        ver = "release"
    try:
        token = _wx_access_token()
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    body = json.dumps({
        "scene": "c=" + case,
        "page": "pages/viewer/viewer",
        "check_path": False,
        "env_version": ver,
        "width": 430,
        "auto_color": False,
    }).encode("utf-8")
    url = "https://api.weixin.qq.com/wxa/getwxacodeunlimited?access_token=" + urllib.parse.quote(token)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        return JSONResponse({"error": "生成小程序码失败: %s" % e.read().decode("utf-8")}, status_code=502)
    if data[:1] == b"{":  # 微信错误返回 json
        try:
            err = json.loads(data.decode("utf-8"))
            return JSONResponse({"error": "生成小程序码失败: %s" % err}, status_code=502)
        except Exception:
            pass
    return Response(content=data, media_type="image/png")


@app.get("/api/qrcode")
def qrcode(case: str, base: str = ""):
    """生成指向网页版深链的普通二维码(PNG)。base 为网页版访问地址，如 http://192.168.1.10:8000。"""
    if not _case_exists(case):
        return JSONResponse({"error": "病例不存在"}, status_code=404)
    try:
        import segno
    except ImportError:
        return JSONResponse({"error": "缺少 segno 库: pip install segno"}, status_code=500)
    link = (base or "http://localhost:8000") + "/?case=" + urllib.parse.quote(case)
    qr = segno.make(link, error="h")
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=8, border=2)
    return Response(content=buf.getvalue(), media_type="image/png")

def cleanup_old_cases(days=7):
    """删除超过 days 天未更新的病例（存储自动清理）。"""
    cutoff = time.time() - days * 86400
    if not OUTPUT_DIR.is_dir():
        return 0
    removed = 0
    for d in OUTPUT_DIR.iterdir():
        try:
            rp = d / "report.json"
            mtime = rp.stat().st_mtime if rp.exists() else d.stat().st_mtime
            if mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
                print("auto-deleted expired case:", d.name)
        except Exception:
            continue
    return removed


def _cleanup_loop(interval_hours=6, days=None):
    if days is None:
        try:
            days = int(os.environ.get("CASE_EXPIRE_DAYS", "7"))
        except Exception:
            days = 7
    while True:
        try:
            cleanup_old_cases(days)
        except Exception:
            pass
        time.sleep(interval_hours * 3600)


@app.on_event("startup")
def _start_auto_cleanup():
    t = threading.Thread(target=_cleanup_loop, args=(6,), kwargs={"days": None}, daemon=True)
    t.start()
    print("[Lung3D] auto cleanup started: delete cases older than CASE_EXPIRE_DAYS(default 7d) every 6h")



def main():
    import uvicorn

    ap = argparse.ArgumentParser(description="Lung3D Web 后端")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--web", default=str(ROOT / "web"),
                    help="网页前端静态目录(可选)")
    args = ap.parse_args()

    web_dir = Path(args.web)
    if web_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Lung3D API: http://{args.host}:{args.port}")
    print(f"输出目录: {OUTPUT_DIR}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()