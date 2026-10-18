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
                         - 病例目录 zip(ct.nii.gz/combined.nii.gz/seg_totalseg/*/report.json/labels.json)
                         - 多个 nii.gz 同时上传(各自为独立结构)
  GET  /api/case/{id}?neutral=1  读取某病例 report.json。neutral=1 时结构的
                         name/key/mesh 替换为中性「结构 N / sN / nN.glb」
                         （微信小程序用，送审安全；缺省返回真实名称，网页端用）
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

结构命名（外部化）：
  本服务不内置任何领域结构名称。显示名 / 配色 / 处理角色来自数据目录内的
  labels.json（或 structures.json），也可用环境变量 STRUCTURE_LABELS 指定外部文件；
  都缺失时再回退读取 combined.nii.gz 的 NIfTI 头扩展（桌面工具写入的自描述标注）。
  全部缺失时，按标签序号生成中性名称「结构 N」并自动分配配色。

存储位置（持久化）：
  CASE_ROOT            病例产物根目录。默认 <项目>/web_output/cases（容器内=临时盘）。
                       生产指向挂载盘：轻量服务器 bind mount 或云托管 CFS，如 /data/cases。
  UPLOAD_STAGING_DIR   上传暂存目录。默认系统临时盘；处理完自动删除，不占持久盘。
  CASE_EXPIRE_DAYS     自动清理保留天数，默认 7；设为 0 关闭自动删除（持久盘场景）。
  命令行 --case-root   等价于设置 CASE_ROOT（优先级更高）。
  GET /api/health      返回 storage 段（backend/path/writable/free_gb/expire_days）。
"""
import argparse
import base64
import gc
import io
import json
import os
import re
import shutil
import tempfile
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

# ---------------------------------------------------------------------------
# 存储位置（持久化改造）
# ---------------------------------------------------------------------------
# 病例产物默认落在 web_output/cases（容器内即临时盘，重启/缩容会丢）。
# 生产环境用环境变量 CASE_ROOT 指到持久化位置：
#   轻量服务器 / 云主机：bind mount 宿主机目录，CASE_ROOT=/data/cases
#   微信云托管：挂载 CFS 文件存储，CASE_ROOT=/data/cases
# 上传的原始文件另有暂存目录 UPLOAD_STAGING_DIR（默认系统临时盘），
# 处理完即删除 —— 大文件不必写进持久盘，病例目录也不再残留 upload/。
CASE_ROOT_ENV = "CASE_ROOT"
UPLOAD_STAGING_ENV = "UPLOAD_STAGING_DIR"
DEFAULT_CASE_ROOT = ROOT / "web_output" / "cases"
STORAGE_CACHE_TTL = 30.0  # 健康检查里存储探测的缓存秒数（NFS 抖动时别拖慢探针）


def _resolve_dir(env_name, default):
    """环境变量里的目录 -> 绝对 Path；相对路径按项目根解析；空值用默认。"""
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return Path(default)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def storage_backend():
    """external = 已指向持久化位置（挂载盘/CFS）；local = 容器内临时盘。"""
    return "external" if (os.environ.get(CASE_ROOT_ENV) or "").strip() else "local"


def case_expire_days():
    """自动清理保留天数；<=0 表示不自动删除（持久盘场景可关掉）。"""
    try:
        return int(os.environ.get("CASE_EXPIRE_DAYS", "7"))
    except Exception:
        return 7


OUTPUT_DIR = _resolve_dir(CASE_ROOT_ENV, DEFAULT_CASE_ROOT)
UPLOAD_STAGING = _resolve_dir(UPLOAD_STAGING_ENV,
                              Path(tempfile.gettempdir()) / "lung3d_uploads")

_storage_cache = {"at": 0.0, "info": None}


def write_json_atomic(path, obj):
    """先写同目录临时文件再 os.replace —— 跨实例/挂载盘上不会读到半截 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("%s.tmp.%d.%d" % (path.name, os.getpid(), threading.get_ident()))
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def storage_info(refresh=False):
    """存储状态（供 /api/health 与启动自检使用），带 TTL 缓存。

    探测失败不抛异常 —— 健康检查必须始终能返回，挂载盘掉线时由
    writable=False / free_gb=null 体现。
    """
    now = time.time()
    if not refresh and _storage_cache["info"] and now - _storage_cache["at"] < STORAGE_CACHE_TTL:
        return _storage_cache["info"]

    info = {
        "backend": storage_backend(),
        "path": str(OUTPUT_DIR),
        "staging": str(UPLOAD_STAGING),
        "writable": False,
        "free_gb": None,
        "total_gb": None,
        "expire_days": case_expire_days(),
    }
    try:
        info["writable"] = OUTPUT_DIR.is_dir() and os.access(str(OUTPUT_DIR), os.W_OK)
    except Exception:
        info["writable"] = False
    try:
        probe = OUTPUT_DIR if OUTPUT_DIR.exists() else OUTPUT_DIR.parent
        usage = shutil.disk_usage(str(probe))
        info["free_gb"] = round(usage.free / (1024.0 ** 3), 2)
        info["total_gb"] = round(usage.total / (1024.0 ** 3), 2)
    except Exception:
        pass
    _storage_cache["info"] = info
    _storage_cache["at"] = now
    return info


def ensure_storage():
    """启动自检：建目录 + 真实写一个探针文件，返回 (ok, message)。"""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_STAGING.mkdir(parents=True, exist_ok=True)
        probe = OUTPUT_DIR / ".write_probe"
        probe.write_text(str(time.time()), encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, "ok"
    except Exception as e:
        return False, str(e)

# 并发处理上限：同时只处理 MAX_CONCURRENT_PROCESSING 个病例，
# 其余排队等待，避免多任务同时占用内存导致容器 OOM。
MAX_CONCURRENT_PROCESSING = max(1, int(os.environ.get("MAX_CONCURRENT_PROCESSING", "1")))
_PROCESS_SEM = threading.BoundedSemaphore(MAX_CONCURRENT_PROCESSING)

# ---------------------------------------------------------------------------
# 结构标注（外部化，服务本身不内置任何领域名词）
# ---------------------------------------------------------------------------
# 本服务是通用的三维模型生成 / 分发后端，不内置任何具体领域的结构名称。
# 结构的显示名、配色、处理角色全部来自「数据侧」提供的标注配置：
#   1) 上传的数据目录内的 labels.json / structures.json（推荐，随数据走）
#   2) 环境变量 STRUCTURE_LABELS 指向的外部配置文件（运维侧可选）
#   3) combined.nii.gz 的 NIfTI 头扩展（桌面工具随文件内嵌的同款 JSON）
# 都没有时，按标签序号生成中性名称（结构 1、结构 2 …）并自动分配配色。
LABELS_FILENAMES = ("labels.json", "structures.json")
LABELS_ENV = "STRUCTURE_LABELS"

# 处理角色：由标注配置的 role 字段声明，缺省空串（按普通结构处理）。
#   vessel —— 管状结构，按管径阈值剪除远端细支后重建
#   wall   —— 管壁类结构，单独降采样且不做平滑，保留细节
ROLE_VESSEL = "vessel"
ROLE_WALL = "wall"

PALETTE = [
    [1.0, 0.5, 0.2], [0.5, 0.2, 1.0], [0.2, 0.9, 0.9],
    [0.9, 0.4, 0.8], [0.6, 0.8, 0.2], [0.4, 0.6, 1.0],
]


def default_structure_name(key):
    """中性结构名：label_3 -> 结构 3；其余（由文件名派生）原样使用。"""
    m = re.fullmatch(r"label_(\d+)", str(key))
    return "结构 %s" % m.group(1) if m else str(key)


def empty_label_config():
    return {"by_label": {}, "by_key": {}}


def parse_label_config(path):
    """解析结构标注配置文件。成功返回 {"by_label": ..., "by_key": ...}，失败返回 None。

    格式（structures 列表的先后顺序即缺省 label 顺序）：
      {
        "schema": "lung3d.labels/1",
        "structures": [
          {"label": 1, "key": "s1", "name": "结构 1",
           "color": [1.0, 0.5, 0.2], "role": "vessel"}
        ]
      }
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    items = raw.get("structures") if isinstance(raw, dict) else raw
    return _label_config_from_items(items)


def _label_config_from_items(items):
    """structures 列表 -> {"by_label": {int: st}, "by_key": {str: st}}；输入不合法返回 None。"""
    if not isinstance(items, list):
        return None
    by_label, by_key = {}, {}
    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        try:
            label = int(it.get("label", i))
        except Exception:
            label = i
        key = str(it.get("key") or ("label_%d" % label))
        color = it.get("color")
        if isinstance(color, (list, tuple)) and len(color) == 3:
            try:
                color = [float(c) for c in color]
            except Exception:
                color = None
        else:
            color = None
        st = {
            "key": key,
            "name": str(it.get("name") or default_structure_name(key)),
            "color": color,
            "role": str(it.get("role") or ""),
        }
        by_label[label] = st
        by_key.setdefault(key, st)
    return {"by_label": by_label, "by_key": by_key}


def parse_label_config_from_nifti(path):
    """从 NIfTI 头扩展读取结构标注 JSON（combined.nii.gz 自描述，桌面工具写入）。

    扩展码 4(Comment) 内容为 labels.json 同款 JSON；读不到/解析失败返回 None。
    """
    try:
        import nibabel as nib
        img = nib.load(str(path))
        for ext in img.header.extensions:
            try:
                content = ext.get_content()
            except Exception:
                continue
            if isinstance(content, bytes):
                try:
                    content = content.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            if not isinstance(content, str) or not content.strip():
                continue
            try:
                raw = json.loads(content)
            except Exception:
                continue
            items = raw.get("structures") if isinstance(raw, dict) else raw
            cfg = _label_config_from_items(items)
            if cfg is not None and cfg["by_label"]:
                return cfg
    except Exception:
        return None
    return None


def load_label_config(*roots):
    """按顺序在给定目录中查找标注配置，再回退到 STRUCTURE_LABELS 环境变量。

    查找顺序：
      1) 各目录内的 labels.json / structures.json
      2) 环境变量 STRUCTURE_LABELS 指向的文件
      3) 各目录内 *.nii(.gz) 的 NIfTI 头扩展（combined.nii.gz 自描述标注）

    返回 {"by_label": ..., "by_key": ...}；一处都找不到时返回空配置
    （调用方据此生成中性结构名）。
    """
    cands = []
    for r in roots:
        if r:
            cands.extend(Path(r) / fn for fn in LABELS_FILENAMES)
    env_path = os.environ.get(LABELS_ENV, "").strip()
    if env_path:
        cands.append(Path(env_path))
    for p in cands:
        if p.is_file():
            cfg = parse_label_config(p)
            if cfg is not None:
                print("[Lung3D] structure labels loaded from", p)
                return cfg
    # 回退：NIfTI 头扩展（上传/zip 数据里可能没有 labels.json，但 combined 自带标注）
    for r in roots:
        if not r:
            continue
        r = Path(r)
        if not r.is_dir():
            continue
        for p in sorted(r.glob("*.nii*")):
            cfg = parse_label_config_from_nifti(p)
            if cfg is not None:
                print("[Lung3D] structure labels loaded from NIfTI header of", p)
                return cfg
    return empty_label_config()


app = FastAPI(title="Lung3D API", version="1.1")
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


def masks_from_nifti(path, base_name, label_cfg=None):
    """返回 {key: (bool_mask, affine)}。

    多标签体数据按标注配置拆分（配置里没有的标签用 label_<数值>），
    二值体数据作为单个结构，键名取自文件名。
    """
    import nibabel as nib

    by_label = (label_cfg or {}).get("by_label", {})
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim != 3:
        return None
    labels = np.unique(data)
    labels = labels[labels > 0]
    out = {}
    if len(labels) > 1:
        for lab in labels:
            st = by_label.get(int(lab))
            key = st["key"] if st else ("label_%d" % int(lab))
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
    """从数据目录收集：masks / ct / report / 标注配置。

    返回 (masks, ct_or_None, ct_affine, report, label_cfg)。
    """
    case_dir = Path(case_dir)
    label_cfg = load_label_config(case_dir)
    masks = {}
    affine = None
    ct_data, ct_affine = None, None

    combined = case_dir / "combined.nii.gz"
    if combined.exists():
        res = masks_from_nifti(combined, "combined.nii.gz", label_cfg)
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
    return masks, ct_data, ct_affine, report, label_cfg


# ---------------------------------------------------------------------------
# 构建病例
# ---------------------------------------------------------------------------
def build_case(case_dir, masks, ct_data, ct_affine, report, label_cfg=None):
    case_dir = Path(case_dir)
    mesh_dir = case_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    affine = None
    structures = []
    by_key = (label_cfg or {}).get("by_key", {})
    palette_idx = 0
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
        st = by_key.get(key)
        name = st["name"] if st else default_structure_name(key)
        if st and st.get("color"):
            color = st["color"]
        else:
            color = PALETTE[palette_idx % len(PALETTE)]
            palette_idx += 1
        role = st.get("role", "") if st else ""
        if role == ROLE_WALL:
            mesh = mask_to_trimesh(m, a, ds_factor=ds_wall, smooth=0)
        else:
            prune_mm = distal_prune_mm if role == ROLE_VESSEL else 0.0
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

    write_json_atomic(case_dir / "report.json", case_info)
    return case_info


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
def _process_upload_worker(case_dir, case_name, upload_dir=None):
    """后台异步处理上传的病例（受并发信号量限制，避免同时占用内存）。

    upload_dir 为上传暂存目录（本地临时盘）；缺省回退到 case_dir/upload，
    兼容旧调用与历史数据。处理结束（成功或失败）都会清掉暂存目录。
    """
    staging = Path(upload_dir) if upload_dir else (case_dir / "upload")
    with _PROCESS_SEM:
        try:
            tmp = staging
            if not tmp.is_dir():
                (case_dir / "error.txt").write_text("上传暂存目录不存在", encoding="utf-8")
                return
            masks, ct_data, ct_affine, report = {}, None, None, {}
            label_cfg = load_label_config(tmp, case_dir)
            for p in sorted(tmp.iterdir()):
                fname = p.name
                if fname.lower().endswith(".zip"):
                    with zipfile.ZipFile(p) as z:
                        z.extractall(tmp / "_zip")
                    masks, ct_data, ct_affine, report, label_cfg = extract_from_case_dir(tmp / "_zip")
                    break
                elif fname.lower().endswith((".nii", ".nii.gz")):
                    img = load_nifti(p)
                    if nifti_is_segmentation(img):
                        res = masks_from_nifti(p, fname, label_cfg)
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
                    "无法识别文件：需要分割 nii.gz、多标签 nii.gz 或数据目录 zip。收到: " + "; ".join(files_info),
                    encoding="utf-8")
                return
            case_info = build_case(case_dir, masks, ct_data, ct_affine, report, label_cfg)
            if not case_info["structures"] and case_info["ct_mesh"] is None:
                (case_dir / "error.txt").write_text("未提取到任何结构网格", encoding="utf-8")
                return
            if case_name and case_name.strip():
                case_info["name"] = case_name.strip()
                write_json_atomic(case_dir / "report.json", case_info)
            print("case processed:", case_dir.name)
        except Exception as e:
            try:
                (case_dir / "error.txt").write_text("处理失败: %s" % e, encoding="utf-8")
            except Exception:
                pass
            print("case processing error:", case_dir.name, e)
        finally:
            # 暂存目录用完即删：持久盘上不再堆积整份上传原件
            if upload_dir:
                shutil.rmtree(staging, ignore_errors=True)
            gc.collect()


@app.post("/api/process")
async def process_files(files: list[UploadFile] = File(...), name: str = Form(None), filename: str = Form(None)):
    if not files:
        return JSONResponse({"error": "未上传文件"}, status_code=400)

    case_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    case_dir = OUTPUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    # 原始上传先落本地临时盘（快、不占持久盘），处理完由 worker 删除
    tmp = UPLOAD_STAGING / case_id
    tmp.mkdir(parents=True, exist_ok=True)

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

    threading.Thread(target=_process_upload_worker, args=(case_dir, name, tmp),
                     daemon=True).start()
    return {"status": "processing", "case_id": case_id}


def neutralize_case_info(info):
    """小程序送审安全：响应中不出现任何人体结构相关字段值。

    - structures[].name  -> 中性「结构 N」
    - structures[].key   -> 中性 sN（原 key 如 lung_arteries 不下发）
    - structures[].mesh  -> /api/mesh/{case}/nN.glb（网格路由按序号别名解析回真实文件）
    只在响应层替换，不改磁盘上的 report.json，网页端仍可取到真实名称。
    """
    if isinstance(info, dict) and isinstance(info.get("structures"), list):
        case_id = str(info.get("case_id") or "")
        structs = []
        for i, st in enumerate(info["structures"], start=1):
            ns = dict(st)
            ns["name"] = "结构 %d" % i
            ns["key"] = "s%d" % i
            if isinstance(st.get("mesh"), str):
                ns["mesh"] = "/api/mesh/%s/n%d.glb" % (case_id, i)
            structs.append(ns)
        info = dict(info)
        info["structures"] = structs
    return info


@app.get("/api/case/{case_id}")
def get_case(case_id: str, neutral: bool = False):
    """neutral=1（小程序请求用）：structures 的 name/key/mesh 一律替换为中性
    「结构 N / sN / nN.glb」，响应中不出现任何人体结构名称；缺省 false 返回
    真实名称（网页端用）。磁盘上的 report.json 不受影响。"""
    d = OUTPUT_DIR / case_id
    rp = d / "report.json"
    if not rp.exists():
        ep = d / "error.txt"
        if ep.exists():
            return JSONResponse({"error": ep.read_text(encoding="utf-8")}, status_code=400)
        return JSONResponse({"error": "数据不存在或处理中"}, status_code=404)
    info = json.loads(rp.read_text(encoding="utf-8"))
    if neutral:
        info = neutralize_case_info(info)
    return info

@app.delete("/api/case/{case_id}")
def delete_case(case_id: str):
    d = OUTPUT_DIR / case_id
    rp = d / "report.json"
    if not rp.exists():
        return JSONResponse({"error": "数据不存在"}, status_code=404)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "case_id": case_id}

@app.post("/api/case/{case_id}/rename")
def rename_case(case_id: str, payload: dict):
    rp = OUTPUT_DIR / case_id / "report.json"
    if not rp.exists():
        return JSONResponse({"error": "数据不存在"}, status_code=404)
    new_name = (payload.get("name") or "").strip()
    if not new_name:
        return JSONResponse({"error": "名称不能为空"}, status_code=400)
    info = json.loads(rp.read_text(encoding="utf-8"))
    info["name"] = new_name
    write_json_atomic(rp, info)
    return {"ok": True, "case_id": case_id, "name": new_name}


@app.get("/api/mesh/{case_id}/{name}")
def get_mesh(case_id: str, name: str):
    p = OUTPUT_DIR / case_id / "meshes" / name
    if not p.exists() or p.suffix.lower() not in (".glb",):
        # 中性别名：nN.glb -> report.json 中第 N 个结构的真实网格文件（s 送审响应用）
        m = re.fullmatch(r"n(\d+)\.glb", name)
        if m and p.parent.is_dir():
            try:
                info = json.loads(
                    (OUTPUT_DIR / case_id / "report.json").read_text(encoding="utf-8"))
                idx = int(m.group(1)) - 1
                structs = info.get("structures") or []
                real = str(structs[idx].get("key")) + ".glb"
                p = p.parent / real
            except Exception:
                p = OUTPUT_DIR / case_id / "meshes" / "__missing__"
        if not p.exists() or p.suffix.lower() not in (".glb",):
            return JSONResponse({"error": "网格不存在"}, status_code=404)
    return FileResponse(p, media_type="model/gltf-binary")


@app.get("/api/health")
def health():
    """存活探针。附带存储状态：挂载盘掉线时 writable=false，便于告警区分。"""
    return {"status": "ok", "storage": storage_info()}


@app.get("/api/storage")
def storage_status():
    """存储详情（强制刷新缓存），排障用。"""
    return storage_info(refresh=True)


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
        return JSONResponse({"error": "数据不存在"}, status_code=404)
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
        return JSONResponse({"error": "数据不存在"}, status_code=404)
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
    """删除超过 days 天未更新的病例（存储自动清理）。days<=0 时不做任何删除。"""
    if days is None or days <= 0:
        return 0
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


def cleanup_stale_uploads(days=1):
    """清掉遗留的上传暂存目录（进程被杀 / 容器重启留下的孤儿）。"""
    cutoff = time.time() - days * 86400
    if not UPLOAD_STAGING.is_dir():
        return 0
    removed = 0
    for d in UPLOAD_STAGING.iterdir():
        try:
            if d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except Exception:
            continue
    return removed


def _cleanup_loop(interval_hours=6, days=None):
    while True:
        try:
            d = case_expire_days() if days is None else days
            if d and d > 0:
                cleanup_old_cases(d)
            cleanup_stale_uploads()
        except Exception:
            pass
        time.sleep(interval_hours * 3600)


@app.on_event("startup")
def _start_auto_cleanup():
    ok, msg = ensure_storage()
    info = storage_info(refresh=True)
    if ok:
        print("[Lung3D] storage ready: backend=%s path=%s staging=%s free=%sGB" % (
            info["backend"], info["path"], info["staging"], info["free_gb"]))
        if info["backend"] == "local":
            print("[Lung3D] WARNING 存储为容器本地盘，重启/缩容会丢数据；"
                  "生产请设置 CASE_ROOT 指向挂载盘或 CFS")
    else:
        print("[Lung3D] ERROR 存储不可写: %s (%s)" % (info["path"], msg))
    t = threading.Thread(target=_cleanup_loop, args=(6,), kwargs={"days": None}, daemon=True)
    t.start()
    d = case_expire_days()
    print("[Lung3D] auto cleanup started: CASE_EXPIRE_DAYS=%s (%s), every 6h" % (
        d, "disabled" if d <= 0 else "days"))



def main():
    import uvicorn

    global OUTPUT_DIR

    ap = argparse.ArgumentParser(description="Lung3D Web 后端")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--web", default=str(ROOT / "web"),
                    help="网页前端静态目录(可选)")
    ap.add_argument("--case-root", default=None,
                    help="病例产物根目录（等价于环境变量 CASE_ROOT）")
    args = ap.parse_args()

    # 命令行参数优先于环境变量（必须在挂载静态目录之前生效）
    if args.case_root:
        os.environ[CASE_ROOT_ENV] = args.case_root
        OUTPUT_DIR = _resolve_dir(CASE_ROOT_ENV, DEFAULT_CASE_ROOT)

    web_dir = Path(args.web)
    if web_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    ok, msg = ensure_storage()
    info = storage_info(refresh=True)
    print(f"Lung3D API: http://{args.host}:{args.port}")
    print(f"存储后端: {info['backend']}  病例目录: {info['path']}  "
          f"暂存目录: {info['staging']}  可用: {info['free_gb']}GB")
    if not ok:
        print(f"⚠️ 存储不可写: {msg}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()