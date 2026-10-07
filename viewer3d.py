#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lung3D 分割模型 3D 查看器（VTK + PySide6/Qt）
=============================================
功能：
  - 加载病例目录（ct.nii.gz / combined.nii.gz / seg_totalseg/*.nii.gz）或单个 nii.gz
  - 复选框切换显示 肺动脉 / 肺静脉 / 气管 / 肺结节 / CT 体渲染
  - 鼠标操作：左键旋转 / 中键平移 / 右键缩放 / 滚轮缩放
  - 快捷键：r 复位视角   s 截图   a 自动旋转开关
  - 模型透明度滑块、网格重建精度、CT 窗宽窗位预设

用法：
  python viewer3d.py                  # 打开文件对话框
  python viewer3d.py <病例目录|nii.gz>
  python viewer3d.py --selftest       # 自检：生成合成数据并渲染测试图
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import vtk
from vtk.util import numpy_support as nps

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QCheckBox, QComboBox, QSlider, QTextEdit, QLabel, QFrame,
    QFileDialog, QMessageBox, QGroupBox, QSizePolicy,
)

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

STRUCTURES = [
    dict(key="lung_arteries", name="肺动脉", color=(0.20, 0.40, 1.00), opacity=0.95),
    dict(key="lung_veins", name="肺静脉", color=(0.55, 0.10, 0.10), opacity=0.95),
    dict(key="lung_airways", name="气管", color=(0.20, 1.00, 0.30), opacity=0.95),
    dict(key="lung_nodules", name="肺结节", color=(1.00, 1.00, 0.10), opacity=1.00),
]
LABEL_MAP = {
    1: "lung_arteries",
    2: "lung_veins",
    3: "lung_airways",
    5: "lung_nodules",
}
PALETTE = [
    (1.0, 0.5, 0.2), (0.5, 0.2, 1.0), (0.2, 0.9, 0.9),
    (0.9, 0.4, 0.8), (0.6, 0.8, 0.2), (0.4, 0.6, 1.0),
]

VOLUME_PRESETS = {
    "肺窗": dict(
        color=[(-1024, 0.05, 0.05, 0.05), (-300, 0.55, 0.55, 0.55),
               (100, 0.90, 0.90, 0.90), (400, 1.0, 1.0, 1.0)],
        opacity=[(-1024, 0.0), (-700, 0.0), (-300, 0.08), (-50, 0.35),
                 (150, 0.55), (500, 0.75), (1500, 1.0)],
    ),
    "纵隔窗": dict(
        color=[(-100, 0.10, 0.10, 0.10), (40, 0.60, 0.60, 0.60),
               (200, 1.0, 1.0, 1.0)],
        opacity=[(-100, 0.0), (0, 0.08), (40, 0.30), (120, 0.60), (300, 0.90), (1000, 1.0)],
    ),
    "骨窗": dict(
        color=[(0, 0.05, 0.05, 0.05), (400, 0.60, 0.60, 0.60), (1500, 1.0, 1.0, 1.0)],
        opacity=[(0, 0.0), (250, 0.10), (400, 0.40), (800, 0.80), (1500, 1.0)],
    ),
}


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------
def np_affine_parts(affine):
    """从 nibabel 仿射矩阵解析 (spacing, origin, R)。"""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    R = affine[:3, :3] / spacing[np.newaxis, :]
    origin = (affine @ np.array([0, 0, 0, 1.0]))[:3]
    return spacing, origin, R


def ras_world_matrix(affine):
    """体素索引 -> 世界坐标(RAS，与 nibabel/NIfTI 标准一致) 的 4x4 矩阵。"""
    spacing, origin, R = np_affine_parts(affine)
    m = np.eye(4)
    m[:3, :3] = R @ np.diag(spacing)
    m[:3, 3] = origin
    return m


def downsample_binary(mask, factor):
    if factor <= 1 or mask.ndim != 3:
        return mask
    from scipy import ndimage
    z = ndimage.zoom(mask.astype(np.float32), 1.0 / factor, order=1)
    return (z > 0.5)


def mask_to_actor(mask, affine, ds_factor=2, smooth=8, color=(1, 1, 1), opacity=0.95):
    """二值掩膜 -> 平滑曲面 -> vtkActor。返回 None 表示空掩膜。"""
    import skimage.measure

    arr = downsample_binary((mask > 0), ds_factor) if ds_factor > 1 else (mask > 0)
    if arr.sum() == 0:
        return None

    verts, faces, _, _ = skimage.measure.marching_cubes(arr, level=0.5)
    # skimage 返回顶点列顺序 = 数组轴顺序 (axis0, axis1, axis2) = (i,j,k)，无需交换

    M = ras_world_matrix(affine)
    world = verts @ (M[:3, :3] * ds_factor).T + M[:3, 3]

    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(world.astype(np.float64), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    farr = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(len(faces), nps.numpy_to_vtk(farr, deep=True, array_type=vtk.VTK_ID_TYPE))
    pd.SetPolys(cells)

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(pd)
    smoother.SetNumberOfIterations(int(smooth))
    smoother.SetPassBand(0.001)
    smoother.SetFeatureAngle(120)
    smoother.BoundarySmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(smoother.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(float(opacity))
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().SetSpecularPower(20)
    actor.GetProperty().SetAmbient(0.15)
    return actor


def numpy_to_vtk_image(data):
    """int16 数组 -> vtkImageData（轴序 x,y,z，x 最快）。"""
    data = np.ascontiguousarray(data, dtype=np.int16)
    img = vtk.vtkImageData()
    x, y, z = data.shape
    img.SetDimensions(x, y, z)
    img.SetSpacing(1.0, 1.0, 1.0)
    arr = nps.numpy_to_vtk(data.ravel(order="F"), deep=True, array_type=vtk.VTK_SHORT)
    arr.SetName("CT")
    img.GetPointData().SetScalars(arr)
    return img


def build_ct_volume(data, affine, ds_factor=2):
    """CT 体素数据 -> vtkVolume（世界坐标与网格一致）。"""
    if ds_factor > 1:
        from scipy import ndimage
        data = ndimage.zoom(data.astype(np.float32), 1.0 / ds_factor, order=1).astype(np.int16)

    spacing, origin, R = np_affine_parts(affine)

    base = numpy_to_vtk_image(data)
    base.SetSpacing(*(spacing * ds_factor))
    base.SetOrigin(*tuple(origin))
    base.SetDirectionMatrix(*(R).ravel())

    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(base)
    prop = vtk.vtkVolumeProperty()
    prop.ShadeOff()
    prop.SetInterpolationTypeToLinear()
    prop.SetScalarOpacityUnitDistance(2.0)

    vol = vtk.vtkVolume()
    vol.SetMapper(mapper)
    vol.SetProperty(prop)
    return vol


def apply_preset(volume, preset_name):
    prop = volume.GetProperty()
    color = vtk.vtkColorTransferFunction()
    opacity = vtk.vtkPiecewiseFunction()
    p = VOLUME_PRESETS[preset_name]
    for pt in p["color"]:
        color.AddRGBPoint(*pt)
    for pt in p["opacity"]:
        opacity.AddPoint(*pt)
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class Viewer3D(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung3D 分割模型 3D 查看器")
        self.resize(1180, 820)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.12, 0.13, 0.16)
        self.actors = {}      # key -> vtkActor
        self.vol_actor = None
        self.masks = {}       # key -> 完整分辨率二值掩膜
        self.affine = None
        self.ct_data = None
        self.ct_affine = None
        self.volumes_cm3 = {}
        self.ds_factor = 2
        self.auto_rotate = False
        self.mesh_cache = {}  # (key, ds) -> actor
        self.source_desc = []
        self.prop_to_key = {}  # vtkActor -> key
        self.selected_key = None
        self.outline_actor = None
        self._press_pos = None

        self._build_ui()

        self.iw = QVTKRenderWindowInteractor(self.render_widget)
        self.iw.setMinimumSize(200, 200)
        self.iw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.renWin = self.iw.GetRenderWindow()
        self.renWin.AddRenderer(self.renderer)
        self.renWin.SetMultiSamples(8)
        self.render_widget.layout().addWidget(self.iw)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iw.SetInteractorStyle(style)
        self.iw.AddObserver("KeyPressEvent", self._on_key)
        self.iw.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self.iw.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        self.iw.Initialize()

        self.rotate_timer = QTimer(self)
        self.rotate_timer.timeout.connect(self._rotate_step)

        self._setup_lights()
        self._render()
        self._set_status("就绪。打开病例目录或 nii.gz 开始查看。")

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)
        lay.setContentsMargins(4, 4, 4, 4)

        # 右侧控制面板
        self.panel = QWidget()
        self.panel.setFixedWidth(280)
        pl = QVBoxLayout(self.panel)
        pl.setContentsMargins(6, 6, 6, 6)

        bar = QHBoxLayout()
        for text, slot in [("打开病例…", self.open_case_dir), ("打开 nii.gz…", self.open_single_file),
                           ("复位", self.reset_camera), ("截图", self.screenshot)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        pl.addLayout(bar)

        bar2 = QHBoxLayout()
        self.btn_hide_sel = QPushButton("隐藏选中")
        self.btn_hide_sel.clicked.connect(self.hide_selected)
        bar2.addWidget(self.btn_hide_sel)
        self.btn_show_all = QPushButton("全部显示")
        self.btn_show_all.clicked.connect(self.show_all)
        bar2.addWidget(self.btn_show_all)
        pl.addLayout(bar2)

        self.btn_rot = QPushButton("自动旋转: 关")
        self.btn_rot.setCheckable(True)
        self.btn_rot.toggled.connect(self.toggle_auto_rotate)
        pl.addWidget(self.btn_rot)

        # 模型选择
        g1 = QGroupBox("显示模型")
        g1lay = QVBoxLayout(g1)
        self.var_frame = QWidget()
        self.var_lay = QVBoxLayout(self.var_frame)
        self.var_lay.setContentsMargins(0, 0, 0, 0)
        self.var_lay.setSpacing(2)
        g1lay.addWidget(self.var_frame)
        self.check_vars = {}    # key -> QCheckBox
        self.color_swatches = {}
        self._rebuild_checkboxes({})
        pl.addWidget(g1)

        # 参数
        g2 = QGroupBox("显示参数")
        g2lay = QGridLayout(g2)

        g2lay.addWidget(QLabel("重建精度"), 0, 0)
        self.ds_combo = QComboBox()
        self.ds_combo.addItems(["低(最快)", "中", "高(最慢)"])
        self.ds_combo.setCurrentText("中")
        self.ds_combo.currentTextChanged.connect(self._on_ds_change)
        g2lay.addWidget(self.ds_combo, 0, 1)

        g2lay.addWidget(QLabel("模型透明度"), 1, 0)
        self.op_slider = QSlider(Qt.Horizontal)
        self.op_slider.setRange(5, 100)
        self.op_slider.setValue(95)
        self.op_slider.valueChanged.connect(self._on_opacity)
        g2lay.addWidget(self.op_slider, 1, 1)

        g2lay.addWidget(QLabel("CT 窗位"), 2, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(VOLUME_PRESETS.keys()))
        self.preset_combo.setCurrentText("肺窗")
        self.preset_combo.currentTextChanged.connect(self._on_preset_change)
        g2lay.addWidget(self.preset_combo, 2, 1)

        pl.addWidget(g2)

        # 信息
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setFixedHeight(170)
        self.info.setStyleSheet("font-family: Consolas; font-size: 11px;")
        pl.addWidget(self.info)

        lay.addWidget(self.panel)

        # 渲染区
        self.render_widget = QWidget()
        self.render_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rlay = QVBoxLayout(self.render_widget)
        rlay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.render_widget, 1)

    def _rebuild_checkboxes(self, masks):
        for i in reversed(range(self.var_lay.count())):
            w = self.var_lay.itemAt(i).widget()
            if w is not None:
                w.deleteLater()
        self.check_vars.clear()
        self.color_swatches.clear()

        palette_iter = iter(PALETTE)
        for st in STRUCTURES:
            if st["key"] not in masks:
                continue
            self._add_checkbox(st["key"], st["name"], st["color"])
        for key in masks:
            if key not in [s["key"] for s in STRUCTURES]:
                self._add_checkbox(key, key, next(palette_iter))

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        self.var_lay.addWidget(sep)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        sw = QLabel()
        sw.setFixedSize(14, 14)
        sw.setStyleSheet("background-color: #888888; border-radius: 7px;")
        rl.addWidget(sw)
        self.ct_cb = QCheckBox("CT 体渲染")
        self.ct_cb.toggled.connect(self.toggle_ct)
        rl.addWidget(self.ct_cb)
        self.var_lay.addWidget(row)

    def _add_checkbox(self, key, label, color):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        sw = QLabel()
        sw.setFixedSize(14, 14)
        sw.setStyleSheet("background-color: #%02x%02x%02x; border-radius: 7px;"
                         % tuple(int(round(v * 255)) for v in color))
        rl.addWidget(sw)
        cb = QCheckBox(label)
        cb.setChecked(True)
        cb.toggled.connect(lambda chk, k=key: self._on_toggle(k))
        rl.addWidget(cb)
        rl.addStretch(1)
        self.var_lay.addWidget(row)
        self.check_vars[key] = cb
        self.color_swatches[key] = color

    # ---------------- 数据加载 ----------------
    def open_case_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择病例目录（含 ct.nii.gz / combined.nii.gz）")
        if path:
            self.load_case_dir(Path(path))

    def open_single_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择分割 nii.gz 文件", "", "NIfTI (*.nii.gz *.nii);;所有文件 (*)")
        if path:
            self.load_single_file(Path(path))

    def load_case_dir(self, case_dir):
        case_dir = Path(case_dir)
        if not case_dir.is_dir():
            self._error(f"目录不存在: {case_dir}")
            return
        self.reset_scene()
        self.source_desc = [f"病例目录: {case_dir.name}"]

        ct = case_dir / "ct.nii.gz"
        if not ct.exists():
            ct = next((p for p in sorted(case_dir.glob("*.nii.gz"))
                       if p.stem.lower().startswith("ct")), None)
        if ct and ct.exists():
            self._load_ct(ct)

        masks = {}
        combined = case_dir / "combined.nii.gz"
        if combined.exists():
            self._set_status("加载 combined.nii.gz …")
            img = self._load_nifti(combined)
            data = np.asarray(img.dataobj)
            for label, key in LABEL_MAP.items():
                masks[key] = (data == label)
            if self.affine is None:
                self.affine = img.affine
            self.source_desc.append("combined.nii.gz")

        seg_dir = case_dir / "seg_totalseg"
        if seg_dir.is_dir():
            for p in sorted(seg_dir.glob("*.nii.gz")):
                key = p.name[:-7]  # 去掉 ".nii.gz"
                if key in masks or not self._is_known(key):
                    continue
                self._set_status(f"加载 {p.name} …")
                img = self._load_nifti(p)
                masks[key] = (np.asarray(img.dataobj) > 0.5)
                if self.affine is None:
                    self.affine = img.affine
                self.source_desc.append(p.name)

        for p in sorted(case_dir.glob("*.nii.gz")):
            key = p.name[:-7]
            if p.name == "combined.nii.gz" or key in masks or not self._is_known(key):
                continue
            img = self._load_nifti(p)
            masks[key] = (np.asarray(img.dataobj) > 0.5)
            if self.affine is None:
                self.affine = img.affine
            self.source_desc.append(p.name)

        if not masks and self.ct_data is None:
            self._error("目录中未找到可识别的分割文件（combined.nii.gz / seg_totalseg / *结构名*.nii.gz）")
            return
        if self.affine is None and self.ct_data is not None:
            self.affine = self.ct_affine

        self._load_volumes_report(case_dir)
        self.masks = {k: v for k, v in masks.items() if v.any()}
        self._rebuild_checkboxes(self.masks)
        self._rebuild_all()
        self._update_info()
        self.reset_camera()
        self._render()
        self._set_status(f"已加载 {len(self.masks)} 个结构。" + " / ".join(self.source_desc))

    def load_single_file(self, path):
        path = Path(path)
        if not path.exists():
            self._error(f"文件不存在: {path}")
            return
        self.reset_scene()
        img = self._load_nifti(path)
        key = path.name[:-7]
        data = np.asarray(img.dataobj)
        if data.ndim != 3:
            self._error(f"仅支持 3D nii.gz，当前维度: {data.ndim}")
            return
        labels = np.unique(data)
        labels = labels[labels > 0]
        if len(labels) > 1:
            masks = {}
            for lab in labels:
                name = LABEL_MAP.get(int(lab), f"label_{int(lab)}")
                masks[name] = (data == lab)
            self.masks = {k: v for k, v in masks.items() if v.any()}
        else:
            self.masks = {key: (data > 0)}
        self.affine = img.affine
        self.ct_affine = img.affine
        self.source_desc = [f"文件: {path.name}"]
        self._rebuild_checkboxes(self.masks)
        self._rebuild_all()
        self._update_info()
        self.reset_camera()
        self._render()
        self._set_status(f"已加载 {len(self.masks)} 个结构 (来自 {path.name})")

    def _load_nifti(self, path):
        import nibabel as nib
        return nib.load(str(path))

    def _load_ct(self, path):
        self._set_status(f"加载 CT {path.name} …")
        img = self._load_nifti(path)
        data = np.asarray(img.dataobj)
        if data.ndim == 4:
            data = data[..., 0]
        if data.dtype.kind in ("u", "i") and data.min() >= 0 and data.max() <= 4095:
            data = data.astype(np.int16) - 1024
        self.ct_data = np.ascontiguousarray(data)
        self.ct_affine = img.affine
        if self.affine is None:
            self.affine = img.affine

    def _load_volumes_report(self, case_dir):
        rp = case_dir / "report.json"
        if not rp.exists():
            return
        try:
            with open(rp, encoding="utf-8") as f:
                report = json.load(f)
            vols = report.get("volumes_cm3") or {}
            self.volumes_cm3 = {k: v for k, v in vols.items() if isinstance(v, (int, float))}
            if report.get("nodules"):
                self.source_desc.append(f"结节数: {len(report['nodules'])}")
        except Exception:
            pass

    @staticmethod
    def _is_known(key):
        return key in [s["key"] for s in STRUCTURES]

    # ---------------- 场景 ----------------
    def reset_scene(self):
        for k, act in list(self.actors.items()):
            self.renderer.RemoveActor(act)
        self.actors.clear()
        self.prop_to_key.clear()
        if self.vol_actor is not None:
            self.renderer.RemoveVolume(self.vol_actor)
            self.vol_actor = None
        self._clear_selection()
        self.mesh_cache.clear()
        self.masks.clear()
        self.affine = None
        self.ct_data = None
        self.ct_affine = None
        self.volumes_cm3 = {}
        self.source_desc = []
        self.auto_rotate = False
        self.btn_rot.setChecked(False)
        self.rotate_timer.stop()
        self._update_info()

    def _get_or_build_mesh(self, key):
        cache_key = (key, self.ds_factor)
        if cache_key in self.mesh_cache:
            return self.mesh_cache[cache_key]
        mask = self.masks.get(key)
        if mask is None or self.affine is None:
            return None
        st = next((s for s in STRUCTURES if s["key"] == key), None)
        color = self.color_swatches.get(key, (1, 1, 0.1))
        opacity = st["opacity"] if st else 1.0
        self._set_status(f"生成网格: {key} …")
        QApplication.processEvents()
        t0 = time.time()
        actor = mask_to_actor(mask, self.affine, ds_factor=self.ds_factor,
                              color=color, opacity=opacity)
        self.mesh_cache[cache_key] = actor
        self._set_status(f"{key}: 网格完成 ({time.time()-t0:.1f}s)")
        return actor

    def _rebuild_all(self):
        for k, act in list(self.actors.items()):
            self.renderer.RemoveActor(act)
        self.actors.clear()
        self.prop_to_key.clear()
        self.mesh_cache.clear()
        self._clear_selection()
        for key, cb in self.check_vars.items():
            if cb.isChecked():
                self._show_structure(key)

    def _show_structure(self, key):
        if key in self.actors:
            return
        actor = self._get_or_build_mesh(key)
        if actor is None:
            return
        self.renderer.AddActor(actor)
        self.actors[key] = actor
        self.prop_to_key[actor] = key

    def _hide_structure(self, key):
        act = self.actors.pop(key, None)
        if act is not None:
            self.renderer.RemoveActor(act)
            self.prop_to_key.pop(act, None)
        if self.selected_key == key:
            self._clear_selection()

    def _on_toggle(self, key):
        if self.check_vars[key].isChecked():
            self._show_structure(key)
        else:
            self._hide_structure(key)
        self._render()

    # ---------------- 3D 对象选择 ----------------
    def _on_left_press(self, obj, event):
        self._press_pos = self.iw.GetEventPosition()

    def _on_left_release(self, obj, event):
        if self._press_pos is None:
            return
        x, y = self.iw.GetEventPosition()
        dx = abs(x - self._press_pos[0]) + abs(y - self._press_pos[1])
        self._press_pos = None
        if dx > 5:  # 拖动=旋转，不触发选择
            return
        picker = vtk.vtkPropPicker()
        if picker.Pick(x, y, 0, self.renderer):
            key = self.prop_to_key.get(picker.GetViewProp())
            if key:
                self._select_structure(key)

    def _select_structure(self, key):
        if key not in self.actors:
            return
        self.selected_key = key
        name = self._structure_name(key)
        self._update_outline()
        self._set_status(f"已选中: {name}（按 H 隐藏，按 U 全部显示，或点按钮）")
        self._render()

    def _clear_selection(self):
        self.selected_key = None
        if self.outline_actor is not None:
            self.renderer.RemoveActor(self.outline_actor)
            self.outline_actor = None

    def _update_outline(self):
        key = self.selected_key
        if self.outline_actor is not None:
            self.renderer.RemoveActor(self.outline_actor)
            self.outline_actor = None
        actor = self.actors.get(key)
        if actor is None:
            return
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(actor.GetMapper().GetInput())
        outline.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        oa = vtk.vtkActor()
        oa.SetMapper(mapper)
        oa.GetProperty().SetColor(1.0, 1.0, 1.0)
        oa.GetProperty().SetLineWidth(2.5)
        self.outline_actor = oa
        self.renderer.AddActor(oa)

    def hide_selected(self):
        key = self.selected_key
        if key is None:
            self._set_status("请先在 3D 图像中点击要隐藏的对象")
            return
        cb = self.check_vars.get(key)
        if cb is not None:
            cb.setChecked(False)  # 触发 _on_toggle -> 隐藏
        self._clear_selection()
        self._set_status(f"已隐藏: {self._structure_name(key)}")
        self._render()

    def show_all(self):
        for key, cb in self.check_vars.items():
            if not cb.isChecked():
                cb.setChecked(True)
        self._clear_selection()
        self._set_status("已显示全部对象")
        self._render()

    def _structure_name(self, key):
        return next((s["name"] for s in STRUCTURES if s["key"] == key), key)

    def _on_ds_change(self, text):
        self.ds_factor = {"低(最快)": 4, "中": 2, "高(最慢)": 1}.get(text, 2)
        if self.ct_cb.isChecked() and self.ct_data is not None:
            self.toggle_ct(False)
            self.toggle_ct(True)
        self._rebuild_all()
        self._render()

    def _on_opacity(self, val):
        opacity = val / 100.0
        for act in self.actors.values():
            act.GetProperty().SetOpacity(opacity)
        self._render()

    def _on_preset_change(self, text):
        if self.vol_actor is not None:
            apply_preset(self.vol_actor, text)
            self._render()

    # ---------------- CT 体渲染 ----------------
    def toggle_ct(self, checked):
        if checked:
            if self.ct_data is None:
                self._error("未加载 CT 数据（ct.nii.gz）")
                self.ct_cb.setChecked(False)
                return
            if self.vol_actor is not None:
                self.renderer.RemoveVolume(self.vol_actor)
                self.vol_actor = None
            self._set_status("生成 CT 体渲染 …")
            QApplication.processEvents()
            t0 = time.time()
            self.vol_actor = build_ct_volume(self.ct_data, self.ct_affine, ds_factor=self.ds_factor)
            apply_preset(self.vol_actor, self.preset_combo.currentText())
            self.renderer.AddVolume(self.vol_actor)
            self._set_status(f"CT 体渲染完成 ({time.time()-t0:.1f}s)")
        else:
            if self.vol_actor is not None:
                self.renderer.RemoveVolume(self.vol_actor)
                self.vol_actor = None
        self._render()

    # ---------------- 视图 ----------------
    def _setup_lights(self):
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(1, 1, 1)
        light.SetFocalPoint(0, 0, 0)
        self.renderer.AddLight(light)

    def reset_camera(self):
        self.renderer.ResetCamera()
        self._render()

    def screenshot(self, filename=None):
        if filename is None:
            default = f"lung3d_view_{time.strftime('%Y%m%d_%H%M%S')}.png"
            filename, _ = QFileDialog.getSaveFileName(
                self, "保存截图", default, "PNG (*.png)")
            if not filename:
                return
        self.renWin.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.renWin)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.SetShouldRerender(1)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        self._set_status(f"截图已保存: {filename}")

    def toggle_auto_rotate(self, checked):
        self.auto_rotate = checked
        self.btn_rot.setText("自动旋转: 开" if checked else "自动旋转: 关")
        if checked:
            self.rotate_timer.start(30)
        else:
            self.rotate_timer.stop()

    def _rotate_step(self):
        cam = self.renderer.GetActiveCamera()
        cam.Azimuth(1.0)
        self.renWin.Render()

    def _on_key(self, obj, event):
        key = obj.GetKeySym()
        if key in ("r", "R"):
            self.reset_camera()
        elif key in ("s", "S"):
            self.screenshot()
        elif key in ("a", "A"):
            self.btn_rot.toggle()
        elif key in ("h", "H"):
            self.hide_selected()
        elif key in ("u", "U"):
            self.show_all()

    # ---------------- 信息 ----------------
    def _update_info(self):
        lines = []
        if self.affine is not None:
            spacing, _, _ = np_affine_parts(self.affine)
            lines.append("体素间距(mm): %.2f x %.2f x %.2f" % tuple(spacing))
        if self.masks:
            total = sum(int(m.sum()) for m in self.masks.values())
            lines.append("分割体素数: %d" % total)
            if self.affine is not None:
                vox = abs(np.linalg.det(self.affine)) / 1000.0  # cm^3
                lines.append("总体积: %.1f cm^3" % (total * vox))
            for key, m in self.masks.items():
                name = next((s["name"] for s in STRUCTURES if s["key"] == key), key)
                vol = self.volumes_cm3.get(key)
                lines.append("  %s: %s" % (name,
                            ("%.1f cm^3" % vol) if vol is not None else ("%d 体素" % int(m.sum()))))
        lines.append("")
        lines.append("操作: 左键旋转 / 中键平移 / 右键或滚轮缩放")
        lines.append("点击对象=选中 / H 隐藏选中 / U 全部显示")
        lines.append("快捷键: r 复位 / s 截图 / a 自动旋转")
        self.info.setPlainText("\n".join(lines))

    def _set_status(self, text):
        self.statusBar().showMessage(text)

    def _error(self, text):
        QMessageBox.critical(self, "Lung3D 查看器", text)

    def _render(self):
        self.renWin.Render()

    def closeEvent(self, event):
        self.rotate_timer.stop()
        try:
            self.iw.Finalize()
        except Exception:
            pass
        event.accept()


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selftest(out_dir):
    """生成合成病例并验证 加载 -> 网格 -> 体渲染 -> 截图。"""
    import nibabel as nib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([1.0, 1.0, 1.5])
    affine[:3, 3] = [-80, -80, -20]

    shape = (160, 160, 120)
    xx, yy, zz = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    c = np.array(shape) / 2.0
    data = np.zeros(shape, dtype=np.uint8)

    tube = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2) <= 6 ** 2
    data[tube & (np.abs(zz - c[2]) < 45)] = 1                      # 肺动脉
    vein = ((xx - c[0] - 18) ** 2 + (yy - c[1]) ** 2) <= 8 ** 2
    data[vein & (np.abs(zz - c[2]) < 45)] = 2                      # 肺静脉
    airway = ((xx - c[0] + 18) ** 2 + (yy - c[1]) ** 2) <= 9 ** 2
    data[airway & (np.abs(zz - c[2]) < 45)] = 3                    # 气管
    balls = ((xx - c[0]) ** 2 + (yy - c[1] - 20) ** 2 + (zz - c[2] + 10) ** 2) <= 4 ** 2
    data[balls] = 5                                                 # 结节

    ct = (data == 0) * -1000 + data * 200
    ct = ct.astype(np.int16)

    nib.save(nib.Nifti1Image(ct, affine), str(out_dir / "ct.nii.gz"))
    nib.save(nib.Nifti1Image(data, affine), str(out_dir / "combined.nii.gz"))
    report = {"volumes_cm3": {"lung_arteries": 1.2, "lung_veins": 1.0,
                              "lung_airways": 1.4, "lung_nodules": 0.3}}
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    app = QApplication.instance() or QApplication([])
    win = Viewer3D()
    win.resize(1000, 800)
    win.show()
    app.processEvents()
    win.load_case_dir(out_dir)
    win.ct_cb.setChecked(True)
    app.processEvents()
    win.ds_factor = 1
    win._rebuild_all()
    win.reset_camera()
    app.processEvents()
    shot = str(out_dir / "selftest.png")
    win.screenshot(shot)
    print("SELFTEST_OK", shot, flush=True)
    win.close()
    app.processEvents()


def main():
    ap = argparse.ArgumentParser(description="Lung3D 分割模型 3D 查看器")
    ap.add_argument("path", nargs="?", help="病例目录或 nii.gz 文件")
    ap.add_argument("--selftest", action="store_true", help="运行自检")
    ap.add_argument("--selftest-out",
                    default=str(Path(os.environ.get("TEMP", ".")) / "lung3d_selftest"))
    args = ap.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    if args.selftest:
        selftest(args.selftest_out)
        return

    win = Viewer3D()
    win.show()
    if args.path:
        p = Path(args.path)
        if p.is_dir():
            QTimer.singleShot(100, lambda: win.load_case_dir(p))
        else:
            QTimer.singleShot(100, lambda: win.load_single_file(p))
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, "Lung3D 查看器发生错误:\n\n" + tb, "Lung3D 错误", 0x10)
        except Exception:
            pass
        print(tb)
        input("发生错误，按回车退出…")