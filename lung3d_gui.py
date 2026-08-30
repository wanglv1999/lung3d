#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lung3D 分割重建图形界面（PySide6）
==================================
对 lung3d_reconstruct.py 命令行批处理脚本的图形化封装：
  - 用目录/文件对话框选择输入（DICOM 目录、NIfTI 文件或含多病例的父目录）
  - 图形化设置：推理设备、快速模式、结节检测阈值、模型目录、输出目录
  - 后台线程批处理，实时日志与进度显示，支持中途停止

用法：
  run_lung3d_gui.bat
  或: python lung3d_gui.py
"""
import argparse
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QGroupBox, QFrame,
)


class StdoutPipe:
    """把 print() 输出转发为 Qt 信号。"""

    def __init__(self, emit):
        self._emit = emit

    def write(self, s):
        if s:
            self._emit(s)

    def flush(self):
        pass


class ReconstructWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int)          # (已完成, 总数)
    done = Signal(int, int, str)         # (成功, 失败, 消息)

    def __init__(self, input_path, output_path, device, fast,
                 no_nodules, nodule_score, nodule_model, parent=None):
        super().__init__(parent)
        self._input = input_path
        self._output = output_path
        self._device = device
        self._fast = fast
        self._no_nodules = no_nodules
        self._nodule_score = nodule_score
        self._nodule_model = nodule_model
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        import lung3d_reconstruct as lr

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = StdoutPipe(self.log)
        sys.stderr = StdoutPipe(self.log)
        try:
            args = argparse.Namespace(
                device=self._device,
                fast=self._fast,
                no_nodules=self._no_nodules,
                nodule_score=self._nodule_score,
                nodule_model=self._nodule_model,
            )
            root = Path(self._input).resolve()
            out_root = Path(self._output).resolve()
            out_root.mkdir(parents=True, exist_ok=True)
            if not root.exists():
                raise RuntimeError(f"输入不存在: {root}")

            cases = lr.find_dicom_cases(root)
            if not cases and (root.is_file() and root.name.lower().endswith((".nii", ".nii.gz"))):
                cases = [root]
            if not cases:
                raise RuntimeError("未找到可处理的病例（DICOM目录或NIfTI）。")

            total = len(cases)
            self.log.emit(f"共 {total} 个病例，输出到 {out_root}\n")
            ok = failed = 0
            for i, case in enumerate(cases, 1):
                if self._stop:
                    self.log.emit("\n[停止] 用户请求停止，已跳过剩余病例。\n")
                    break
                case_dir = out_root / f"case_{i:03d}_{case.name}"
                try:
                    lr.process_case(case, case_dir, args)
                    ok += 1
                except Exception as e:
                    self.log.emit(f"!!! 病例 {case} 处理失败: {e}\n{traceback.format_exc()}\n")
                    failed += 1
                self.progress.emit(i, total)
            msg = "全部完成。" if ok else "未完成任何病例。"
            self.done.emit(ok, failed, msg)
        except Exception:
            self.log.emit(traceback.format_exc())
            self.done.emit(0, 1, f"处理失败: {traceback.format_exc().splitlines()[-1]}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung3D 分割重建")
        self.resize(920, 640)

        self.settings = QSettings("Lung3D", "ReconstructGUI")
        self.worker = None

        self._build_ui()
        self._load_settings()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(10, 10, 10, 10)

        g1 = QGroupBox("输入 / 输出")
        g1l = QGridLayout(g1)

        g1l.addWidget(QLabel("输入路径"), 0, 0)
        self.in_edit = QLineEdit()
        self.in_edit.setPlaceholderText("DICOM目录 / NIfTI文件 / 含多个病例的父目录")
        g1l.addWidget(self.in_edit, 0, 1)
        self.btn_in_dir = QPushButton("浏览目录…")
        self.btn_in_dir.clicked.connect(self.pick_input_dir)
        g1l.addWidget(self.btn_in_dir, 0, 2)
        self.btn_in_file = QPushButton("选择NIfTI…")
        self.btn_in_file.clicked.connect(self.pick_input_file)
        g1l.addWidget(self.btn_in_file, 0, 3)

        g1l.addWidget(QLabel("输出目录"), 1, 0)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("留空则自动设为输入目录下的 lung3d_output")
        g1l.addWidget(self.out_edit, 1, 1, 1, 2)
        self.btn_out = QPushButton("浏览…")
        self.btn_out.clicked.connect(self.pick_output_dir)
        g1l.addWidget(self.btn_out, 1, 3)
        lay.addWidget(g1)

        g2 = QGroupBox("参数设置")
        g2l = QGridLayout(g2)

        g2l.addWidget(QLabel("推理设备"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        g2l.addWidget(self.device_combo, 0, 1)

        self.fast_cb = QCheckBox("快速模式（省显存）")
        g2l.addWidget(self.fast_cb, 0, 2)
        self.no_nodule_cb = QCheckBox("跳过肺结节检测")
        g2l.addWidget(self.no_nodule_cb, 0, 3)

        g2l.addWidget(QLabel("结节分数阈值"), 1, 0)
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.01, 1.0)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setValue(0.3)
        g2l.addWidget(self.score_spin, 1, 1)

        g2l.addWidget(QLabel("结节模型目录"), 2, 0)
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("默认: 本目录下 monai_nodule")
        g2l.addWidget(self.model_edit, 2, 1, 1, 2)
        self.btn_model = QPushButton("浏览…")
        self.btn_model.clicked.connect(self.pick_model_dir)
        g2l.addWidget(self.btn_model, 2, 3)
        lay.addWidget(g2)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_start.setMinimumHeight(34)
        self.btn_start.clicked.connect(self.start)
        btn_row.addWidget(self.btn_start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setMinimumHeight(34)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        btn_row.addWidget(self.btn_stop)
        self.btn_open_out = QPushButton("打开输出目录")
        self.btn_open_out.setEnabled(False)
        self.btn_open_out.clicked.connect(self.open_output)
        btn_row.addWidget(self.btn_open_out)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m 个病例")
        lay.addWidget(self.progress)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("font-family: Consolas; font-size: 12px;")
        lay.addWidget(self.log_area, 1)

    # ---------------- 文件选择 ----------------
    def pick_input_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输入目录（病例或含多病例的父目录）")
        if d:
            self.in_edit.setText(d)

    def pick_input_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择 NIfTI 文件", "", "NIfTI (*.nii.gz *.nii);;所有文件 (*)")
        if f:
            self.in_edit.setText(f)

    def pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.out_edit.setText(d)

    def pick_model_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择 MONAI 结节模型目录")
        if d:
            self.model_edit.setText(d)

    # ---------------- 配置持久化 ----------------
    def _load_settings(self):
        self.in_edit.setText(self.settings.value("input", ""))
        self.out_edit.setText(self.settings.value("output", ""))
        dev = self.settings.value("device", "cuda")
        self.device_combo.setCurrentText(dev if dev in ("cuda", "cpu") else "cuda")
        self.fast_cb.setChecked(self.settings.value("fast", "false") == "true")
        self.no_nodule_cb.setChecked(self.settings.value("no_nodules", "false") == "true")
        self.score_spin.setValue(float(self.settings.value("nodule_score", 0.3)))
        self.model_edit.setText(self.settings.value("nodule_model", ""))

    def _save_settings(self):
        self.settings.setValue("input", self.in_edit.text())
        self.settings.setValue("output", self.out_edit.text())
        self.settings.setValue("device", self.device_combo.currentText())
        self.settings.setValue("fast", "true" if self.fast_cb.isChecked() else "false")
        self.settings.setValue("no_nodules", "true" if self.no_nodule_cb.isChecked() else "false")
        self.settings.setValue("nodule_score", self.score_spin.value())
        self.settings.setValue("nodule_model", self.model_edit.text())

    # ---------------- 控制 ----------------
    def start(self):
        inp = self.in_edit.text().strip()
        if not inp:
            QMessageBox.warning(self, "提示", "请选择输入目录或 NIfTI 文件。")
            return
        if not Path(inp).exists():
            QMessageBox.warning(self, "提示", f"输入路径不存在:\n{inp}")
            return

        out = self.out_edit.text().strip()
        if not out:
            p = Path(inp)
            out = str((p.parent if p.is_file() else p) / "lung3d_output")
            self.out_edit.setText(out)

        model = self.model_edit.text().strip()
        if not model:
            model = str(Path(__file__).resolve().parent / "monai_nodule")

        self._save_settings()
        self.log_area.clear()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_open_out.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self.worker = ReconstructWorker(
            inp, out,
            device=self.device_combo.currentText(),
            fast=self.fast_cb.isChecked(),
            no_nodules=self.no_nodule_cb.isChecked(),
            nodule_score=float(self.score_spin.value()),
            nodule_model=model,
        )
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.start()
        self.statusBar().showMessage("处理中…")

    def stop(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.btn_stop.setEnabled(False)
            self.statusBar().showMessage("正在停止（当前病例完成后生效）…")

    def _append_log(self, text):
        self.log_area.moveCursor(self.log_area.textCursor().End)
        self.log_area.insertPlainText(text)
        self.log_area.moveCursor(self.log_area.textCursor().End)

    def _on_progress(self, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _on_done(self, ok, failed, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_open_out.setEnabled(bool(self.out_edit.text()) and Path(self.out_edit.text()).is_dir())
        self.statusBar().showMessage(msg)
        QMessageBox.information(
            self, "完成", f"{msg}\n成功: {ok} 例   失败: {failed} 例"
            + (f"\n输出目录: {self.out_edit.text()}" if ok else ""))

    def open_output(self):
        d = self.out_edit.text()
        if d and Path(d).is_dir():
            import os
            os.startfile(d)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(3000)
        self._save_settings()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Lung3D 分割重建")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, "Lung3D 分割重建发生错误:\n\n" + tb, "Lung3D 错误", 0x10)
        except Exception:
            pass
        print(tb)