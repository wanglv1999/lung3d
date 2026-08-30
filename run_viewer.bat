@echo off
rem Lung3D 3D segmentation model viewer launcher.
rem Usage: run_viewer.bat [case_dir | nii.gz file]
chcp 65001 >nul
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    echo Please create it first:  py -3.12 -m venv .venv
    echo Then install deps:       .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
start "" "%PY%" "%~dp0viewer3d.py" %*