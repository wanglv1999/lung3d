@echo off
rem Lung3D 3D segmentation model viewer launcher.
rem Usage: run_viewer.bat [case_dir | nii.gz file]
rem Optional: put the path to an existing virtualenv in venv_path.txt (one line).
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "APP=%~dp0viewer3d.py"

set "VENV="
if exist "%~dp0venv_path.txt" set /p VENV=<"%~dp0venv_path.txt"

set "PY="
if defined VENV (
    set "PY=%VENV%\Scripts\python.exe"
) else (
    set "PY=%~dp0.venv\Scripts\python.exe"
)

if not exist "%PY%" (where python.exe >nul 2>&1 && set "PY=python.exe")

if not exist "%PY%" (
    echo [ERROR] No Python interpreter found.
    echo Create a venv first:  py -3.12 -m venv .venv
    echo Install deps:         .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo Or place the venv path into venv_path.txt next to this script.
    pause
    exit /b 1
)

start "" "%PY%" "%APP%" %*
exit /b 0