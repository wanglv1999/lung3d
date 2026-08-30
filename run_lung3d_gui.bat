@echo off
rem Lung3D segmentation GUI launcher.
rem Usage: run_lung3d_gui.bat
rem Optional: put the path to an existing virtualenv in venv_path.txt (one line).
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "APP=%~dp0lung3d_gui.py"

set "VENV="
if exist "%~dp0venv_path.txt" set /p VENV=<"%~dp0venv_path.txt"

set "PYW="
set "PY="
if defined VENV (
    set "PYW=%VENV%\Scripts\pythonw.exe"
    set "PY=%VENV%\Scripts\python.exe"
) else (
    set "PYW=%~dp0.venv\Scripts\pythonw.exe"
    set "PY=%~dp0.venv\Scripts\python.exe"
)

set "PYF="
if exist "%PYW%" set "PYF=%PYW%"
if not defined PYF if exist "%PY%" set "PYF=%PY%"
if not defined PYF (where pythonw.exe >nul 2>&1 && set "PYF=pythonw.exe")
if not defined PYF (where python.exe >nul 2>&1 && set "PYF=python.exe")

if not defined PYF (
    echo [ERROR] No Python interpreter found.
    echo Create a venv first:  py -3.12 -m venv .venv
    echo Install deps:         .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo Or place the venv path into venv_path.txt next to this script.
    pause
    exit /b 1
)

"%PYF%" -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PySide6 is not installed in: "%PYF%"
    echo Run:  .venv\Scripts\python.exe -m pip install PySide6
    pause
    exit /b 1
)

echo Using interpreter: "%PYF%"
start "" "%PYF%" "%APP%" %*
exit /b 0