@echo off
rem Lung3D segmentation GUI launcher.
rem Usage: run_lung3d_gui.bat
chcp 65001 >nul
set PY=%~dp0.venv\Scripts\pythonw.exe
if not exist "%PY%" set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    echo Please create it first:  py -3.12 -m venv .venv
    echo Then install deps:       .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
start "" "%PY%" "%~dp0lung3d_gui.py" %*
exit /b 0