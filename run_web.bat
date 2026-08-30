@echo off
rem Lung3D Web server launcher.
rem Usage: run_web.bat [--host 0.0.0.0] [--port 8000]
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "APP=%~dp0lung3d_api.py"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (where python.exe >nul 2>&1 && set "PY=python.exe")

if not exist "%PY%" (
    echo [ERROR] No Python interpreter found.
    echo Create a venv first:  py -3.12 -m venv .venv
    echo Install deps:         .venv\Scripts\python.exe -m pip install -r requirements-web.txt
    pause
    exit /b 1
)

start "" "http://localhost:8000"
echo [Lung3D] Web server: http://localhost:8000
echo [Lung3D] Press Ctrl+C to stop.
"%PY%" "%APP%" --host 0.0.0.0 --port 8000 --web "%~dp0web"

echo.
if errorlevel 1 (
    echo [Lung3D] Server exited with error code %errorlevel%
    pause
)
endlocal