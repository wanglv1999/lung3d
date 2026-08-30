@echo off
rem Lung3D Web server launcher (ASCII only - do not add Chinese)
setlocal
set ROOT=%~dp0
set PY=%ROOT%.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [Lung3D] venv not found: %PY%
    echo Create it first:  .venv\Scripts\python -m venv .venv  or use your installed Python
    pause
    exit /b 1
)

start "" "http://localhost:8000"
echo [Lung3D] Web server: http://localhost:8000
echo [Lung3D] Press Ctrl+C to stop.
"%PY%" "%ROOT%lung3d_api.py" --host 0.0.0.0 --port 8000 --web "%ROOT%web"

echo.
if errorlevel 1 (
    echo [Lung3D] Server exited with error code %errorlevel%
    pause
)
endlocal