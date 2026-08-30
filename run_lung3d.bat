@echo off
rem Lung CT 3D reconstruction launcher (command line).
rem Usage: run_lung3d.bat --input <DICOM dir | NIfTI file | parent dir> --output <out dir> [--device cuda|cpu] [--fast] [--no-nodules]
rem Optional: put the path to an existing virtualenv in venv_path.txt (one line).
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "APP=%~dp0lung3d_reconstruct.py"

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

echo ============================================================
echo  LungCT 3D reconstruction started...
echo ============================================================
"%PY%" "%APP%" %*
set EXITCODE=%ERRORLEVEL%

echo.
if "%EXITCODE%"=="0" (
    echo [OK] Processing finished. Check the case output folder listed above.
) else (
    echo [ERROR] Processing failed, exit code %EXITCODE%.
    echo Look for "!!!" or traceback lines above for the reason.
)
echo.
echo Press any key to close this window...
pause >nul
exit /b %EXITCODE%