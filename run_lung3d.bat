@echo off
rem Lung CT 3D reconstruction launcher.
rem Usage: run_lung3d.bat --input <DICOM dir | NIfTI file | parent dir> --output <out dir> [--device cuda|cpu] [--fast] [--no-nodules]
chcp 65001 >nul
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    echo Please create it first:  py -3.12 -m venv .venv
    echo Then install deps:       .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo ============================================================
echo  LungCT 3D reconstruction started...
echo ============================================================
"%PY%" "%~dp0lung3d_reconstruct.py" %*
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