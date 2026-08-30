@echo off
rem Open 3D Slicer and auto-load CT + STL meshes for a case.
rem Usage:  view_in_slicer.bat [case_dir]
rem If no case_dir given, uses the current directory.
chcp 65001 >nul
set CASE=%~1
if "%CASE%"=="" set CASE=%CD%
set SCRIPTS=%~dp0
set SLICER_EXE=%LOCALAPPDATA%\slicer.org\3D Slicer 5.10.0\Slicer.exe
if not exist "%SLICER_EXE%" (
    echo [ERROR] 3D Slicer not found at "%SLICER_EXE%"
    exit /b 1
)
if not exist "%CASE%\mesh" (
    echo [ERROR] no "mesh" folder found in "%CASE%"
    exit /b 1
)
set SLICER_CASE_DIR=%CASE%
echo Opening Slicer with case: %CASE%
"%SLICER_EXE%" --python-script "%SCRIPTS%view_scene.py"