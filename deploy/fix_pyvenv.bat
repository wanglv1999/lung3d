@echo off
rem Fix pyvenv.cfg home to point to the bundled runtime dir (run at install time).
setlocal
set "APP=%~dp0"
set "CFG=%APP%.venv\pyvenv.cfg"
if not exist "%CFG%" exit /b 1
set "NEW="
(for /f "usebackq delims=" %%L in ("%CFG%") do (
    echo %%L | findstr /b /i "home =" >nul
    if not errorlevel 1 (
        echo home = %APP%runtime
    ) else (
        echo %%L
    )
)) > "%CFG%.tmp"
move /y "%CFG%.tmp" "%CFG%" >nul
exit /b 0