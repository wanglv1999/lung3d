@echo off
rem push existing image (no build)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push_existing.ps1"
if errorlevel 1 pause
