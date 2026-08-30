@echo off
rem Lung3D cloudrun build & push helper (calls build_push.ps1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_push.ps1"
if errorlevel 1 pause