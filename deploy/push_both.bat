@echo off
rem push stable + latest
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push_both.ps1"
if errorlevel 1 pause
