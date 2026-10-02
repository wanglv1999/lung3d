@echo off
rem Lung3D Docker Hub push helper (ASCII only)
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push_dockerhub.ps1"
if errorlevel 1 pause
