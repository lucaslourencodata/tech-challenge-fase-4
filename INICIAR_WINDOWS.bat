@echo off
chcp 65001 > nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\iniciar_windows.ps1"
if errorlevel 1 (
  echo.
  echo Ocorreu um problema. Veja as mensagens acima.
  pause
)
