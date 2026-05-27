@echo off
setlocal

chcp 65001 >nul

cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PY=%VENV_DIR%\Scripts\python.exe"


:RUN_APP

echo.
echo ===== Starting Application =====
echo.

"%PY%" run_app.py --mode cuda

set EXIT_CODE=%ERRORLEVEL%

echo.
echo Application finished with code %EXIT_CODE%
pause

exit /b %EXIT_CODE%