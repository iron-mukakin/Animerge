@echo off
setlocal

chcp 65001 >nul

cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

echo ==========================================
echo Requirements Reinstall
echo ==========================================
echo.

REM ==================================================
REM venv sonzai kakunin
REM ==================================================

if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    echo         run setup_start.bat first.
    echo.
    pause
    exit /b 1
)

echo [OK] venv found
echo.

REM ==================================================
REM requirements.txt sonzai kakunin
REM ==================================================

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found
    echo.
    pause
    exit /b 1
)

echo [OK] requirements.txt found
echo.

REM ==================================================
REM pip upgrade
REM ==================================================

echo [INFO] Upgrading pip...

"%PY%" -m pip install --upgrade pip setuptools wheel

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] pip upgrade failed
    echo.
    pause
    exit /b 1
)

echo.

REM ==================================================
REM requirements install
REM ==================================================

echo [INFO] Checking and installing requirements...
echo.

"%PY%" -m pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] requirements install failed
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Reinstall Complete
echo ==========================================
echo.

"%PY%" -c "import torch; print('torch:', torch.__version__)"
"%PY%" -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo.
pause
exit /b 0
