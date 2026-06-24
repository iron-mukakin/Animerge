@echo off
setlocal

chcp 65001 >nul

cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PY=%VENV_DIR%\Scripts\python.exe"

echo ==========================================
echo Python GPU Environment Setup
echo ==========================================
echo.

REM ==================================================
REM Python version detect
REM ==================================================

set "PYTHON_CMD="

py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
)

if not defined PYTHON_CMD (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3.11"
    )
)

if not defined PYTHON_CMD (
    py -3.10 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3.10"
    )
)

if not defined PYTHON_CMD (
    echo.
    echo [ERROR] Python 3.10 - 3.12 not found
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
echo %PYTHON_CMD%
echo.

REM ==================================================
REM Create venv
REM ==================================================

set "NEEDS_INSTALL=0"
if not exist "%PY%" set "NEEDS_INSTALL=1"

if "%NEEDS_INSTALL%"=="1" (

    echo [INFO] Creating venv...

    call %PYTHON_CMD% -m venv "%VENV_DIR%"

    if %ERRORLEVEL% neq 0 (
        echo.
        echo [ERROR] venv creation failed
        echo.
        pause
        exit /b 1
    )

    echo [OK] venv created
    echo.

    REM ==================================================
    REM Upgrade pip
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
    REM Install requirements
    REM ==================================================

    echo [INFO] Installing packages...
    echo.

    "%PY%" -m pip install -r requirements.txt

    if %ERRORLEVEL% neq 0 (
        echo.
        echo [ERROR] requirements install failed
        echo.
        pause
        exit /b 1
    )

) else (
    echo [OK] venv already exists, skipping install
)

echo.
echo ==========================================
echo Setup Complete
echo ==========================================
echo.

"%PY%" -c "import torch; print(torch.__version__)"
"%PY%" -c "import torch; print(torch.cuda.is_available())"

echo.

REM ==================================================
REM Start Application
REM ==================================================

echo.
echo ===== Starting Application =====
echo.

"%PY%" run_app.py --mode cuda

set EXIT_CODE=%ERRORLEVEL%

echo.
echo Application finished with code %EXIT_CODE%
pause

exit /b %EXIT_CODE%