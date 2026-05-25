@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

cd /d "%~dp0"

set VENV_DIR=.venv
set PY=%VENV_DIR%\Scripts\python.exe
set REQ=requirements.txt

echo.
echo ===== Python Environment Bootstrap =====
echo.

REM ------------------------------
REM 1. venv 作成
REM ------------------------------
if not exist "%PY%" (
    echo [INFO] Creating virtual environment...

    py -3 -m venv %VENV_DIR% 2>nul
    if errorlevel 1 python -m venv %VENV_DIR%

    if not exist "%PY%" (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo [INFO] Using python:
"%PY%" --version

REM ------------------------------
REM 2. pip upgrade
REM ------------------------------
echo.
echo [INFO] Upgrading pip...
"%PY%" -m pip install --upgrade pip >nul

REM ------------------------------
REM 3. requirements 存在確認
REM ------------------------------
if not exist "%REQ%" (
    echo [WARN] requirements.txt not found
    goto RUN_APP
)

REM ------------------------------
REM 4. 依存インストール判定
REM ------------------------------
echo.
echo [INFO] Checking dependencies...

"%PY%" -m pip freeze > .installed.tmp

set INSTALL_REQUIRED=0

for /f "usebackq delims=" %%i in ("%REQ%") do (

    set pkg=%%i
    set pkg=!pkg: =!

    if "!pkg!"=="" (
        rem skip
    ) else (
        findstr /i "!pkg!" .installed.tmp >nul
        if errorlevel 1 (
            set INSTALL_REQUIRED=1
        )
    )
)

del .installed.tmp

REM ------------------------------
REM 5. pip install
REM ------------------------------
if "%INSTALL_REQUIRED%"=="1" (
    echo [INFO] Installing dependencies...
    "%PY%" -m pip install -r "%REQ%"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed
        pause
        exit /b 1
    )
) else (
    echo [INFO] Dependencies already satisfied
)

REM ------------------------------
REM 6. アプリ起動
REM ------------------------------
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