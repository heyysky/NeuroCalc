@echo off
:: NeuroCalc Windows Startup Script
:: One-command launch for Windows 10/11
:: Usage: start.bat

title NeuroCalc - Handwritten Math Solver

echo.
echo  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗
echo  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗
echo  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║
echo  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║
echo  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝
echo  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
echo.
echo  CPU-Only Handwritten Math Solver
echo  ─────────────────────────────────
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: Create venv if not exists
if not exist "venv\" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    echo [SETUP] Installing dependencies (first run takes a minute)...
    venv\Scripts\pip install --upgrade pip -q
    venv\Scripts\pip install -r requirements.txt -q
    echo [SETUP] Done!
    echo.
)

:: Activate venv
call venv\Scripts\activate

:: Create models dir
if not exist "models\" mkdir models

:: Start the server
echo [START] Launching NeuroCalc server...
echo [START] Opening browser at http://localhost:8000
echo [START] Press Ctrl+C to stop
echo.

:: Open browser after short delay
start /min cmd /c "timeout /t 2 >nul && start http://localhost:8000"

:: Run backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

pause
