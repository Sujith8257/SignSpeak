@echo off
title SignSpeak (Python)
echo.
echo ============================================
echo   SignSpeak - Starting with Python
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11 and add it to PATH.
    echo See RUN-ON-ANOTHER-WINDOWS.md for instructions.
    echo.
    pause
    exit /b 1
)

REM Use venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo Using virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No venv found. Using system Python.
    echo To create venv: python -m venv venv
    echo.
)

REM Check model exists
if not exist "model\indian_sign_model.h5" (
    echo [WARNING] model\indian_sign_model.h5 not found.
    echo Copy your .h5 model file into the model\ folder.
    echo.
)

echo Starting SignSpeak...
echo.
echo When the server is ready, open:  http://localhost:5000
echo Press Ctrl+C to stop.
echo.

python app.py

pause
