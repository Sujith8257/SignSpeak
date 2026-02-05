@echo off
title SignSpeak (Docker)
echo.
echo ============================================
echo   SignSpeak - Starting with Docker
echo ============================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running.
    echo Please start "Docker Desktop" and try again.
    echo.
    pause
    exit /b 1
)

REM Check model exists
if not exist "model\indian_sign_model.h5" (
    echo [WARNING] model\indian_sign_model.h5 not found.
    echo Copy your .h5 model file into the model\ folder.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
)

echo Starting Docker containers...
echo.
echo When the server is ready, open:  http://localhost:5000
echo Press Ctrl+C to stop.
echo.

docker-compose up --build

pause
