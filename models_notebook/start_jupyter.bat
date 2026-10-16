@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Environment not found.
    echo Run install_windows.bat first.
    pause
    exit /b 1
)

echo ============================================
echo OCR Student Practice
echo ============================================
echo.

echo Starting JupyterLab...
echo.

".venv\Scripts\python.exe" -m jupyterlab

if errorlevel 1 (
    echo.
    echo ERROR: JupyterLab failed to start.
    pause
)