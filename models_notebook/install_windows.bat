@echo off
setlocal

cd /d "%~dp0"

echo ============================================
echo OCR Student Practice - Installation
echo ============================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo Virtual environment already exists.
    goto install_packages
)

echo Looking for Python 3.12...

set "PYTHON_EXE="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py -3.12"
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; exit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=python"
        )
    )
)

if not defined PYTHON_EXE (
    echo.
    echo ERROR: Python 3.12 was not found.
    echo Install Python 3.12 from:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found: %PYTHON_EXE%

echo Creating virtual environment...

%PYTHON_EXE% -m venv .venv

if errorlevel 1 (
    echo ERROR: Cannot create virtual environment.
    pause
    exit /b 1
)

:install_packages

echo.
echo Installing packages...
echo.

".venv\Scripts\python.exe" -m pip install --upgrade pip

if errorlevel 1 (
    echo ERROR: pip update failed.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Package installation failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo Installation completed.
echo ============================================
echo.

".venv\Scripts\python.exe" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

echo.
echo Run start_jupyter.bat to open the notebook.
echo.

pause