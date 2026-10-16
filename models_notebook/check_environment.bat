@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    pause
    exit /b 1
)

echo ============================================
echo Checking OCR environment
echo ============================================
echo.

".venv\Scripts\python.exe" -c "import sys; print('Python:', sys.version)"

".venv\Scripts\python.exe" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

".venv\Scripts\python.exe" -c "import numpy as np; print('NumPy:', np.__version__)"

".venv\Scripts\python.exe" -c "import cv2; print('OpenCV:', cv2.__version__)"

".venv\Scripts\python.exe" -c "import PIL; print('Pillow:', PIL.__version__)"

".venv\Scripts\python.exe" -c "import jupyterlab; print('JupyterLab:', jupyterlab.__version__)"

echo.
echo Available models:
echo.

dir /b /ad models

echo.
echo Environment check completed.
echo.

pause