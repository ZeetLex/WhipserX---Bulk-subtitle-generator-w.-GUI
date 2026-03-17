@echo off
echo ============================================
echo  WhisperX Subtitle Generator - Installer
echo ============================================
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11 from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check nvidia-smi (GPU)
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: No NVIDIA GPU detected. The tool will run on CPU which is very slow.
    echo Press Ctrl+C to cancel, or any key to continue anyway...
    pause
)

:: Get CUDA version from nvidia-smi
echo Detecting CUDA version...
for /f "tokens=9" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VER=%%i

echo Detected CUDA: %CUDA_VER%
echo.

:: Install PyTorch with correct CUDA version
echo Installing PyTorch with CUDA support...
echo (This may take a while - PyTorch is ~2.5GB)
echo.

:: Default to cu128 (works for CUDA 12.4+)
set TORCH_INDEX=https://download.pytorch.org/whl/cu128

:: Check if CUDA version starts with 12.1, 12.2, 12.3 -> use cu121
echo %CUDA_VER% | findstr /b "12.1 12.2 12.3" >nul
if not errorlevel 1 set TORCH_INDEX=https://download.pytorch.org/whl/cu121

:: Check if CUDA version starts with 11 -> use cu118
echo %CUDA_VER% | findstr /b "11." >nul
if not errorlevel 1 set TORCH_INDEX=https://download.pytorch.org/whl/cu118

echo Using PyTorch index: %TORCH_INDEX%
pip install torch==2.8.0 torchaudio==2.8.0 --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo.
    echo PyTorch install failed. Trying cu121 fallback...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo.
echo Installing remaining dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Installation complete!
echo  Run the tool with: run.bat
echo ============================================
echo.
pause
