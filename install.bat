@echo off
echo ============================================
echo  WhisperX Subtitle Generator - Installer
echo ============================================
echo.

:: Change to script directory so relative paths work
cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Install Python 3.11 from https://www.python.org/downloads/release/python-3110/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

python --version
echo.

:: Check NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: No NVIDIA GPU detected. Processing will run on CPU, which is very slow.
    echo Press Ctrl+C to cancel, or any key to continue anyway...
    pause
)

:: Detect CUDA version
echo Detecting CUDA version...
for /f "tokens=9" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VER=%%i
echo Detected CUDA: %CUDA_VER%
echo.

:: Select PyTorch index based on CUDA version
set TORCH_INDEX=https://download.pytorch.org/whl/cu128

echo %CUDA_VER% | findstr /b "12.1 12.2 12.3" >nul
if not errorlevel 1 set TORCH_INDEX=https://download.pytorch.org/whl/cu121

echo %CUDA_VER% | findstr /b "11." >nul
if not errorlevel 1 set TORCH_INDEX=https://download.pytorch.org/whl/cu118

echo Installing PyTorch from %TORCH_INDEX%
echo This may take a while (~2.5GB)...
echo.
pip install torch==2.8.0 torchaudio==2.8.0 --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo PyTorch 2.8.0 not available for this index. Trying latest stable...
    pip install torch torchaudio --index-url %TORCH_INDEX%
    if errorlevel 1 (
        echo Trying cu121 as fallback...
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
)

echo.
echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Installing Argos Translate language pack (en to Norwegian)...
python -c "import argostranslate.package as ap; ap.update_package_index(); pkgs = ap.get_available_packages(); pkg = next((p for p in pkgs if p.from_code=='en' and p.to_code=='nb'), None); ap.install_from_path(pkg.download()) if pkg else print('WARNING: en-nb pack not found, will fall back to Google Translate')"
if errorlevel 1 (
    echo WARNING: Argos language pack install failed. Google Translate will be used as fallback.
)

echo.
echo ============================================
echo  Installation complete!
echo  Run the tool with: run.bat
echo ============================================
echo.
pause
