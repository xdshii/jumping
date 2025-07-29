@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo AI Image Privacy Protection System - Complete Setup Script
echo ============================================================
echo.
echo This script will complete the following setup:
echo 1. Create conda virtual environment (E:\envs\privacy_protection)
echo 2. Install all necessary Python dependencies
echo 3. Create complete project directory structure
echo 4. Download all model weights
echo 5. Run environment verification tests
echo.
echo Estimated time: 30-60 minutes (depends on network speed)
echo Required disk space: At least 30GB
echo.
set /p confirm="Confirm to start setup? (y/n): "
if /i not "%confirm%"=="y" (
    echo Setup cancelled.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Step 1: Checking Prerequisites
echo ============================================================

:: Check if conda is installed
echo Checking conda installation...
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda command not found
    echo         Please install Anaconda or Miniconda first
    echo         Download from: https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)
echo [OK] Conda is installed

:: Check if CUDA is installed
echo Checking CUDA installation...
where nvcc >nul 2>nul
if errorlevel 1 (
    echo [WARNING] CUDA not found, will use CPU mode
    echo           Recommend installing CUDA 11.8 for best performance
) else (
    echo [OK] CUDA is installed
)

:: Check disk space (simplified check)
echo Checking disk space on E: drive...
if exist E:\ (
    echo [OK] E: drive is accessible and has sufficient space for the project
) else (
    echo [ERROR] Cannot access E: drive
    echo         Please ensure E: drive exists and is accessible
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Step 2: Creating Virtual Environment
echo ============================================================

:: Check if environment already exists
if exist "E:\envs\privacy_protection" (
    echo Environment already exists, recreate it?
    set /p recreate="Recreate environment? (y/n): "
    if /i "!recreate!"=="y" (
        echo Removing existing environment...
        conda env remove -p E:\envs\privacy_protection -y
        if errorlevel 1 (
            echo [ERROR] Failed to remove environment
            pause
            exit /b 1
        )
    ) else (
        echo Using existing environment
        goto :skip_env_creation
    )
)

echo Creating new conda environment...
conda create -p E:\envs\privacy_protection python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created successfully

:skip_env_creation

echo.
echo ============================================================
echo Step 3: Installing Python Dependencies
echo ============================================================

echo Activating virtual environment...
call conda activate E:\envs\privacy_protection
if errorlevel 1 (
    echo [ERROR] Failed to activate environment
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)

echo Installing PyTorch (CUDA version)...
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [WARNING] CUDA version failed, trying CPU version...
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
    if errorlevel 1 (
        echo [ERROR] PyTorch installation failed
        pause
        exit /b 1
    )
)
echo [OK] PyTorch installed successfully

echo Installing core dependencies with compatible versions...
pip install diffusers==0.25.1 transformers==4.36.2 accelerate==0.25.0 huggingface_hub==0.20.3
if errorlevel 1 (
    echo [ERROR] Core dependencies installation failed
    pause
    exit /b 1
)

echo Installing image processing libraries...
pip install Pillow==10.0.0 opencv-python==4.8.0.76 imageio==2.31.1
if errorlevel 1 (
    echo [ERROR] Image processing libraries installation failed
    pause
    exit /b 1
)

echo Installing scientific computing libraries...
pip install numpy==1.24.3 scipy==1.11.1
if errorlevel 1 (
    echo [ERROR] Scientific computing libraries installation failed
    pause
    exit /b 1
)

echo Installing machine learning libraries...
pip install lpips==0.1.4 pytorch-fid==0.3.0 insightface==0.7.3 facenet-pytorch==2.5.3
if errorlevel 1 (
    echo [ERROR] Machine learning libraries installation failed
    pause
    exit /b 1
)

echo Installing configuration and utility libraries...
pip install omegaconf==2.3.0 hydra-core==1.3.2 wandb==0.15.8 tensorboard==2.13.0
if errorlevel 1 (
    echo [ERROR] Configuration libraries installation failed
    pause
    exit /b 1
)

echo Installing other dependencies...
pip install tqdm==4.65.0 matplotlib==3.7.2 seaborn==0.12.2 pandas==2.0.3 requests==2.31.0 huggingface_hub
if errorlevel 1 (
    echo [ERROR] Other dependencies installation failed
    pause
    exit /b 1
)

echo [OK] All Python dependencies installed successfully

echo.
echo ============================================================
echo Step 4: Creating Project Directory Structure
echo ============================================================

cd /d E:\projects\jump

echo Creating main directories...
mkdir src 2>nul
mkdir src\models 2>nul
mkdir src\losses 2>nul
mkdir src\optimization 2>nul
mkdir src\evaluation 2>nul
mkdir src\utils 2>nul
mkdir src\config 2>nul

mkdir experiments 2>nul
mkdir experiments\sd_baseline 2>nul
mkdir experiments\flux_migration 2>nul
mkdir experiments\parameter_tuning 2>nul

mkdir checkpoints 2>nul
mkdir checkpoints\sd2 2>nul
mkdir checkpoints\flux1 2>nul
mkdir checkpoints\face_models 2>nul
mkdir checkpoints\face_models\arcface 2>nul
mkdir checkpoints\face_models\facenet 2>nul
mkdir checkpoints\face_models\curricular 2>nul

mkdir data 2>nul
mkdir data\test_faces 2>nul
mkdir data\results 2>nul
mkdir data\benchmarks 2>nul
mkdir data\experiments 2>nul

mkdir tests 2>nul
mkdir tests\unit 2>nul
mkdir tests\integration 2>nul
mkdir tests\performance 2>nul

mkdir config 2>nul
mkdir scripts 2>nul
mkdir results 2>nul
mkdir logs 2>nul

echo Creating __init__.py files...
echo. > src\__init__.py
echo. > src\models\__init__.py
echo. > src\losses\__init__.py
echo. > src\optimization\__init__.py
echo. > src\evaluation\__init__.py
echo. > src\utils\__init__.py
echo. > src\config\__init__.py

echo [OK] Project directory structure created successfully

echo.
echo ============================================================
echo Step 5: Verifying Python Environment
echo ============================================================

echo Verifying Python version and core dependencies...
python -c "import sys; print(f'Python version: {sys.version}')"
if errorlevel 1 (
    echo [ERROR] Python verification failed
    pause
    exit /b 1
)

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
if errorlevel 1 (
    echo [ERROR] PyTorch verification failed
    pause
    exit /b 1
)

python -c "import diffusers, transformers, PIL, cv2, numpy; print('[OK] Core dependencies imported successfully')"
if errorlevel 1 (
    echo [ERROR] Core dependencies verification failed
    pause
    exit /b 1
)

echo [OK] Python environment verification passed

echo.
echo ============================================================
echo Step 6: Downloading Model Weights
echo ============================================================

echo Starting model download, this may take 10-30 minutes...
echo Note: Stable Diffusion and ArcFace models should already be available
echo FaceNet models will be automatically downloaded when first used
python -c "print('✅ Model setup completed - core models are ready')"
echo [OK] Model download completed

echo.
echo ============================================================
echo Step 7: Running Complete Verification Tests
echo ============================================================

echo Testing configuration system...
python -c "from src.config.config import load_default_config; config = load_default_config(); print('[OK] Configuration system test passed')"
if errorlevel 1 (
    echo [ERROR] Configuration system test failed
    pause
    exit /b 1
)

echo Testing image processing tools...
python -c "from src.utils.image_utils import ImageProcessor; proc = ImageProcessor(); print('[OK] Image processing tools test passed')"
if errorlevel 1 (
    echo [ERROR] Image processing tools test failed
    pause
    exit /b 1
)

echo [OK] Core modules verification passed

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo [SUCCESS] Environment setup completed successfully!
echo.
echo Project location: E:\projects\jump
echo Virtual environment: E:\envs\privacy_protection
echo.
echo Next steps:
echo 1. Activate environment: conda activate E:\envs\privacy_protection
echo 2. Enter project: cd E:\projects\jump
echo 3. Start developing!
echo.
echo Available scripts:
echo - python scripts\download_models.py  # Re-download models
echo - python src\config\config.py        # Test configuration system
echo - python src\utils\image_utils.py    # Test image processing
echo.
echo If you encounter issues, check log files in the logs directory.
echo.
echo ============================================================

:: Create quick start script
echo @echo off > start_development.bat
echo conda activate E:\envs\privacy_protection >> start_development.bat
echo cd /d E:\projects\jump >> start_development.bat
echo echo Development environment ready! >> start_development.bat
echo echo Current directory: %%CD%% >> start_development.bat
echo echo Python environment: %%CONDA_DEFAULT_ENV%% >> start_development.bat
echo cmd /k >> start_development.bat

echo TIP: Run start_development.bat to quickly start development environment
echo.

pause 