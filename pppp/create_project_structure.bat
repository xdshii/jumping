@echo off
chcp 65001 >nul
echo ========================================
echo Creating AI Image Privacy Protection System Project Structure
echo ========================================

cd /d E:\projects\jump

echo Creating main directories...
mkdir src 2>nul
mkdir experiments 2>nul
mkdir checkpoints 2>nul
mkdir data 2>nul
mkdir tests 2>nul
mkdir config 2>nul
mkdir scripts 2>nul
mkdir results 2>nul
mkdir logs 2>nul

echo Creating src subdirectories...
mkdir src\models 2>nul
mkdir src\losses 2>nul
mkdir src\optimization 2>nul
mkdir src\evaluation 2>nul
mkdir src\utils 2>nul
mkdir src\config 2>nul

echo Creating checkpoints subdirectories...
mkdir checkpoints\sd2 2>nul
mkdir checkpoints\flux1 2>nul
mkdir checkpoints\face_models 2>nul
mkdir checkpoints\face_models\arcface 2>nul
mkdir checkpoints\face_models\facenet 2>nul
mkdir checkpoints\face_models\curricular 2>nul

echo Creating data subdirectories...
mkdir data\test_faces 2>nul
mkdir data\results 2>nul
mkdir data\benchmarks 2>nul
mkdir data\experiments 2>nul

echo Creating experiments subdirectories...
mkdir experiments\sd_baseline 2>nul
mkdir experiments\flux_migration 2>nul
mkdir experiments\parameter_tuning 2>nul

echo Creating tests subdirectories...
mkdir tests\unit 2>nul
mkdir tests\integration 2>nul
mkdir tests\performance 2>nul

echo Creating __init__.py files...
echo. > src\__init__.py
echo. > src\models\__init__.py
echo. > src\losses\__init__.py
echo. > src\optimization\__init__.py
echo. > src\evaluation\__init__.py
echo. > src\utils\__init__.py
echo. > src\config\__init__.py

echo ========================================
echo Project directory structure created successfully!
echo ========================================

dir /s /b E:\projects\jump

pause 