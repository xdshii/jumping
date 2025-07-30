@echo off
chcp 65001 >nul
echo ========================================
echo AI Image Privacy Protection System - Environment Setup
echo ========================================

echo 1. Creating conda environment...
conda create -p E:\envs\privacy_protection python=3.10 -y

echo 2. Activating environment...
call conda activate E:\envs\privacy_protection

echo 3. Verifying Python version...
python --version

echo 4. Upgrading pip...
python -m pip install --upgrade pip

echo 5. Installing PyTorch with CUDA support...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo 6. Verifying PyTorch CUDA installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo 7. Installing core dependencies...
pip install diffusers==0.21.4
pip install transformers==4.33.2
pip install accelerate==0.21.0

echo 8. Installing image processing libraries...
pip install Pillow==10.0.0
pip install opencv-python==4.8.0.76
pip install imageio==2.31.1

echo 9. Installing scientific computing libraries...
pip install numpy==1.24.3
pip install scipy==1.11.1

echo 10. Installing perceptual loss and evaluation libraries...
pip install lpips==0.1.4
pip install pytorch-fid==0.3.0

echo 11. Installing face recognition libraries...
pip install insightface==0.7.3
pip install facenet-pytorch==2.5.3

echo 12. Installing experiment management tools...
pip install wandb==0.15.8
pip install tensorboard==2.13.0

echo 13. Installing configuration management...
pip install omegaconf==2.3.0
pip install hydra-core==1.3.2

echo 14. Installing development tools...
pip install pytest==7.4.0
pip install black==23.7.0
pip install flake8==6.0.0
pip install jupyter==1.0.0

echo 15. Installing other tools...
pip install tqdm==4.65.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install pandas==2.0.3
pip install requests==2.31.0

echo 16. Installing Hugging Face Hub...
pip install huggingface_hub

echo ========================================
echo Environment setup completed!
echo Please run: conda activate E:\envs\privacy_protection
echo ========================================

pause 