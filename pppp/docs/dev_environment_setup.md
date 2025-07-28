# AI图像隐私保护系统开发环境配置指南
# Development Environment Setup Guide

**版本**: v1.0  
**创建日期**: 2025年1月28日  
**最后更新**: 2025年1月28日  
**适用系统**: Windows 10+, Ubuntu 20.04+  

---

## 概述 (Overview)

本指南将帮助您搭建AI图像隐私保护系统的完整开发环境，包括必要的软件安装、模型下载、依赖配置等。请按照步骤顺序执行，确保每个步骤都成功完成。

## 硬件要求 (Hardware Requirements)

### 最低配置 (Minimum Requirements)
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) 或更高
- **CPU**: Intel i5-10400 或 AMD Ryzen 5 3600 或更高
- **内存**: 16GB RAM 
- **存储**: 100GB 可用空间 (SSD推荐)

### 推荐配置 (Recommended Requirements)
- **GPU**: NVIDIA RTX 5060Ti (32GB VRAM) 或 RTX 4090 (24GB VRAM)
- **CPU**: Intel i7-12700K 或 AMD Ryzen 7 5800X 或更高  
- **内存**: 32GB RAM
- **存储**: 500GB SSD可用空间

### 云端开发配置 (Cloud Development)
适用于FLUX.1开发阶段：
- **AWS**: p3.2xlarge (Tesla V100 16GB) 或 p4d.xlarge (A100 40GB)
- **Google Cloud**: n1-standard-8 + Tesla V100 或 a2-highgpu-1g
- **Azure**: Standard_NC6s_v3 (Tesla V100) 或 Standard_ND40rs_v2

---

## 第一步：基础环境搭建 (Step 1: Basic Environment Setup)

### 1.1 安装Python 3.10

#### Windows系统
1. 下载Python 3.10.x从 [python.org](https://www.python.org/downloads/)
2. 运行安装程序，**务必勾选"Add Python to PATH"**
3. 验证安装：
```cmd
python --version
# 应显示: Python 3.10.x
pip --version
# 应显示: pip 22.x.x 或更高版本
```

#### Ubuntu/Linux系统
```bash
# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装Python 3.10
sudo apt install python3.10 python3.10-dev python3.10-venv python3-pip -y

# 设置Python 3.10为默认
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 验证安装
python --version
pip --version
```

### 1.2 安装CUDA Toolkit

#### Windows系统
1. 访问 [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads)
2. 选择：Windows → x86_64 → 11.8 → exe (local)
3. 下载并运行安装程序 (约3GB)
4. 安装完成后重启系统
5. 验证安装：
```cmd
nvcc --version
nvidia-smi
```

#### Ubuntu/Linux系统
```bash
# 下载CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8

# 设置环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
nvidia-smi
```

### 1.3 安装Git
```bash
# Windows: 下载并安装 Git for Windows
# Ubuntu/Linux:
sudo apt install git -y

# 配置Git (替换为您的信息)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 第二步：项目环境搭建 (Step 2: Project Environment Setup)

### 2.1 克隆项目代码库

```bash
# 创建项目目录
mkdir -p E:/projects/jump  # Windows
mkdir -p ~/projects/jump   # Linux

cd E:/projects/jump  # Windows
cd ~/projects/jump   # Linux

# 初始化git仓库 (如果还没有远程仓库)
git init
```

### 2.2 创建虚拟环境

#### 使用venv (推荐)
```bash
# 创建虚拟环境
python -m venv venv_privacy

# 激活虚拟环境
# Windows:
venv_privacy\Scripts\activate
# Linux:
source venv_privacy/bin/activate

# 确认激活成功
which python  # Linux
where python   # Windows
# 应该显示虚拟环境中的python路径
```

#### 使用conda (可选)
```bash
# 安装Miniconda (如果没有)
# Windows: 下载并安装 Miniconda3
# Linux: 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建conda环境
conda create -n privacy_protection python=3.10 -y
conda activate privacy_protection
```

### 2.3 安装Python依赖

创建 `requirements.txt` 文件：
```bash
# 创建requirements.txt文件
cat > requirements.txt << 'EOF'
# 深度学习框架
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
--index-url https://download.pytorch.org/whl/cu118

# 扩散模型库
diffusers==0.21.4
transformers==4.33.2
accelerate==0.21.0

# 图像处理
Pillow==10.0.0
opencv-python==4.8.0.76
imageio==2.31.1

# 科学计算
numpy==1.24.3
scipy==1.11.1

# 感知损失和评估
lpips==0.1.4
pytorch-fid==0.3.0

# 面部识别模型
insightface==0.7.3
facenet-pytorch==2.5.3

# 实验管理
wandb==0.15.8
tensorboard==2.13.0

# 配置管理
omegaconf==2.3.0
hydra-core==1.3.2

# 用户界面
gradio==3.41.2
streamlit==1.25.0

# 工具库
tqdm==4.65.0
matplotlib==3.7.2
seaborn==0.12.2
pandas==2.0.3

# 开发工具
pytest==7.4.0
black==23.7.0
flake8==6.0.0
jupyter==1.0.0
EOF
```

安装依赖：
```bash
# 升级pip
pip install --upgrade pip

# 安装PyTorch (CUDA 11.8版本)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 验证PyTorch CUDA安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 安装其他依赖
pip install -r requirements.txt
```

### 2.4 验证环境安装

创建验证脚本 `test_environment.py`：
```python
#!/usr/bin/env python3
"""
环境验证脚本
验证所有关键依赖是否正确安装
"""

import sys
import torch
import numpy as np
from PIL import Image
import cv2

def test_python_version():
    """测试Python版本"""
    print(f"✓ Python版本: {sys.version}")
    assert sys.version_info >= (3, 10), "Python版本必须≥3.10"

def test_pytorch():
    """测试PyTorch安装"""
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"✓ CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory//1024//1024}MB)")
    else:
        print("⚠ 警告: CUDA不可用，将使用CPU运行")

def test_libraries():
    """测试关键库"""
    try:
        import diffusers
        print(f"✓ Diffusers版本: {diffusers.__version__}")
    except ImportError as e:
        print(f"✗ Diffusers导入失败: {e}")
    
    try:
        import transformers
        print(f"✓ Transformers版本: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers导入失败: {e}")
    
    try:
        import lpips
        print("✓ LPIPS导入成功")
    except ImportError as e:
        print(f"✗ LPIPS导入失败: {e}")

def test_image_processing():
    """测试图像处理功能"""
    try:
        # 创建测试图像
        test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        pil_img = Image.fromarray(test_img)
        
        # 测试PIL
        resized = pil_img.resize((256, 256))
        print("✓ PIL图像处理正常")
        
        # 测试OpenCV
        cv_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        cv_resized = cv2.resize(cv_img, (256, 256))
        print("✓ OpenCV图像处理正常")
        
        # 测试PyTorch张量转换
        tensor = torch.from_numpy(test_img).permute(2, 0, 1).float() / 255.0
        print(f"✓ PyTorch张量转换正常: {tensor.shape}")
        
    except Exception as e:
        print(f"✗ 图像处理测试失败: {e}")

def test_gpu_memory():
    """测试GPU内存"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        
        # 测试内存分配
        try:
            # 分配1GB内存进行测试
            test_tensor = torch.randn(1024, 1024, 256, device=device)
            print(f"✓ GPU内存测试通过")
            
            # 显示内存使用情况
            allocated = torch.cuda.memory_allocated(device) // 1024 // 1024
            cached = torch.cuda.memory_reserved(device) // 1024 // 1024
            print(f"  已分配内存: {allocated}MB")
            print(f"  缓存内存: {cached}MB")
            
            # 清理内存
            del test_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ GPU内存测试失败: {e}")
    else:
        print("⚠ GPU不可用，跳过内存测试")

def main():
    """主测试函数"""
    print("=" * 50)
    print("AI图像隐私保护系统 - 环境验证")
    print("=" * 50)
    
    try:
        test_python_version()
        test_pytorch()
        test_libraries()
        test_image_processing()
        test_gpu_memory()
        
        print("\n" + "=" * 50)
        print("✅ 环境验证通过！可以开始开发。")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 环境验证失败: {e}")
        print("请检查安装步骤并重新配置环境。")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

运行验证：
```bash
python test_environment.py
```

---

## 第三步：模型下载配置 (Step 3: Model Download Setup)

### 3.1 创建模型目录结构

```bash
# 创建模型存储目录
mkdir -p checkpoints/stable_diffusion
mkdir -p checkpoints/flux
mkdir -p checkpoints/face_models
mkdir -p checkpoints/evaluation_models

# 创建数据目录
mkdir -p data/test_images
mkdir -p data/results
mkdir -p data/benchmarks
```

### 3.2 Hugging Face配置

```bash
# 安装Hugging Face Hub
pip install huggingface_hub

# 登录Hugging Face (可选，用于访问受限模型)
# huggingface-cli login
```

### 3.3 下载Stable Diffusion模型

创建模型下载脚本 `download_models.py`：
```python
#!/usr/bin/env python3
"""
模型下载脚本
自动下载所有必需的模型权重
"""

import os
from huggingface_hub import snapshot_download
import torch

def download_stable_diffusion():
    """下载Stable Diffusion 2.0模型"""
    print("正在下载Stable Diffusion 2.0...")
    
    model_path = "./checkpoints/stable_diffusion"
    
    try:
        snapshot_download(
            repo_id="stabilityai/stable-diffusion-2-base",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        print("✓ Stable Diffusion 2.0下载完成")
    except Exception as e:
        print(f"✗ Stable Diffusion下载失败: {e}")

def download_face_models():
    """下载面部识别模型"""
    print("正在下载面部识别模型...")
    
    # 这里添加具体的面部识别模型下载逻辑
    # 由于insightface模型通常需要特殊处理，暂时跳过
    print("⚠ 面部识别模型需要手动配置")

def check_disk_space():
    """检查磁盘空间"""
    import shutil
    
    total, used, free = shutil.disk_usage("./")
    free_gb = free // (1024**3)
    
    print(f"可用磁盘空间: {free_gb}GB")
    
    if free_gb < 50:
        print("⚠ 警告: 磁盘空间不足50GB，可能影响模型下载")
        return False
    return True

def main():
    """主函数"""
    print("开始下载模型...")
    
    if not check_disk_space():
        print("请清理磁盘空间后重试")
        return
    
    # 创建目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # 下载模型
    download_stable_diffusion()
    download_face_models()
    
    print("模型下载完成！")

if __name__ == "__main__":
    main()
```

### 3.4 配置模型路径

创建配置文件 `config/model_paths.yaml`：
```yaml
# 模型路径配置
models:
  stable_diffusion:
    path: "./checkpoints/stable_diffusion"
    model_id: "stabilityai/stable-diffusion-2-base"
    
  flux:
    path: "./checkpoints/flux"  
    model_id: "black-forest-labs/FLUX.1-dev"
    
  face_recognition:
    arcface:
      path: "./checkpoints/face_models/arcface"
    facenet:
      path: "./checkpoints/face_models/facenet"
    
# 设备配置
device:
  primary: "cuda:0"  # 主GPU
  fallback: "cpu"    # 后备设备
  
# 内存配置  
memory:
  max_batch_size: 1
  use_fp16: true
  gradient_checkpointing: true
```

---

## 第四步：开发工具配置 (Step 4: Development Tools Setup)

### 4.1 IDE配置推荐

#### Visual Studio Code (推荐)
1. 安装VSCode
2. 安装必要插件：
   - Python
   - Pylance
   - Jupyter
   - Python Docstring Generator
   - GitLens
   - Better Comments

VSCode配置文件 `.vscode/settings.json`：
```json
{
    "python.defaultInterpreterPath": "./venv_privacy/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "editor.formatOnSave": true,
    "editor.rulers": [88],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/checkpoints": true
    }
}
```

#### PyCharm (专业版)
- 配置Python解释器指向虚拟环境
- 启用代码检查和格式化
- 配置GPU调试支持

### 4.2 Jupyter Notebook配置

```bash
# 安装Jupyter
pip install jupyter jupyterlab

# 安装Jupyter插件
pip install ipywidgets jupyter-widgets-extension

# 启动Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### 4.3 实验管理工具

#### Weights & Biases
```bash
# 安装wandb
pip install wandb

# 登录wandb
wandb login
```

#### TensorBoard
```bash
# 启动TensorBoard
tensorboard --logdir=./logs --port=6006
```

---

## 第五步：项目结构创建 (Step 5: Project Structure Setup)

创建标准项目结构：
```bash
# 创建完整项目结构
mkdir -p src/{models,losses,optimization,evaluation,utils,config}
mkdir -p experiments
mkdir -p tests
mkdir -p docs
mkdir -p scripts
mkdir -p configs

# 创建__init__.py文件
touch src/__init__.py
touch src/models/__init__.py
touch src/losses/__init__.py
touch src/optimization/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py
touch src/config/__init__.py
```

创建项目根目录下的 `setup.py`：
```python
from setuptools import setup, find_packages

setup(
    name="ai-privacy-protection",
    version="0.1.0",
    description="AI Image Privacy Protection System",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.21.0",
        "transformers>=4.33.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
```

---

## 第六步：测试开发环境 (Step 6: Test Development Environment)

### 6.1 运行基础测试

创建 `test_basic_functionality.py`：
```python
#!/usr/bin/env python3
"""
基础功能测试
验证开发环境的核心功能
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import os

def test_stable_diffusion_loading():
    """测试Stable Diffusion模型加载"""
    print("测试Stable Diffusion模型加载...")
    
    model_path = "./checkpoints/stable_diffusion"
    
    if not os.path.exists(model_path):
        print("⚠ 模型路径不存在，跳过测试")
        return
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda:0")
        
        print("✓ Stable Diffusion模型加载成功")
        
        # 清理内存
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")

def test_image_processing():
    """测试图像处理流水线"""
    print("测试图像处理流水线...")
    
    try:
        # 创建测试图像
        test_image = Image.new('RGB', (512, 512), color='red')
        
        # PIL处理
        resized = test_image.resize((256, 256))
        
        # 转换为tensor
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
        
        # 添加batch维度
        batch_tensor = tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
        
        print(f"✓ 图像处理成功: {batch_tensor.shape}")
        
    except Exception as e:
        print(f"✗ 图像处理失败: {e}")

def main():
    """主测试函数"""
    print("=" * 50)
    print("基础功能测试")
    print("=" * 50)
    
    test_stable_diffusion_loading()
    test_image_processing()
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
```

运行测试：
```bash
python test_basic_functionality.py
```

---

## 故障排除 (Troubleshooting)

### 常见问题及解决方案

#### 1. CUDA相关错误
```bash
# 问题：RuntimeError: CUDA out of memory
# 解决：减少batch_size或使用CPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 问题：CUDA版本不匹配
# 解决：重新安装对应版本的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 模型下载问题
```bash
# 问题：网络连接超时
# 解决：使用镜像源或VPN
export HF_ENDPOINT=https://hf-mirror.com

# 问题：磁盘空间不足
# 解决：清理无用文件或使用软链接
ln -s /path/to/large/storage/checkpoints ./checkpoints
```

#### 3. 依赖冲突
```bash
# 创建全新环境
conda deactivate
conda env remove -n privacy_protection
conda create -n privacy_protection python=3.10 -y
conda activate privacy_protection
```

### 性能优化建议

1. **启用混合精度训练**：
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

2. **使用DataLoader优化**：
```python
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)
```

3. **GPU内存优化**：
```python
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

---

## 下一步 (Next Steps)

环境配置完成后，您可以：

1. **运行环境验证脚本**确保一切正常
2. **下载必要的模型权重**
3. **开始第一个开发任务**：复现DiffPrivate算法
4. **设置代码版本控制**和协作流程

---

## 附录 (Appendix)

### A. 完整的requirements.txt
```txt
# 深度学习框架
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2

# 扩散模型
diffusers==0.21.4
transformers==4.33.2
accelerate==0.21.0

# 图像处理
Pillow==10.0.0
opencv-python==4.8.0.76
imageio==2.31.1

# 科学计算
numpy==1.24.3
scipy==1.11.1

# 机器学习工具
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# 感知损失和评估
lpips==0.1.4
pytorch-fid==0.3.0

# 面部识别
insightface==0.7.3
facenet-pytorch==2.5.3

# 实验管理
wandb==0.15.8
tensorboard==2.13.0

# 配置管理
omegaconf==2.3.0
hydra-core==1.3.2

# 用户界面
gradio==3.41.2
streamlit==1.25.0

# 开发工具
pytest==7.4.0
black==23.7.0
flake8==6.0.0
jupyter==1.0.0

# 其他工具
tqdm==4.65.0
pandas==2.0.3
requests==2.31.0
```

### B. 推荐的VSCode插件

- **Python** - Python语言支持
- **Pylance** - Python静态类型检查
- **Jupyter** - Jupyter Notebook支持  
- **GitLens** - Git增强工具
- **Better Comments** - 注释高亮
- **Python Docstring Generator** - 自动生成文档字符串
- **autoDocstring** - 智能文档生成
- **Python Type Hint** - 类型提示支持

### C. 有用的命令别名

将以下内容添加到 `.bashrc` 或 `.zshrc`：
```bash
# 项目快捷命令
alias activate_privacy="source ~/Projects/jump/venv_privacy/bin/activate"
alias gpu_status="nvidia-smi"
alias gpu_memory="nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
alias clean_cache="python -c 'import torch; torch.cuda.empty_cache(); print(\"GPU缓存已清理\")'"
```

---

*本开发环境配置指南将根据项目进展和新技术不断更新。如遇到问题，请参考故障排除部分或联系技术支持。* 