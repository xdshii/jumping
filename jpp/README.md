# AI图像隐私保护系统

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于深度学习的AI图像隐私保护系统，能够为人脸图像添加不可见的对抗性扰动，有效规避AI图像识别与生成模型的分析。

## ✨ 主要特性

- 🛡️ **强效隐私保护**: 基于DiffPrivate算法，提供三档保护强度
- 🎯 **高度精准**: 针对面部特征的精确优化，保护身份隐私
- 🔄 **跨模型兼容**: 支持Stable Diffusion和FLUX.1两种先进架构
- 👁️ **视觉无损**: 保持图像质量的同时实现隐私保护
- 🚀 **高性能**: 优化的GPU加速，支持RTX 5060Ti等高端显卡
- 📊 **完整评估**: 内置多模型评估框架，量化保护效果

## 🏗️ 系统架构

```
├── src/                    # 核心源代码
│   ├── models/            # 模型加载和管理
│   ├── losses/            # 损失函数实现
│   ├── optimization/      # 对抗性优化算法
│   ├── evaluation/        # 效果评估工具
│   ├── utils/             # 图像处理工具
│   └── config/            # 配置管理系统
├── experiments/           # 实验和测试
├── checkpoints/           # 模型权重存储
├── data/                  # 数据集和测试图像
├── config/                # 配置文件
├── scripts/               # 辅助脚本
└── docs/                  # 项目文档
```

## 🚀 快速开始

### 前置要求

- **硬件**: NVIDIA RTX 5060Ti (32GB) 或类似GPU
- **软件**: Anaconda/Miniconda, CUDA 11.8
- **存储**: 至少30GB可用空间
- **网络**: 稳定的网络连接 (用于下载模型)

### 一键安装

1. **克隆项目** (如果从远程仓库)
```bash
git clone <repository-url>
cd jump
```

2. **运行完整设置脚本**
```cmd
setup_complete.bat
```

这个脚本将自动完成：
- ✅ 创建conda虚拟环境 (`E:\envs\privacy_protection`)
- ✅ 安装所有Python依赖
- ✅ 创建项目目录结构
- ✅ 下载模型权重 (Stable Diffusion 2.0, ArcFace, FaceNet)
- ✅ 运行环境验证测试

**预计时间**: 30-60分钟 (取决于网络速度)

### 分步安装 (可选)

如果你更喜欢分步安装，可以依次运行：

```cmd
# 1. 设置Python环境
setup_environment.bat

# 2. 创建项目结构
create_project_structure.bat

# 3. 下载模型
conda activate E:\envs\privacy_protection
cd E:\projects\jump
python scripts\download_models.py
```

## 📖 使用指南

### 快速启动开发环境

```cmd
# 方法1: 使用快速启动脚本
start_development.bat

# 方法2: 手动启动
conda activate E:\envs\privacy_protection
cd E:\projects\jump
```

### 基础用法示例

```python
# 导入核心模块
from src.config.config import load_default_config
from src.utils.image_utils import load_image, save_image
from src.models.sd_loader import StableDiffusionLoader

# 加载配置
config = load_default_config()

# 加载图像
image = load_image("path/to/your/image.jpg")

# TODO: 添加隐私保护处理代码
# protected_image = privacy_protector.protect(image, strength="medium")

# 保存结果
# save_image(protected_image, "path/to/protected_image.jpg")
```

### 配置系统

项目使用YAML配置文件管理参数：

```python
from src.config.config import ConfigManager, ProtectionStrength

# 创建配置管理器
config_manager = ConfigManager()

# 加载默认配置
config = config_manager.load_config("default.yaml")

# 更新保护强度
config.update_for_strength(ProtectionStrength.STRONG)

# 保存自定义配置
config_manager.save_config(config, "my_config.yaml")
```

## 🧪 测试验证

### 环境验证

```python
# 测试Python环境
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 测试核心模块
python src/config/config.py        # 配置系统测试
python src/utils/image_utils.py    # 图像处理测试
```

### 模型验证

```python
# 重新下载模型 (如有需要)
python scripts/download_models.py

# 验证模型加载
python -c "from diffusers import StableDiffusionPipeline; print('✓ Diffusers正常')"
```

## 📊 开发进度

### 阶段一：基础环境搭建 ✅

- [x] 创建conda虚拟环境配置
- [x] 实现配置管理系统
- [x] 实现图像预处理工具
- [x] 创建模型下载脚本
- [x] 完成环境验证测试

### 阶段二：Stable Diffusion算法实现 🔄

- [ ] 实现Stable Diffusion模型加载器
- [ ] 实现DDIM反向采样函数
- [ ] 实现无条件嵌入优化
- [ ] 创建注意力控制机制
- [ ] 实现核心损失函数
- [ ] 创建主优化循环

### 阶段三：FLUX.1架构迁移 ⏳

- [ ] 下载FLUX.1模型权重
- [ ] 实现FLUX.1模型加载器
- [ ] 适配Flow Matching算法
- [ ] 迁移注意力控制机制
- [ ] 调优损失函数权重

### 阶段四：产品化界面 ⏳

- [ ] 选择UI技术栈 (Gradio/Streamlit)
- [ ] 实现图像上传和预览
- [ ] 创建保护强度选择界面
- [ ] 集成实时进度显示
- [ ] 添加结果对比功能

## 🛠️ 开发工具

### 可用脚本

```cmd
# 环境管理
setup_complete.bat              # 完整环境设置
start_development.bat           # 快速启动开发环境

# 模型管理
python scripts/download_models.py     # 下载所有模型
python scripts/verify_models.py       # 验证模型完整性

# 测试工具
python src/config/config.py           # 配置系统测试
python src/utils/image_utils.py       # 图像工具测试
```

### 项目配置

主要配置文件在 `config/default.yaml`:

```yaml
optimization:
  learning_rate: 0.01
  max_iterations: 250
  strength_weights:
    light: {identity: 0.5, lpips: 1.0, max_iterations: 100}
    medium: {identity: 1.0, lpips: 0.6, max_iterations: 250}
    strong: {identity: 1.5, lpips: 0.4, max_iterations: 350}

model:
  stable_diffusion_path: checkpoints/sd2
  face_models_path: checkpoints/face_models

system:
  device: cuda:0
  use_fp16: true
  image_size: 512
```

## 📚 技术文档

详细的技术文档请参考 `docs/` 目录：

- 📋 [项目章程](docs/project_charter.md) - 项目目标和成功标准
- 🏗️ [技术设计文档](docs/technical_design_document.md) - 详细技术实现方案
- 🗺️ [产品路线图](docs/product_roadmap.md) - 完整的产品演进计划
- 🔧 [开发环境配置](docs/dev_environment_setup.md) - 环境搭建详细指南
- 📖 [代码规范](docs/coding_standards.md) - 编程规范和最佳实践
- 🧪 [实验日志](docs/research_experiment_log.md) - 研究实验记录

## 🐛 故障排除

### 常见问题

**1. conda命令不可用**
```cmd
# 确保Anaconda/Miniconda已正确安装并添加到PATH
# 重新打开命令提示符或重启系统
```

**2. CUDA相关错误**
```cmd
# 检查CUDA安装
nvcc --version

# 如果没有CUDA，系统会自动切换到CPU模式
# 建议安装CUDA 11.8以获得最佳性能
```

**3. 模型下载失败**
```cmd
# 检查网络连接
# 可以稍后重新运行下载脚本
python scripts/download_models.py
```

**4. 内存不足错误**
```python
# 在配置中调整批处理大小
config.system.max_batch_size = 1
config.system.low_mem = True
```

### 获取帮助

如果遇到问题：

1. 📋 检查 `logs/` 目录中的日志文件
2. 🔍 查看相关的技术文档
3. 🧪 运行验证脚本确认环境
4. 💡 参考故障排除部分

## 🤝 贡献指南

欢迎贡献代码和改进建议！请遵循以下流程：

1. Fork项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 代码规范

请遵循 [代码规范文档](docs/coding_standards.md) 中的编程标准。

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [DiffPrivate](https://github.com/diffprivate/diffprivate) - 核心算法参考
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) - 基础模型架构
- [FLUX.1](https://github.com/black-forest-labs/flux) - 先进生成模型
- [Hugging Face](https://huggingface.co/) - 模型托管和工具库

## 📞 联系方式

- 📧 项目邮箱: [your-email@example.com]
- 🐛 问题反馈: [GitHub Issues]
- 📖 技术讨论: [GitHub Discussions]

---

**最后更新**: 2025年1月28日  
**项目状态**: 🔄 开发中  
**当前版本**: v0.1.0-alpha 