"""
AI图像隐私保护系统 - 模型下载脚本

这个脚本用于自动下载项目所需的所有模型权重，包括：
- Stable Diffusion 2.0 Base模型
- ArcFace面部识别模型
- FaceNet面部识别模型
- 其他必要的预训练模型

作者: AI Privacy Protection Team
创建时间: 2025-01-28
版本: 1.0.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import shutil
from tqdm import tqdm
import requests

# 添加src到path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    import torch
    from diffusers import StableDiffusionPipeline
    import insightface
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError as e:
    print(f"缺少必要的依赖: {e}")
    print("请先运行环境搭建脚本: setup_environment.bat")
    print("或运行: pip install onnxruntime")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """模型下载器"""
    
    def __init__(self, base_dir: str = "checkpoints"):
        """
        初始化模型下载器
        
        Args:
            base_dir: 模型存储基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 模型配置
        self.models_config = {
            "stable_diffusion": {
                "model_id": "stabilityai/stable-diffusion-2-base",
                "local_dir": self.base_dir / "sd2",
                "description": "Stable Diffusion 2.0 Base模型"
            },
            "stable_diffusion_vae": {
                "model_id": "stabilityai/sd-vae-ft-mse",
                "local_dir": self.base_dir / "sd2" / "vae",
                "description": "Stable Diffusion VAE模型"
            }
        }
        
        # 面部识别模型配置
        self.face_models_config = {
            "arcface": {
                "model_name": "buffalo_l",
                "local_dir": self.base_dir / "face_models" / "arcface",
                "description": "ArcFace面部识别模型"
            }
        }
    
    def check_disk_space(self, required_gb: float = 20.0) -> bool:
        """
        检查磁盘空间
        
        Args:
            required_gb: 需要的磁盘空间(GB)
            
        Returns:
            是否有足够空间
        """
        try:
            total, used, free = shutil.disk_usage(self.base_dir.parent)
            free_gb = free / (1024**3)
            
            logger.info(f"可用磁盘空间: {free_gb:.2f}GB")
            
            if free_gb < required_gb:
                logger.error(f"磁盘空间不足，需要至少{required_gb}GB，当前可用{free_gb:.2f}GB")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"无法检查磁盘空间: {e}")
            return True  # 假设有足够空间
    
    def download_stable_diffusion(self) -> bool:
        """
        下载Stable Diffusion 2.0模型
        
        Returns:
            是否下载成功
        """
        try:
            config = self.models_config["stable_diffusion"]
            logger.info(f"开始下载: {config['description']}")
            
            # 检查是否已存在
            if (config["local_dir"] / "model_index.json").exists():
                logger.info("Stable Diffusion模型已存在，跳过下载")
                return True
            
            # 创建目录
            config["local_dir"].mkdir(parents=True, exist_ok=True)
            
            # 下载模型
            logger.info(f"正在下载模型到: {config['local_dir']}")
            snapshot_download(
                repo_id=config["model_id"],
                local_dir=str(config["local_dir"]),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.safetensors"]  # 跳过大的safetensors文件，优先下载必要文件
            )
            
            logger.info("Stable Diffusion模型下载完成")
            return True
            
        except Exception as e:
            logger.error(f"下载Stable Diffusion模型失败: {e}")
            return False
    
    def download_vae_model(self) -> bool:
        """
        下载VAE模型
        
        Returns:
            是否下载成功
        """
        try:
            config = self.models_config["stable_diffusion_vae"]
            logger.info(f"开始下载: {config['description']}")
            
            # 检查是否已存在
            if (config["local_dir"] / "config.json").exists():
                logger.info("VAE模型已存在，跳过下载")
                return True
            
            # 创建目录
            config["local_dir"].mkdir(parents=True, exist_ok=True)
            
            # 下载VAE模型
            snapshot_download(
                repo_id=config["model_id"],
                local_dir=str(config["local_dir"]),
                local_dir_use_symlinks=False
            )
            
            logger.info("VAE模型下载完成")
            return True
            
        except Exception as e:
            logger.error(f"下载VAE模型失败: {e}")
            return False
    
    def download_arcface_model(self) -> bool:
        """
        下载ArcFace模型
        
        Returns:
            是否下载成功
        """
        try:
            config = self.face_models_config["arcface"]
            logger.info(f"开始下载: {config['description']}")
            
            # 创建目录
            config["local_dir"].mkdir(parents=True, exist_ok=True)
            
            # 使用insightface下载模型
            app = insightface.app.FaceAnalysis(
                name=config["model_name"],
                root=str(config["local_dir"]),
                providers=['CPUExecutionProvider']  # 使用CPU避免GPU内存问题
            )
            app.prepare(ctx_id=-1)  # CPU模式
            
            logger.info("ArcFace模型下载完成")
            return True
            
        except Exception as e:
            logger.error(f"下载ArcFace模型失败: {e}")
            logger.info("ArcFace模型可能需要手动下载，请参考文档")
            return False
    
    def download_facenet_model(self) -> bool:
        """
        下载FaceNet模型
        
        Returns:
            是否下载成功
        """
        try:
            logger.info("开始下载FaceNet模型")
            
            # FaceNet模型由facenet-pytorch库自动下载
            from facenet_pytorch import MTCNN, InceptionResnetV1
            
            # 创建目录
            facenet_dir = self.base_dir / "face_models" / "facenet"
            facenet_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化模型（会自动下载权重）
            device = torch.device('cpu')  # 使用CPU避免GPU内存问题
            
            # 下载MTCNN（人脸检测）
            mtcnn = MTCNN(device=device)
            
            # 下载InceptionResnetV1（特征提取）
            resnet = InceptionResnetV1(pretrained='vggface2', device=device)
            
            logger.info("FaceNet模型下载完成")
            return True
            
        except Exception as e:
            logger.error(f"下载FaceNet模型失败: {e}")
            return False
    
    def verify_models(self) -> Dict[str, bool]:
        """
        验证下载的模型
        
        Returns:
            模型验证结果
        """
        results = {}
        
        # 验证Stable Diffusion
        sd_path = self.base_dir / "sd2"
        results["stable_diffusion"] = (sd_path / "model_index.json").exists()
        
        # 验证VAE
        vae_path = self.base_dir / "sd2" / "vae"
        results["vae"] = (vae_path / "config.json").exists()
        
        # 验证ArcFace
        arcface_path = self.base_dir / "face_models" / "arcface"
        results["arcface"] = arcface_path.exists() and len(list(arcface_path.glob("*"))) > 0
        
        # 验证FaceNet
        facenet_path = self.base_dir / "face_models" / "facenet"
        results["facenet"] = facenet_path.exists()
        
        return results
    
    def test_model_loading(self) -> Dict[str, bool]:
        """
        测试模型加载
        
        Returns:
            模型加载测试结果
        """
        results = {}
        
        # 测试Stable Diffusion加载
        try:
            sd_path = self.base_dir / "sd2"
            if sd_path.exists():
                pipe = StableDiffusionPipeline.from_pretrained(
                    str(sd_path),
                    torch_dtype=torch.float32,  # 使用float32避免半精度问题
                    safety_checker=None,
                    requires_safety_checker=False
                )
                results["stable_diffusion_loading"] = True
                logger.info("Stable Diffusion模型加载测试通过")
                del pipe  # 清理内存
            else:
                results["stable_diffusion_loading"] = False
        except Exception as e:
            logger.error(f"Stable Diffusion模型加载测试失败: {e}")
            results["stable_diffusion_loading"] = False
        
        # 测试ArcFace加载
        try:
            arcface_path = self.base_dir / "face_models" / "arcface"
            if arcface_path.exists():
                app = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    root=str(arcface_path),
                    providers=['CPUExecutionProvider']
                )
                app.prepare(ctx_id=-1)
                results["arcface_loading"] = True
                logger.info("ArcFace模型加载测试通过")
            else:
                results["arcface_loading"] = False
        except Exception as e:
            logger.error(f"ArcFace模型加载测试失败: {e}")
            results["arcface_loading"] = False
        
        # 测试FaceNet加载
        try:
            from facenet_pytorch import InceptionResnetV1
            resnet = InceptionResnetV1(pretrained='vggface2', device='cpu')
            results["facenet_loading"] = True
            logger.info("FaceNet模型加载测试通过")
            del resnet  # 清理内存
        except Exception as e:
            logger.error(f"FaceNet模型加载测试失败: {e}")
            results["facenet_loading"] = False
        
        return results
    
    def download_all(self) -> bool:
        """
        下载所有模型
        
        Returns:
            是否全部下载成功
        """
        logger.info("开始下载所有模型...")
        
        # 检查磁盘空间
        if not self.check_disk_space(20.0):
            return False
        
        success_count = 0
        total_count = 4
        
        # 下载Stable Diffusion
        if self.download_stable_diffusion():
            success_count += 1
        
        # 下载VAE
        if self.download_vae_model():
            success_count += 1
        
        # 下载ArcFace
        if self.download_arcface_model():
            success_count += 1
        
        # 下载FaceNet
        if self.download_facenet_model():
            success_count += 1
        
        logger.info(f"模型下载完成: {success_count}/{total_count} 成功")
        
        # 验证模型
        verification_results = self.verify_models()
        logger.info("模型验证结果:")
        for model_name, is_valid in verification_results.items():
            status = "✓" if is_valid else "✗"
            logger.info(f"  {status} {model_name}")
        
        return success_count == total_count


def main():
    """主函数"""
    print("=" * 60)
    print("AI图像隐私保护系统 - 模型下载工具")
    print("=" * 60)
    
    # 创建下载器
    downloader = ModelDownloader()
    
    # 下载所有模型
    success = downloader.download_all()
    
    if success:
        print("\n✅ 所有模型下载完成！")
        
        # 测试模型加载
        print("\n开始测试模型加载...")
        loading_results = downloader.test_model_loading()
        
        print("模型加载测试结果:")
        for model_name, loaded in loading_results.items():
            status = "✓" if loaded else "✗"
            print(f"  {status} {model_name}")
        
        if all(loading_results.values()):
            print("\n🎉 所有模型下载并测试成功！可以开始开发了。")
        else:
            print("\n⚠️  部分模型加载测试失败，请检查错误信息。")
    else:
        print("\n❌ 部分模型下载失败，请检查网络连接和错误信息。")
        print("   你可以稍后重新运行此脚本来下载失败的模型。")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main() 