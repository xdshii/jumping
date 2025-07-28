"""
Stable Diffusion模型加载器

负责加载和管理Stable Diffusion 2.0模型的各个组件，包括：
- UNet模型 (去噪网络)
- VAE模型 (变分自编码器)
- Text Encoder (文本编码器)
- Tokenizer (分词器)
- Scheduler (调度器)

作者: AI Privacy Protection Team
创建时间: 2025-01-28
版本: 1.0.0
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelComponents:
    """存储Stable Diffusion模型的各个组件"""
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    scheduler: DDIMScheduler
    device: torch.device
    dtype: torch.dtype


class StableDiffusionLoader:
    """Stable Diffusion模型加载器"""
    
    def __init__(
        self,
        model_path: str = "checkpoints/sd2",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        scheduler_type: str = "ddim"
    ):
        """
        初始化模型加载器
        
        Args:
            model_path: 模型路径
            device: 运行设备 ('cuda', 'cpu', 'auto')
            dtype: 数据类型 (torch.float16, torch.float32)
            scheduler_type: 调度器类型 ('ddim', 'dpm')
        """
        self.model_path = Path(model_path)
        self.scheduler_type = scheduler_type
        
        # 自动检测设备和数据类型
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if dtype is None:
            # CUDA使用float16以节省显存，CPU使用float32
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = dtype
            
        self.components: Optional[ModelComponents] = None
        self.pipeline: Optional[StableDiffusionPipeline] = None
        
        logger.info(f"SD加载器初始化: device={self.device}, dtype={self.dtype}")
    
    def load_components(self) -> ModelComponents:
        """
        加载所有模型组件
        
        Returns:
            ModelComponents: 包含所有模型组件的数据类
        """
        if self.components is not None:
            logger.info("模型组件已加载，直接返回")
            return self.components
        
        logger.info(f"开始加载模型组件: {self.model_path}")
        
        try:
            # 检查模型路径
            if not self.model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
            
            # 加载各个组件
            logger.info("加载UNet模型...")
            unet = UNet2DConditionModel.from_pretrained(
                self.model_path / "unet",
                torch_dtype=self.dtype
            )
            
            logger.info("加载VAE模型...")
            vae = AutoencoderKL.from_pretrained(
                self.model_path / "vae",
                torch_dtype=self.dtype
            )
            
            logger.info("加载文本编码器...")
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_path / "text_encoder",
                torch_dtype=self.dtype
            )
            
            logger.info("加载分词器...")
            tokenizer = CLIPTokenizer.from_pretrained(
                self.model_path / "tokenizer"
            )
            
            logger.info("初始化调度器...")
            if self.scheduler_type == "ddim":
                scheduler = DDIMScheduler.from_pretrained(
                    self.model_path / "scheduler"
                )
            elif self.scheduler_type == "dpm":
                scheduler = DPMSolverMultistepScheduler.from_pretrained(
                    self.model_path / "scheduler"
                )
            else:
                raise ValueError(f"不支持的调度器类型: {self.scheduler_type}")
            
            # 移动模型到指定设备
            logger.info(f"移动模型到设备: {self.device}")
            unet = unet.to(self.device)
            vae = vae.to(self.device)
            text_encoder = text_encoder.to(self.device)
            
            # 创建组件对象
            self.components = ModelComponents(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                device=self.device,
                dtype=self.dtype
            )
            
            logger.info("✅ 所有模型组件加载完成")
            return self.components
            
        except Exception as e:
            logger.error(f"模型组件加载失败: {e}")
            raise
    
    def load_pipeline(self) -> StableDiffusionPipeline:
        """
        加载完整的Stable Diffusion Pipeline
        
        Returns:
            StableDiffusionPipeline: 完整的SD流水线
        """
        if self.pipeline is not None:
            logger.info("Pipeline已加载，直接返回")
            return self.pipeline
        
        logger.info("加载Stable Diffusion Pipeline...")
        
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=self.dtype,
                use_safetensors=True
            )
            self.pipeline = self.pipeline.to(self.device)
            
            # 配置调度器
            if self.scheduler_type == "ddim":
                self.pipeline.scheduler = DDIMScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            
            logger.info("✅ Stable Diffusion Pipeline加载完成")
            return self.pipeline
            
        except Exception as e:
            logger.error(f"Pipeline加载失败: {e}")
            raise
    
    def get_vae_scale_factor(self) -> float:
        """获取VAE的缩放因子"""
        if self.components is None:
            self.load_components()
        return 2 ** (len(self.components.vae.config.block_out_channels) - 1)
    
    def encode_text(self, prompt: str) -> torch.Tensor:
        """
        编码文本提示
        
        Args:
            prompt: 文本提示
            
        Returns:
            torch.Tensor: 文本嵌入
        """
        if self.components is None:
            self.load_components()
        
        # 分词
        text_inputs = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.components.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码
        with torch.no_grad():
            text_embeddings = self.components.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        使用VAE编码图像到潜空间
        
        Args:
            images: 图像张量 [B, C, H, W], 值域[0, 1]
            
        Returns:
            torch.Tensor: 潜在表示 [B, C_latent, H//8, W//8]
        """
        if self.components is None:
            self.load_components()
        
        # 转换到[-1, 1]
        images = 2.0 * images - 1.0
        
        with torch.no_grad():
            latents = self.components.vae.encode(images).latent_dist.sample()
            latents = latents * self.components.vae.config.scaling_factor
        
        return latents
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        使用VAE解码潜在表示到图像
        
        Args:
            latents: 潜在表示 [B, C_latent, H//8, W//8]
            
        Returns:
            torch.Tensor: 图像张量 [B, C, H, W], 值域[0, 1]
        """
        if self.components is None:
            self.load_components()
        
        # 缩放
        latents = latents / self.components.vae.config.scaling_factor
        
        with torch.no_grad():
            images = self.components.vae.decode(latents).sample
        
        # 转换到[0, 1]
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        return images
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.components is None:
            self.load_components()
        
        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "scheduler_type": self.scheduler_type,
            "unet_config": self.components.unet.config,
            "vae_config": self.components.vae.config,
            "vae_scale_factor": self.get_vae_scale_factor(),
            "text_encoder_max_length": self.components.tokenizer.model_max_length
        }
    
    def free_memory(self):
        """释放GPU内存"""
        if self.components is not None:
            del self.components
            self.components = None
        
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("GPU内存已释放")


def create_sd_loader(
    model_path: str = "checkpoints/sd2",
    device: str = "auto",
    dtype: Optional[torch.dtype] = None,
    scheduler_type: str = "ddim"
) -> StableDiffusionLoader:
    """
    创建Stable Diffusion加载器的便捷函数
    
    Args:
        model_path: 模型路径
        device: 运行设备
        dtype: 数据类型
        scheduler_type: 调度器类型
        
    Returns:
        StableDiffusionLoader: 配置好的模型加载器
    """
    return StableDiffusionLoader(
        model_path=model_path,
        device=device,
        dtype=dtype,
        scheduler_type=scheduler_type
    )


# 测试函数
def test_sd_loader():
    """测试模型加载器"""
    logger.info("开始测试Stable Diffusion加载器...")
    
    try:
        # 创建加载器
        loader = create_sd_loader()
        
        # 加载组件
        components = loader.load_components()
        logger.info("✅ 组件加载测试通过")
        
        # 测试文本编码
        text_emb = loader.encode_text("a beautiful landscape")
        logger.info(f"✅ 文本编码测试通过: {text_emb.shape}")
        
        # 测试图像编码解码
        dummy_image = torch.randn(1, 3, 512, 512, dtype=loader.dtype, device=loader.device)
        # 确保图像在[0,1]范围内
        dummy_image = torch.clamp(dummy_image * 0.5 + 0.5, 0.0, 1.0)
        latents = loader.encode_images(dummy_image)
        decoded = loader.decode_latents(latents)
        logger.info(f"✅ 图像编码解码测试通过: {dummy_image.shape} -> {latents.shape} -> {decoded.shape}")
        
        # 显示模型信息
        info = loader.get_model_info()
        logger.info(f"✅ 模型信息: VAE缩放因子={info['vae_scale_factor']}")
        
        logger.info("🎉 所有测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    test_sd_loader() 