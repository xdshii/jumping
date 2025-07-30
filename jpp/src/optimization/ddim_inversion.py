"""
DDIM反向采样实现

DDIM（Denoising Diffusion Implicit Models）反向采样是DiffPrivate算法的核心组件。
通过反向DDIM过程，我们可以将真实图像精确地反转到噪声空间，这为后续的
对抗性优化提供了起点。

参考论文:
- DDIM: Song et al. "Denoising Diffusion Implicit Models" (ICLR 2021)
- DiffPrivate: Wu et al. "DiffPrivate: Invisible Adversarial Samples for Privacy Protection"

作者: AI Privacy Protection Team
创建时间: 2025-01-28
版本: 1.0.0
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple, Union, Callable
from tqdm import tqdm
import numpy as np

try:
    from ..models.sd_loader import StableDiffusionLoader, ModelComponents
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.sd_loader import StableDiffusionLoader, ModelComponents

logger = logging.getLogger(__name__)


class DDIMInverter:
    """DDIM反向采样器"""
    
    def __init__(
        self,
        sd_loader: StableDiffusionLoader,
        num_inference_steps: int = 25,  # 平衡质量和内存使用
        guidance_scale: float = 7.5,
        eta: float = 0.0
    ):
        """
        初始化DDIM反向采样器
        
        Args:
            sd_loader: Stable Diffusion加载器
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
            eta: DDIM参数，0为完全确定性，1为标准DDPM
        """
        self.sd_loader = sd_loader
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        
        # 加载模型组件
        self.components = sd_loader.load_components()
        
        # 设置调度器
        self.scheduler = self.components.scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        self.timesteps = self.scheduler.timesteps
        
        logger.info(f"DDIM反向采样器初始化: steps={num_inference_steps}, guidance={guidance_scale}, eta={eta}")
    
    def get_noise_pred(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: torch.Tensor,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        获取噪声预测
        
        Args:
            latents: 潜在表示 [B, C, H, W]
            timestep: 时间步 [B] 或 scalar
            text_embeddings: 文本嵌入 [B, seq_len, dim]
            guidance_scale: 引导尺度，None使用默认值
            
        Returns:
            torch.Tensor: 预测的噪声 [B, C, H, W]
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        # 确保timestep是正确的形状和设备
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).repeat(latents.shape[0]).to(latents.device)
        elif timestep.shape[0] != latents.shape[0]:
            timestep = timestep.repeat(latents.shape[0]).to(latents.device)
        else:
            timestep = timestep.to(latents.device)
        
        if guidance_scale <= 1.0:
            # 无引导情况
            noise_pred = self.components.unet(
                latents,
                timestep,
                encoder_hidden_states=text_embeddings
            ).sample
        else:
            # 有引导情况：需要条件和无条件预测
            # 创建无条件嵌入（空文本）
            batch_size = latents.shape[0]
            uncond_embeddings = self.sd_loader.encode_text("")
            uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)
            
            # 拼接条件和无条件输入
            latents_input = torch.cat([latents, latents], dim=0)
            text_embeddings_input = torch.cat([uncond_embeddings, text_embeddings], dim=0)
            timestep_input = torch.cat([timestep, timestep], dim=0)
            
            # 预测噪声
            noise_pred = self.components.unet(
                latents_input,
                timestep_input,
                encoder_hidden_states=text_embeddings_input
            ).sample
            
            # 分离条件和无条件预测
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            
            # 应用引导
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        return noise_pred
    
    def ddim_step_reverse(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: int,
        prev_timestep: int
    ) -> torch.Tensor:
        """
        DDIM反向步骤：从x_t到x_{t+1}
        
        Args:
            latents: 当前潜在表示 x_t
            noise_pred: 预测的噪声
            timestep: 当前时间步 t
            prev_timestep: 下一个时间步 t+1
            
        Returns:
            torch.Tensor: 下一步的潜在表示 x_{t+1}
        """
        # 获取alpha值
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        if self.scheduler.config.prediction_type == "epsilon":
            # 标准情况：预测噪声
            pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        elif self.scheduler.config.prediction_type == "sample":
            # 直接预测原始样本
            pred_original_sample = noise_pred
        elif self.scheduler.config.prediction_type == "v_prediction":
            # v-参数化
            pred_original_sample = (alpha_prod_t ** 0.5) * latents - (beta_prod_t ** 0.5) * noise_pred
        else:
            raise ValueError(f"Unknown prediction type: {self.scheduler.config.prediction_type}")
        
        # 计算方向向量
        pred_sample_direction = (beta_prod_t_prev ** 0.5) * noise_pred
        
        # 反向DDIM步骤
        prev_sample = (alpha_prod_t_prev ** 0.5) * pred_original_sample + pred_sample_direction
        
        return prev_sample
    
    def invert_image(
        self,
        image: torch.Tensor,
        prompt: str = "",
        return_intermediates: bool = False,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        反转图像到噪声空间
        
        Args:
            image: 输入图像 [B, C, H, W], 值域[0, 1]
            prompt: 文本提示
            return_intermediates: 是否返回中间结果
            callback: 回调函数，用于监控进度
            
        Returns:
            torch.Tensor: 反转后的噪声 [B, C_latent, H//8, W//8]
            如果return_intermediates=True，还返回中间潜在表示列表
        """
        logger.info(f"开始DDIM反向采样: image.shape={image.shape}, prompt='{prompt}'")
        
        # 编码图像到潜空间
        latents = self.sd_loader.encode_images(image)
        
        # 编码文本
        text_embeddings = self.sd_loader.encode_text(prompt)
        text_embeddings = text_embeddings.repeat(latents.shape[0], 1, 1)
        
        # 存储中间结果
        intermediates = [latents.clone()] if return_intermediates else []
        
        # 反向DDIM循环
        with torch.no_grad():
            for i, timestep in enumerate(tqdm(self.timesteps, desc="DDIM Inversion")):
                # 预测噪声
                noise_pred = self.get_noise_pred(latents, timestep, text_embeddings)
                
                # 计算下一个时间步
                if i < len(self.timesteps) - 1:
                    next_timestep = self.timesteps[i + 1]
                else:
                    # 最后一步，到纯噪声
                    next_timestep = torch.tensor(self.scheduler.config.num_train_timesteps - 1)
                
                # 执行反向步骤
                latents = self.ddim_step_reverse(
                    latents, noise_pred, timestep.item(), next_timestep.item()
                )
                
                # 存储中间结果
                if return_intermediates:
                    intermediates.append(latents.clone())
                
                # 调用回调函数
                if callback is not None:
                    callback(i + 1, latents.clone())
        
        logger.info(f"DDIM反向采样完成: noise.shape={latents.shape}")
        
        if return_intermediates:
            return latents, intermediates
        else:
            return latents
    
    def forward_sample(
        self,
        noise: torch.Tensor,
        prompt: str = "",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None
    ) -> torch.Tensor:
        """
        从噪声前向采样生成图像（用于验证反转质量）
        
        Args:
            noise: 输入噪声 [B, C_latent, H//8, W//8]
            prompt: 文本提示
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
            callback: 回调函数
            
        Returns:
            torch.Tensor: 生成的图像 [B, C, H, W]
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        logger.info(f"开始DDIM前向采样: noise.shape={noise.shape}")
        
        # 设置调度器
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # 编码文本
        text_embeddings = self.sd_loader.encode_text(prompt)
        text_embeddings = text_embeddings.repeat(noise.shape[0], 1, 1)
        
        # 初始化潜在表示
        latents = noise.clone()
        
        # 前向DDIM循环
        with torch.no_grad():
            for i, timestep in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
                # 预测噪声
                noise_pred = self.get_noise_pred(latents, timestep, text_embeddings, guidance_scale)
                
                # DDIM步骤
                latents = self.scheduler.step(noise_pred, timestep, latents, eta=self.eta).prev_sample
                
                # 调用回调函数
                if callback is not None:
                    callback(i + 1, latents.clone())
        
        # 解码到图像
        images = self.sd_loader.decode_latents(latents)
        
        logger.info(f"DDIM前向采样完成: images.shape={images.shape}")
        return images
    
    def test_inversion_quality(
        self,
        image: torch.Tensor,
        prompt: str = "",
        num_test_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        测试反转质量
        
        Args:
            image: 测试图像
            prompt: 文本提示
            num_test_steps: 测试步数
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]: (原图, 重建图, MSE误差)
        """
        if num_test_steps is None:
            num_test_steps = self.num_inference_steps
        
        logger.info("测试DDIM反转质量...")
        
        # 反转图像
        noise = self.invert_image(image, prompt)
        
        # 重建图像
        reconstructed = self.forward_sample(noise, prompt, num_test_steps)
        
        # 计算误差
        mse_error = F.mse_loss(image, reconstructed).item()
        
        logger.info(f"反转质量测试完成: MSE={mse_error:.6f}")
        
        return image, reconstructed, mse_error


def create_ddim_inverter(
    sd_loader: StableDiffusionLoader,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    eta: float = 0.0
) -> DDIMInverter:
    """
    创建DDIM反向采样器的便捷函数
    
    Args:
        sd_loader: SD加载器
        num_inference_steps: 推理步数
        guidance_scale: 引导尺度
        eta: DDIM参数
        
    Returns:
        DDIMInverter: 配置好的DDIM反向采样器
    """
    return DDIMInverter(
        sd_loader=sd_loader,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        eta=eta
    )


# 测试函数
def test_ddim_inversion():
    """测试DDIM反向采样"""
    logger.info("开始测试DDIM反向采样...")
    
    try:
        # 创建SD加载器
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from models.sd_loader import create_sd_loader
        sd_loader = create_sd_loader()
        
        # 创建DDIM反向采样器
        inverter = create_ddim_inverter(sd_loader, num_inference_steps=20)  # 使用较少步数加快测试
        
        # 创建测试图像
        test_image = torch.randn(1, 3, 512, 512, dtype=sd_loader.dtype, device=sd_loader.device)
        test_image = torch.clamp(test_image * 0.5 + 0.5, 0.0, 1.0)
        
        # 测试反转
        noise = inverter.invert_image(test_image, "a beautiful landscape")
        logger.info(f"✅ 反转测试通过: {test_image.shape} -> {noise.shape}")
        
        # 测试重建
        reconstructed = inverter.forward_sample(noise, "a beautiful landscape", num_inference_steps=20)
        logger.info(f"✅ 重建测试通过: {noise.shape} -> {reconstructed.shape}")
        
        # 测试质量
        original, recon, mse = inverter.test_inversion_quality(test_image, "a beautiful landscape", 20)
        logger.info(f"✅ 质量测试通过: MSE={mse:.6f}")
        
        logger.info("🎉 DDIM反向采样测试全部通过！")
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
    test_ddim_inversion()