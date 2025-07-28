"""
扩散步骤函数实现

该模块提供扩散过程中单步计算的封装，包括：
1. DDIM前向/反向步骤
2. 噪声预测与引导
3. 潜空间更新
4. 调度器集成

作者: AI Privacy Protection System
日期: 2025-07-28
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union, Dict, Any, Callable
from tqdm import tqdm
import numpy as np

try:
    from ..models.sd_loader import StableDiffusionLoader, ModelComponents
    from .ddim_inversion import DDIMInverter
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.sd_loader import StableDiffusionLoader, ModelComponents
    from optimization.ddim_inversion import DDIMInverter

logger = logging.getLogger(__name__)

class DiffusionStepper:
    """
    扩散步骤处理器
    
    封装扩散过程中的单步计算，包括DDIM步骤、噪声预测和潜空间更新
    """
    
    def __init__(
        self,
        sd_loader: StableDiffusionLoader,
        guidance_scale: float = 7.5,
        eta: float = 0.0
    ):
        """
        初始化扩散步骤处理器
        
        Args:
            sd_loader: SD加载器
            guidance_scale: 分类引导强度
            eta: DDIM参数，0为确定性，1为随机
        """
        self.sd_loader = sd_loader
        self.guidance_scale = guidance_scale
        self.eta = eta
        
        # 确保组件已加载
        if not hasattr(sd_loader, 'components') or sd_loader.components is None:
            self.components = sd_loader.load_components()
        else:
            self.components = sd_loader.components
        
        # 创建DDIM反转器（用于某些计算）
        self.ddim_inverter = DDIMInverter(sd_loader, guidance_scale=guidance_scale, eta=eta)
        
        logger.info(f"扩散步骤处理器初始化: guidance_scale={guidance_scale}, eta={eta}")
    
    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        text_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        预测噪声
        
        Args:
            latents: 潜空间表示 [batch_size, 4, H, W]
            timestep: 时间步 
            text_embeddings: 条件文本嵌入
            uncond_embeddings: 无条件文本嵌入（可选）
            guidance_scale: 引导强度（可选，覆盖默认值）
            
        Returns:
            预测的噪声 [batch_size, 4, H, W]
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
            
        # 确保timestep格式正确
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], device=latents.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # 确保timestep在正确设备上且形状匹配
        timestep = timestep.to(latents.device)
        if timestep.shape[0] != latents.shape[0]:
            timestep = timestep.repeat(latents.shape[0])
        
        # 条件预测
        noise_pred_cond = self.components.unet(
            latents,
            timestep,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # 如果有无条件嵌入且引导强度>1，进行引导
        if uncond_embeddings is not None and guidance_scale > 1.0:
            # 无条件预测
            noise_pred_uncond = self.components.unet(
                latents,
                timestep,
                encoder_hidden_states=uncond_embeddings
            ).sample
            
            # 分类引导
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
            
        return noise_pred
    
    def ddim_step_forward(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        eta: Optional[float] = None
    ) -> torch.Tensor:
        """
        DDIM前向步骤
        
        Args:
            latents: 当前潜空间
            noise_pred: 预测的噪声
            timestep: 当前时间步
            prev_timestep: 前一个时间步
            eta: DDIM参数
            
        Returns:
            更新后的潜空间
        """
        if eta is None:
            eta = self.eta
            
        # 获取调度器参数
        alpha_prod_t = self.components.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.components.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.components.scheduler.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 计算预测的原始样本
        pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        
        # 计算方向向量
        pred_sample_direction = (beta_prod_t_prev ** 0.5) * noise_pred
        
        # 计算前一步的样本
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # 添加随机性（如果eta > 0）
        if eta > 0:
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            sigma = eta * variance ** 0.5
            
            # 添加噪声
            noise = torch.randn_like(latents)
            prev_sample = prev_sample + sigma * noise
            
        return prev_sample
    
    def ddim_step_reverse(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: int,
        next_timestep: int
    ) -> torch.Tensor:
        """
        DDIM反向步骤（用于反转）
        
        Args:
            latents: 当前潜空间
            noise_pred: 预测的噪声
            timestep: 当前时间步
            next_timestep: 下一个时间步
            
        Returns:
            反转后的潜空间
        """
        # 使用DDIMInverter中的实现
        return self.ddim_inverter.ddim_step_reverse(latents, noise_pred, timestep, next_timestep)
    
    def scheduler_step(
        self,
        noise_pred: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: Optional[float] = None,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        使用调度器进行单步更新
        
        Args:
            noise_pred: 预测的噪声
            timestep: 当前时间步
            sample: 当前样本
            eta: DDIM参数
            use_clipped_model_output: 是否使用裁剪的模型输出
            generator: 随机数生成器
            
        Returns:
            更新后的样本
        """
        if eta is None:
            eta = self.eta
            
        # 设置调度器的eta参数
        original_eta = getattr(self.components.scheduler, 'eta', 0.0)
        self.components.scheduler.eta = eta
        
        try:
            # 使用调度器进行步骤更新
            scheduler_output = self.components.scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=sample,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
                return_dict=True
            )
            
            return scheduler_output.prev_sample
            
        finally:
            # 恢复原始eta值
            self.components.scheduler.eta = original_eta
    
    def compute_velocity(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        计算速度参数化（用于某些扩散模型变体）
        
        Args:
            latents: 潜空间表示
            noise: 噪声
            timestep: 时间步
            
        Returns:
            速度向量
        """
        alpha_prod_t = self.components.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        
        velocity = alpha_prod_t ** 0.5 * noise - beta_prod_t ** 0.5 * latents
        return velocity
    
    def apply_guidance(
        self,
        noise_pred_uncond: torch.Tensor,
        noise_pred_cond: torch.Tensor,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        应用分类引导
        
        Args:
            noise_pred_uncond: 无条件噪声预测
            noise_pred_cond: 条件噪声预测
            guidance_scale: 引导强度
            
        Returns:
            引导后的噪声预测
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
            
        return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    
    def get_timesteps(
        self,
        num_inference_steps: int,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        获取推理时间步序列
        
        Args:
            num_inference_steps: 推理步数
            strength: 强度（用于图像到图像）
            
        Returns:
            时间步张量
        """
        # 设置调度器时间步
        self.components.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.components.scheduler.timesteps
        
        # 如果strength < 1.0，跳过一些初始步骤
        if strength < 1.0:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = timesteps[t_start:]
            
        return timesteps


def create_diffusion_stepper(
    sd_loader: StableDiffusionLoader,
    guidance_scale: float = 7.5,
    eta: float = 0.0
) -> DiffusionStepper:
    """
    创建扩散步骤处理器的便捷函数
    
    Args:
        sd_loader: SD加载器
        guidance_scale: 分类引导强度
        eta: DDIM参数
        
    Returns:
        扩散步骤处理器实例
    """
    return DiffusionStepper(sd_loader, guidance_scale, eta)


def test_diffusion_stepper():
    """测试扩散步骤处理器"""
    print("🧪 测试扩散步骤处理器...")
    
    try:
        # 导入SD加载器
        from models.sd_loader import create_sd_loader
        
        # 创建SD加载器并加载组件
        sd_loader = create_sd_loader()
        components = sd_loader.load_components()
        
        # 创建扩散步骤处理器
        stepper = create_diffusion_stepper(sd_loader)
        
        # 创建测试数据
        test_latents = torch.randn(1, 4, 64, 64, device=components.device, dtype=components.dtype)
        test_timestep = 100
        test_text_embeds = sd_loader.encode_text("a beautiful landscape")
        test_uncond_embeds = sd_loader.encode_text("")
        
        print(f"✅ 扩散步骤处理器创建成功")
        print(f"   设备: {components.device}")
        print(f"   数据类型: {components.dtype}")
        print(f"   引导强度: {stepper.guidance_scale}")
        
        # 测试噪声预测
        print("🔮 测试噪声预测...")
        with torch.no_grad():
            noise_pred = stepper.predict_noise(
                test_latents, 
                test_timestep, 
                test_text_embeds,
                test_uncond_embeds
            )
        
        print(f"✅ 噪声预测成功: {noise_pred.shape}")
        
        # 测试DDIM前向步骤
        print("➡️ 测试DDIM前向步骤...")
        with torch.no_grad():
            next_latents = stepper.ddim_step_forward(
                test_latents,
                noise_pred,
                timestep=100,
                prev_timestep=90
            )
        
        print(f"✅ DDIM前向步骤成功: {next_latents.shape}")
        
        # 测试调度器步骤
        print("📅 测试调度器步骤...")
        timesteps = stepper.get_timesteps(num_inference_steps=20)
        print(f"✅ 时间步获取成功: {len(timesteps)} 步")
        
        with torch.no_grad():
            scheduler_output = stepper.scheduler_step(
                noise_pred,
                timestep=timesteps[0].item(),
                sample=test_latents
            )
        
        print(f"✅ 调度器步骤成功: {scheduler_output.shape}")
        
        # 测试引导应用
        print("🎯 测试引导应用...")
        with torch.no_grad():
            uncond_noise = stepper.predict_noise(test_latents, test_timestep, test_uncond_embeds)
            cond_noise = stepper.predict_noise(test_latents, test_timestep, test_text_embeds)
            guided_noise = stepper.apply_guidance(uncond_noise, cond_noise, guidance_scale=7.5)
        
        print(f"✅ 引导应用成功: {guided_noise.shape}")
        
        print("🎉 扩散步骤处理器测试全部通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_diffusion_stepper() 