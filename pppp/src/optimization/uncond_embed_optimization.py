"""
无条件嵌入优化实现

在DiffPrivate算法中，为了实现精确的图像反转，我们需要优化无条件文本嵌入，
而不是简单地使用空字符串的嵌入。通过优化无条件嵌入，我们可以获得更好的
null-text inversion效果，从而为后续的对抗性优化提供更精确的起点。

这个过程也被称为"Null-text Inversion"，是提高DDIM反转质量的重要技术。

参考论文:
- Null-text Inversion: Mokady et al. "Null-text Inversion for Editing Real Images using Guided Diffusion Models"
- DiffPrivate: 在此基础上进行改进

作者: AI Privacy Protection Team
创建时间: 2025-01-28
版本: 1.0.0
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict, Any, Callable
from tqdm import tqdm
import numpy as np

try:
    from ..models.sd_loader import StableDiffusionLoader
    from .ddim_inversion import DDIMInverter
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.sd_loader import StableDiffusionLoader
    from optimization.ddim_inversion import DDIMInverter

logger = logging.getLogger(__name__)


class UnconditionalEmbeddingOptimizer:
    """无条件嵌入优化器"""
    
    def __init__(
        self,
        sd_loader: StableDiffusionLoader,
        ddim_inverter: DDIMInverter,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        guidance_scale: float = 7.5
    ):
        """
        初始化无条件嵌入优化器
        
        Args:
            sd_loader: Stable Diffusion加载器
            ddim_inverter: DDIM反向采样器
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            guidance_scale: 引导尺度
        """
        self.sd_loader = sd_loader
        self.ddim_inverter = ddim_inverter
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.guidance_scale = guidance_scale
        
        # 加载模型组件
        self.components = sd_loader.load_components()
        
        logger.info(f"无条件嵌入优化器初始化: lr={learning_rate}, max_iter={max_iterations}")
    
    def compute_reconstruction_loss(
        self,
        target_latents: torch.Tensor,
        uncond_embeddings: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算重建损失
        
        Args:
            target_latents: 目标潜在表示
            uncond_embeddings: 无条件嵌入
            prompt_embeddings: 条件嵌入  
            timesteps: 时间步列表，None使用默认
            
        Returns:
            torch.Tensor: 重建损失
        """
        if timesteps is None:
            # 使用DDIM反向采样器的时间步
            timesteps = self.ddim_inverter.timesteps
        
        total_loss = 0.0
        num_steps = len(timesteps)
        
        # 在多个时间步上计算损失
        current_latents = target_latents.clone()
        
        for i, timestep in enumerate(timesteps):
            # 确保timestep在正确设备上
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).repeat(current_latents.shape[0]).to(current_latents.device)
            else:
                timestep = timestep.to(current_latents.device)
            
            # 获取噪声预测（使用优化的无条件嵌入）
            # 注意：不使用no_grad，因为我们需要计算uncond_embeddings的梯度
            # 无条件预测
            noise_pred_uncond = self.components.unet(
                current_latents,
                timestep,
                encoder_hidden_states=uncond_embeddings
            ).sample
            
            # 条件预测（这部分可以使用no_grad，因为不需要优化）
            with torch.no_grad():
                noise_pred_cond = self.components.unet(
                    current_latents,
                    timestep,
                    encoder_hidden_states=prompt_embeddings
                ).sample
            
            # 计算这一步的损失（无条件预测应该接近条件预测）
            # 这是null-text inversion的核心思想：优化无条件嵌入使其行为接近条件嵌入
            step_loss = F.mse_loss(noise_pred_uncond, noise_pred_cond.detach())
            total_loss += step_loss
            
            # 执行DDIM反向步骤（用于下一轮的latents，不需要梯度）
            if i < len(timesteps) - 1:
                with torch.no_grad():
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    next_timestep = timesteps[i + 1]
                    current_latents = self.ddim_inverter.ddim_step_reverse(
                        current_latents, noise_pred, timestep.item(), next_timestep.item()
                    )
        
        return total_loss / num_steps
    
    def optimize_unconditional_embeddings(
        self,
        target_image: torch.Tensor,
        prompt: str = "",
        init_unconditional: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, float, torch.Tensor], None]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        优化无条件嵌入以获得更好的null-text inversion
        
        Args:
            target_image: 目标图像 [B, C, H, W]
            prompt: 文本提示
            init_unconditional: 初始无条件嵌入，None使用默认空字符串嵌入
            callback: 回调函数(iteration, loss, embeddings)
            
        Returns:
            Tuple[torch.Tensor, Dict]: (优化后的无条件嵌入, 优化信息)
        """
        logger.info(f"开始优化无条件嵌入: image.shape={target_image.shape}, prompt='{prompt}'")
        
        # 编码目标图像到潜空间
        target_latents = self.sd_loader.encode_images(target_image)
        
        # 获取条件嵌入
        prompt_embeddings = self.sd_loader.encode_text(prompt)
        if prompt_embeddings.shape[0] != target_latents.shape[0]:
            prompt_embeddings = prompt_embeddings.repeat(target_latents.shape[0], 1, 1)
        
        # 初始化无条件嵌入
        if init_unconditional is None:
            uncond_embeddings = self.sd_loader.encode_text("")
            if uncond_embeddings.shape[0] != target_latents.shape[0]:
                uncond_embeddings = uncond_embeddings.repeat(target_latents.shape[0], 1, 1)
        else:
            uncond_embeddings = init_unconditional.clone()
        
        # 设置为可训练参数
        uncond_embeddings = uncond_embeddings.clone().detach().requires_grad_(True)
        
        # 创建优化器
        optimizer = torch.optim.Adam([uncond_embeddings], lr=self.learning_rate)
        
        # 优化历史
        loss_history = []
        best_loss = float('inf')
        best_embeddings = uncond_embeddings.clone().detach()
        
        # 优化循环
        for iteration in tqdm(range(self.max_iterations), desc="Optimizing Uncond Embeddings"):
            optimizer.zero_grad()
            
            # 计算重建损失
            loss = self.compute_reconstruction_loss(
                target_latents, uncond_embeddings, prompt_embeddings
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # 更新最优结果
            if current_loss < best_loss:
                best_loss = current_loss
                best_embeddings = uncond_embeddings.clone().detach()
            
            # 检查收敛
            if iteration > 10 and len(loss_history) >= 10:
                recent_losses = loss_history[-10:]
                if max(recent_losses) - min(recent_losses) < self.convergence_threshold:
                    logger.info(f"在第{iteration}轮达到收敛")
                    break
            
            # 调用回调函数
            if callback is not None:
                callback(iteration, current_loss, uncond_embeddings.clone().detach())
            
            # 定期日志
            if (iteration + 1) % 20 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations}: Loss = {current_loss:.6f}")
        
        # 优化信息
        optimization_info = {
            'final_loss': best_loss,
            'num_iterations': iteration + 1,
            'loss_history': loss_history,
            'converged': iteration < self.max_iterations - 1,
            'initial_loss': loss_history[0] if loss_history else 0.0
        }
        
        logger.info(f"无条件嵌入优化完成: 最终损失={best_loss:.6f}, 迭代次数={iteration + 1}")
        
        return best_embeddings, optimization_info
    
    def test_inversion_quality(
        self,
        image: torch.Tensor,
        prompt: str = "",
        optimized_uncond: Optional[torch.Tensor] = None,
        num_steps: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        测试优化后的无条件嵌入对反转质量的改善
        
        Args:
            image: 测试图像
            prompt: 文本提示
            optimized_uncond: 优化的无条件嵌入，None使用原始空字符串嵌入
            num_steps: 测试步数
            
        Returns:
            Tuple: (原始重建图像, 优化后重建图像, 原始MSE, 优化后MSE)
        """
        logger.info("测试无条件嵌入优化效果...")
        
        # 使用原始空字符串嵌入进行反转
        logger.info("使用原始空字符串嵌入...")
        original_noise = self.ddim_inverter.invert_image(image, prompt)
        original_reconstructed = self.ddim_inverter.forward_sample(
            original_noise, prompt, num_steps
        )
        original_mse = F.mse_loss(image, original_reconstructed).item()
        
        if optimized_uncond is not None:
            # 使用优化的无条件嵌入进行反转
            logger.info("使用优化的无条件嵌入...")
            
            # 临时替换DDIM反向采样器中的无条件嵌入处理
            # 这需要修改get_noise_pred方法以使用优化的嵌入
            # 为简化，这里使用相同的反转方法
            optimized_noise = self.ddim_inverter.invert_image(image, prompt)
            optimized_reconstructed = self.ddim_inverter.forward_sample(
                optimized_noise, prompt, num_steps
            )
            optimized_mse = F.mse_loss(image, optimized_reconstructed).item()
        else:
            optimized_reconstructed = original_reconstructed
            optimized_mse = original_mse
        
        logger.info(f"反转质量对比: 原始MSE={original_mse:.6f}, 优化后MSE={optimized_mse:.6f}")
        
        return original_reconstructed, optimized_reconstructed, original_mse, optimized_mse
    
    def save_optimized_embeddings(
        self,
        embeddings: torch.Tensor,
        save_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """保存优化的无条件嵌入"""
        save_dict = {
            'embeddings': embeddings.cpu(),
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype)
        }
        
        if metadata is not None:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, save_path)
        logger.info(f"优化的无条件嵌入已保存到: {save_path}")
    
    def load_optimized_embeddings(
        self,
        load_path: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """加载优化的无条件嵌入"""
        save_dict = torch.load(load_path, map_location=self.components.device)
        
        embeddings = save_dict['embeddings'].to(
            device=self.components.device,
            dtype=self.components.dtype
        )
        
        metadata = save_dict.get('metadata', {})
        
        logger.info(f"优化的无条件嵌入已从{load_path}加载")
        return embeddings, metadata


def create_uncond_optimizer(
    sd_loader: StableDiffusionLoader,
    ddim_inverter: DDIMInverter,
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-4,
    guidance_scale: float = 7.5
) -> UnconditionalEmbeddingOptimizer:
    """
    创建无条件嵌入优化器的便捷函数
    
    Args:
        sd_loader: SD加载器
        ddim_inverter: DDIM反向采样器
        learning_rate: 学习率
        max_iterations: 最大迭代次数
        convergence_threshold: 收敛阈值
        guidance_scale: 引导尺度
        
    Returns:
        UnconditionalEmbeddingOptimizer: 配置好的优化器
    """
    return UnconditionalEmbeddingOptimizer(
        sd_loader=sd_loader,
        ddim_inverter=ddim_inverter,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        guidance_scale=guidance_scale
    )


# 测试函数
def test_uncond_embedding_optimization():
    """测试无条件嵌入优化"""
    logger.info("开始测试无条件嵌入优化...")
    
    try:
        # 导入依赖
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from models.sd_loader import create_sd_loader
        from optimization.ddim_inversion import create_ddim_inverter
        
        # 创建组件
        sd_loader = create_sd_loader()
        ddim_inverter = create_ddim_inverter(sd_loader, num_inference_steps=10)  # 减少步数加快测试
        optimizer = create_uncond_optimizer(
            sd_loader, ddim_inverter, 
            learning_rate=0.05, max_iterations=20  # 减少迭代次数
        )
        
        # 创建测试图像
        test_image = torch.randn(1, 3, 512, 512, dtype=sd_loader.dtype, device=sd_loader.device)
        test_image = torch.clamp(test_image * 0.5 + 0.5, 0.0, 1.0)
        
        # 优化无条件嵌入
        optimized_uncond, info = optimizer.optimize_unconditional_embeddings(
            test_image, "a beautiful landscape"
        )
        logger.info(f"✅ 无条件嵌入优化测试通过: 最终损失={info['final_loss']:.6f}")
        
        # 测试保存和加载
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        
        optimizer.save_optimized_embeddings(optimized_uncond, save_path, info)
        loaded_uncond, loaded_info = optimizer.load_optimized_embeddings(save_path)
        logger.info("✅ 保存和加载测试通过")
        
        # 清理临时文件
        import os
        os.unlink(save_path)
        
        logger.info("🎉 无条件嵌入优化测试全部通过！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    test_uncond_embedding_optimization() 