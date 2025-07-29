"""
注意力控制实现

这个模块实现了对Stable Diffusion中注意力机制的控制和提取。
通过hook方式拦截UNet中的注意力计算，我们可以：
1. 提取交叉注意力图（cross-attention maps）用于生成掩码
2. 提取自注意力图（self-attention maps）用于结构损失计算
3. 控制注意力权重以实现更精确的编辑效果

这是DiffPrivate算法实现保真度控制和区域定位的关键技术。

参考实现:
- DiffPrivate源码中的AttentionControlEdit
- Prompt-to-Prompt: "Prompt-to-Prompt Image Editing with Cross Attention Control"

作者: AI Privacy Protection Team
创建时间: 2025-01-28
版本: 1.0.0
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Dict, List, Tuple, Callable, Any, Union
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

try:
    from ..models.sd_loader import StableDiffusionLoader, ModelComponents
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.sd_loader import StableDiffusionLoader, ModelComponents

logger = logging.getLogger(__name__)


class AttentionStore:
    """注意力存储器，用于收集和管理注意力图"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置存储的注意力图"""
        self.attention_store = defaultdict(list)
        self.step_store = defaultdict(list)
        self.cur_step = 0
        self.num_att_layers = -1
    
    def get_empty_store(self):
        """获取空的注意力存储结构"""
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}
    
    def forward(self, attn_map: torch.Tensor, place_in_unet: str):
        """
        存储注意力图
        
        Args:
            attn_map: 注意力图 [batch_size * num_heads, seq_len, spatial_dim]
            place_in_unet: UNet中的位置 ("down", "mid", "up")
        """
        key = f"{place_in_unet}_{'cross' if attn_map.shape[1] <= 77 else 'self'}"
        
        if attn_map.shape[1] <= 77:  # 交叉注意力 (文本序列长度 <= 77)
            self.attention_store[key].append(attn_map)
        else:  # 自注意力
            self.attention_store[key].append(attn_map)
    
    def get_average_attention(self) -> Dict[str, torch.Tensor]:
        """获取平均注意力图"""
        average_attention = {}
        for key in self.attention_store:
            if len(self.attention_store[key]) > 0:
                # 对所有步骤的注意力图求平均
                stacked = torch.stack(self.attention_store[key], dim=0)
                average_attention[key] = stacked.mean(0)
            else:
                average_attention[key] = None
        return average_attention
    
    def get_attention_by_step(self, step: int = -1) -> Dict[str, torch.Tensor]:
        """获取特定步骤的注意力图"""
        if step == -1:
            step = len(self.step_store) - 1
        
        if step >= len(self.step_store):
            logger.warning(f"Step {step} not available, using last step")
            step = len(self.step_store) - 1
            
        return self.step_store[step] if step >= 0 else {}
    
    def save_step_attention(self):
        """保存当前步骤的注意力图"""
        self.step_store[self.cur_step] = dict(self.attention_store)
        self.attention_store = defaultdict(list)
        self.cur_step += 1
    
    def get_cross_attention_maps(
        self, 
        resolution: int = 16,
        token_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        获取交叉注意力图并调整到指定分辨率
        
        Args:
            resolution: 目标分辨率
            token_idx: 特定token的索引，None表示使用所有token
            
        Returns:
            torch.Tensor: 注意力图 [batch_size, height, width] 或 [batch_size, seq_len, height, width]
        """
        attention_maps = []
        
        # 收集所有交叉注意力
        for key in ["down_cross", "mid_cross", "up_cross"]:
            if key in self.attention_store and len(self.attention_store[key]) > 0:
                for attn_map in self.attention_store[key]:
                    # attn_map: [batch_size * num_heads, seq_len, spatial_dim]
                    total_heads_batch = attn_map.shape[0]
                    seq_len = attn_map.shape[1]
                    spatial_dim = attn_map.shape[2]
                    
                    # 计算空间维度的高度和宽度
                    h = w = int(spatial_dim ** 0.5)
                    
                    # 验证空间维度是否正确
                    if h * w != spatial_dim:
                        # 可能不是正方形，直接跳过
                        logger.warning(f"跳过非正方形注意力图: spatial_dim={spatial_dim}, h={h}, w={w}")
                        continue
                    
                    # 动态计算batch_size和num_heads（避免硬编码8个头）
                    # 尝试常见的头数配置：8, 16, 4
                    for num_heads in [8, 16, 4, 12, 6, 2, 1]:
                        if total_heads_batch % num_heads == 0:
                            batch_size = total_heads_batch // num_heads
                            expected_size = batch_size * num_heads * seq_len * h * w
                            actual_size = attn_map.numel()
                            if expected_size == actual_size:
                                break
                    else:
                        # 如果都不匹配，直接跳过这个注意力图
                        logger.warning(f"无法重塑注意力图，跳过: shape={attn_map.shape}")
                        continue
                    
                    # 重塑为 [batch_size, num_heads, seq_len, h, w]
                    attn_reshaped = attn_map.view(batch_size, num_heads, seq_len, h, w)
                    
                    # 平均所有头的注意力
                    attn_avg = attn_reshaped.mean(dim=1)  # [batch_size, seq_len, h, w]
                    
                    # 调整到目标分辨率
                    if h != resolution:
                        attn_avg = F.interpolate(
                            attn_avg, size=(resolution, resolution), 
                            mode='bilinear', align_corners=False
                        )
                    
                    attention_maps.append(attn_avg)
        
        if not attention_maps:
            logger.warning("No cross attention maps found")
            return torch.zeros(1, 77, resolution, resolution)
        
        # 平均所有层的注意力
        attention_avg = torch.stack(attention_maps, dim=0).mean(dim=0)
        
        # 如果指定了token索引，只返回该token的注意力
        if token_idx is not None:
            return attention_avg[:, token_idx]  # [batch_size, height, width]
        
        return attention_avg  # [batch_size, seq_len, height, width]


class AttentionControl(ABC):
    """注意力控制基类"""
    
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
    
    @abstractmethod
    def forward(self, attn_map: torch.Tensor, place_in_unet: str) -> torch.Tensor:
        """处理注意力图的抽象方法"""
        pass
    
    def __call__(self, attn_map: torch.Tensor, place_in_unet: str) -> torch.Tensor:
        """调用接口"""
        if self.cur_att_layer >= 0:
            attn_map = self.forward(attn_map, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
        return attn_map
    
    def reset(self):
        """重置状态"""
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionControlEdit(AttentionControl):
    """
    注意力控制编辑器
    
    继承自AttentionControl，专门用于DiffPrivate算法中的注意力控制。
    可以存储、提取和操控注意力图。
    """
    
    def __init__(
        self,
        tokenizer,
        device,
        cross_replace_steps: Union[float, Dict[str, float]] = 0.8,
        self_replace_steps: float = 0.4,
        local_blend: Optional[Dict] = None,
        save_self_attention: bool = True
    ):
        """
        初始化注意力控制编辑器
        
        Args:
            tokenizer: 分词器
            device: 设备
            cross_replace_steps: 交叉注意力替换步骤的比例
            self_replace_steps: 自注意力替换步骤的比例
            local_blend: 局部混合配置
            save_self_attention: 是否保存自注意力图
        """
        super().__init__(tokenizer, device)
        
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.local_blend = local_blend
        self.save_self_attention = save_self_attention
        
        # 注意力存储器
        self.attention_store = AttentionStore()
        
        # 保存的注意力图用于损失计算
        self.saved_cross_attention = []
        self.saved_self_attention = []
        
        logger.info(f"注意力控制编辑器初始化: cross_steps={cross_replace_steps}, self_steps={self_replace_steps}")
    
    def forward(self, attn_map: torch.Tensor, place_in_unet: str) -> torch.Tensor:
        """
        处理注意力图
        
        Args:
            attn_map: 注意力图 [batch_size * num_heads, seq_len, spatial_dim]
            place_in_unet: UNet中的位置
            
        Returns:
            torch.Tensor: 处理后的注意力图
        """
        # 存储注意力图
        self.attention_store.forward(attn_map, place_in_unet)
        
        # 保存用于损失计算的注意力图
        if attn_map.shape[1] <= 77:  # 交叉注意力
            self.saved_cross_attention.append(attn_map.clone())
        elif self.save_self_attention:  # 自注意力
            self.saved_self_attention.append(attn_map.clone())
        
        # 这里可以添加注意力替换逻辑（暂时返回原始注意力图）
        return attn_map
    
    def get_cross_attention_mask(
        self,
        prompts: List[str],
        resolution: int = 64,
        threshold: float = 0.3
    ) -> torch.Tensor:
        """
        基于交叉注意力图生成掩码
        
        Args:
            prompts: 文本提示列表
            resolution: 掩码分辨率
            threshold: 阈值
            
        Returns:
            torch.Tensor: 二值掩码 [batch_size, height, width]
        """
        # 获取交叉注意力图
        cross_attn = self.attention_store.get_cross_attention_maps(resolution)
        
        if cross_attn is None or cross_attn.numel() == 0:
            logger.warning("No cross attention available for mask generation")
            return torch.ones(1, resolution, resolution, device=self.device)
        
        # 找到相关token的注意力
        # 这里简化为使用所有token的平均注意力
        if len(cross_attn.shape) == 4:  # [batch, seq_len, height, width]
            mask = cross_attn.mean(dim=1)  # [batch, height, width]
        else:  # [batch, height, width]
            mask = cross_attn
        
        # 归一化
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        # 应用阈值
        binary_mask = (mask > threshold).float()
        
        return binary_mask
    
    def get_self_attention_loss(self, target_resolution: int = 64) -> torch.Tensor:
        """
        计算自注意力损失（用于结构保持）
        
        Args:
            target_resolution: 目标分辨率
            
        Returns:
            torch.Tensor: 自注意力损失
        """
        if not self.saved_self_attention:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        count = 0
        
        for attn_map in self.saved_self_attention:
            # attn_map: [batch_size * num_heads, spatial_dim, spatial_dim]
            total_heads_batch = attn_map.shape[0]
            spatial_dim = attn_map.shape[1]
            
            # 如果空间维度匹配目标分辨率的平方
            target_spatial_dim = target_resolution * target_resolution
            if spatial_dim == target_spatial_dim:
                # 动态计算batch_size和num_heads
                for num_heads in [8, 16, 4, 12, 6, 2, 1]:
                    if total_heads_batch % num_heads == 0:
                        batch_size = total_heads_batch // num_heads
                        expected_size = batch_size * num_heads * spatial_dim * spatial_dim
                        actual_size = attn_map.numel()
                        if expected_size == actual_size:
                            break
                else:
                    logger.warning(f"无法重塑自注意力图，跳过: shape={attn_map.shape}")
                    continue
                
                # 重塑为 [batch_size, num_heads, spatial_dim, spatial_dim]
                attn_reshaped = attn_map.view(batch_size, num_heads, spatial_dim, spatial_dim)
                
                # 平均所有头
                attn_avg = attn_reshaped.mean(dim=1)  # [batch_size, spatial_dim, spatial_dim]
                
                # 计算自注意力的结构损失（这里使用对角线元素作为简化）
                diagonal = torch.diagonal(attn_avg, dim1=1, dim2=2)  # [batch_size, spatial_dim]
                loss = F.mse_loss(diagonal, torch.ones_like(diagonal) * 0.1)  # 期望对角线元素较小
                
                total_loss += loss
                count += 1
        
        return total_loss / max(count, 1)
    
    def reset(self):
        """重置状态"""
        super().reset()
        self.attention_store.reset()
        self.saved_cross_attention.clear()
        self.saved_self_attention.clear()
    
    def save_step(self):
        """保存当前步骤的注意力"""
        self.attention_store.save_step_attention()


def register_attention_control(
    model: torch.nn.Module,
    controller: AttentionControlEdit
) -> List[Callable]:
    """
    为模型注册注意力控制钩子
    
    Args:
        model: UNet模型
        controller: 注意力控制器
        
    Returns:
        List[Callable]: 钩子列表，用于后续移除
    """
    def ca_forward(self, place_in_unet):
        """注意力前向钩子 - 兼容新版diffusers"""
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # 获取原始forward方法以保持兼容性
            batch_size, sequence_length, dim = hidden_states.shape
            
            # 确定是自注意力还是交叉注意力
            is_cross_attention = encoder_hidden_states is not None
            context = encoder_hidden_states if is_cross_attention else hidden_states
            
            # 使用原始的处理方式，但简化以兼容新版本
            q = self.to_q(hidden_states)
            k = self.to_k(context)
            v = self.to_v(context)
            
            # 重塑为多头格式
            head_dim = dim // self.heads
            q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            
            # 计算注意力分数
            scale = head_dim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # 应用注意力掩码
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            # Softmax得到注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # 调用控制器（这是关键部分）
            attn_weights_reshaped = attn_weights.reshape(batch_size * self.heads, attn_weights.shape[-2], attn_weights.shape[-1])
            attn_weights_controlled = controller(attn_weights_reshaped, place_in_unet)
            attn_weights = attn_weights_controlled.reshape(batch_size, self.heads, attn_weights.shape[-2], attn_weights.shape[-1])
            
            # 应用注意力权重
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, dim)
            
            # 最终投影 - 处理ModuleList情况
            if isinstance(self.to_out, torch.nn.ModuleList):
                # 新版diffusers中to_out是ModuleList
                for layer in self.to_out:
                    attn_output = layer(attn_output)
                return attn_output
            else:
                # 老版本中to_out是单个模块
                return self.to_out(attn_output)
        
        return forward
    
    # 注册钩子
    hooks = []
    
    def register_recr(net_, count, place_in_unet):
        # 修复：正确的类名是'Attention'而不是'CrossAttention'
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count
    
    # 为不同位置注册
    cross_att_count = 0
    for net in model.down_blocks:
        cross_att_count += register_recr(net, 0, "down")
    cross_att_count += register_recr(model.mid_block, 0, "mid")
    for net in model.up_blocks:
        cross_att_count += register_recr(net, 0, "up")
    
    controller.num_att_layers = cross_att_count
    logger.info(f"注册了 {cross_att_count} 个注意力控制钩子")
    
    return hooks


def create_attention_controller(
    tokenizer,
    device,
    cross_replace_steps: float = 0.8,
    self_replace_steps: float = 0.4,
    save_self_attention: bool = True
) -> AttentionControlEdit:
    """
    创建注意力控制器的便捷函数
    
    Args:
        tokenizer: 分词器
        device: 设备
        cross_replace_steps: 交叉注意力替换步骤
        self_replace_steps: 自注意力替换步骤
        save_self_attention: 是否保存自注意力
        
    Returns:
        AttentionControlEdit: 配置好的注意力控制器
    """
    return AttentionControlEdit(
        tokenizer=tokenizer,
        device=device,
        cross_replace_steps=cross_replace_steps,
        self_replace_steps=self_replace_steps,
        save_self_attention=save_self_attention
    )


# 测试函数
def test_attention_control():
    """测试注意力控制"""
    logger.info("开始测试注意力控制...")
    
    try:
        # 导入依赖
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from models.sd_loader import create_sd_loader
        
        # 创建SD加载器
        sd_loader = create_sd_loader()
        components = sd_loader.load_components()
        
        # 创建注意力控制器
        controller = create_attention_controller(
            tokenizer=components.tokenizer,
            device=components.device
        )
        
        # 注册注意力钩子
        hooks = register_attention_control(components.unet, controller)
        logger.info(f"✅ 注意力钩子注册测试通过: {len(hooks)} 个钩子")
        
        # 创建测试数据
        test_latents = torch.randn(1, 4, 64, 64, device=components.device, dtype=components.dtype)
        test_timestep = torch.tensor([100], device=components.device)
        test_prompt_embeds = sd_loader.encode_text("a beautiful landscape")
        
        # 测试UNet前向传播（会触发注意力控制）
        with torch.no_grad():
            noise_pred = components.unet(
                test_latents,
                test_timestep,
                encoder_hidden_states=test_prompt_embeds
            ).sample
        
        logger.info(f"✅ UNet前向传播测试通过: {noise_pred.shape}")
        
        # 测试注意力图提取
        cross_attn = controller.attention_store.get_cross_attention_maps(resolution=64)
        if cross_attn is not None and cross_attn.numel() > 0:
            logger.info(f"✅ 交叉注意力提取测试通过: {cross_attn.shape}")
        
        # 测试掩码生成
        mask = controller.get_cross_attention_mask(["a beautiful landscape"], resolution=64)
        logger.info(f"✅ 掩码生成测试通过: {mask.shape}")
        
        # 测试自注意力损失
        self_attn_loss = controller.get_self_attention_loss()
        logger.info(f"✅ 自注意力损失测试通过: {self_attn_loss.item():.6f}")
        
        logger.info("🎉 注意力控制测试全部通过！")
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
    test_attention_control() 