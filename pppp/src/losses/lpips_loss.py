"""
LPIPS感知损失函数实现

该模块提供LPIPS（Learned Perceptual Image Patch Similarity）感知损失，
用于保证图像变换后的视觉质量保持。

功能包括：
1. LPIPS网络加载和推理
2. 感知损失计算
3. 多尺度感知损失
4. 梯度支持的可微分实现

基于DiffPrivate论文中的保真度损失设计。

作者: AI Privacy Protection System
日期: 2025-07-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any, Union, List
import numpy as np

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.warning("LPIPS库不可用。感知损失功能将被禁用。")

logger = logging.getLogger(__name__)

class LPIPSLoss(nn.Module):
    """
    LPIPS感知损失函数
    
    实现基于学习的感知图像补丁相似性损失，用于衡量图像的感知质量差异
    """
    
    def __init__(
        self,
        net: str = "alex",  # 'alex', 'vgg', 'squeeze'
        version: str = "0.1",
        use_gpu: bool = True,
        spatial_average: bool = True,
        pixel_loss_weight: float = 0.0,
        normalize_input: bool = True
    ):
        """
        初始化LPIPS损失函数
        
        Args:
            net: 使用的网络骨架 ('alex', 'vgg', 'squeeze')
            version: LPIPS版本
            use_gpu: 是否使用GPU
            spatial_average: 是否对空间维度求平均
            pixel_loss_weight: 像素损失权重（与LPIPS结合）
            normalize_input: 是否对输入进行归一化
        """
        super().__init__()
        
        if not LPIPS_AVAILABLE:
            raise ImportError("LPIPS库不可用，请安装: pip install lpips")
        
        self.net = net
        self.version = version
        self.use_gpu = use_gpu
        self.spatial_average = spatial_average
        self.pixel_loss_weight = pixel_loss_weight
        self.normalize_input = normalize_input
        
        # 初始化LPIPS网络
        self.lpips_net = lpips.LPIPS(
            net=net,
            version=version,
            spatial=not spatial_average  # spatial=True返回空间图，False返回标量
        )
        
        if use_gpu and torch.cuda.is_available():
            self.lpips_net = self.lpips_net.cuda()
        
        # 冻结LPIPS网络参数
        for param in self.lpips_net.parameters():
            param.requires_grad = False
        
        logger.info(f"LPIPS损失初始化: net={net}, version={version}, spatial_avg={spatial_average}")
    
    def preprocess_images(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        预处理图像用于LPIPS计算
        
        Args:
            images: 输入图像 [batch_size, 3, H, W]，值域[0,1]或[-1,1]
            
        Returns:
            预处理后的图像
        """
        # LPIPS期望输入范围为[-1, 1]
        if self.normalize_input:
            if images.min() >= 0:  # 如果输入是[0,1]
                images = images * 2.0 - 1.0  # 转换为[-1,1]
        
        return images
    
    def compute_pixel_loss(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        loss_type: str = "l2"
    ) -> torch.Tensor:
        """
        计算像素级损失
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            loss_type: 损失类型 ('l1', 'l2', 'mse')
            
        Returns:
            像素损失
        """
        if loss_type == "l1":
            return F.l1_loss(img1, img2, reduction='mean')
        elif loss_type in ["l2", "mse"]:
            return F.mse_loss(img1, img2, reduction='mean')
        else:
            raise ValueError(f"不支持的像素损失类型: {loss_type}")
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算LPIPS感知损失
        
        Args:
            img1: 第一张图像 [batch_size, 3, H, W]
            img2: 第二张图像 [batch_size, 3, H, W]
            return_components: 是否返回损失组件
            
        Returns:
            LPIPS损失或损失字典
        """
        # 预处理图像
        img1_proc = self.preprocess_images(img1)
        img2_proc = self.preprocess_images(img2)
        
        # 计算LPIPS损失
        lpips_loss = self.lpips_net(img1_proc, img2_proc)
        
        # 如果spatial_average=False，需要手动求平均
        if not self.spatial_average:
            lpips_loss = lpips_loss.mean()
        else:
            lpips_loss = lpips_loss.mean()  # 确保是标量
        
        # 计算像素损失（如果需要）
        pixel_loss = 0.0
        if self.pixel_loss_weight > 0:
            pixel_loss = self.compute_pixel_loss(img1, img2)
        
        # 总损失
        total_loss = lpips_loss + self.pixel_loss_weight * pixel_loss
        
        if return_components:
            return {
                "total_loss": total_loss,
                "lpips_loss": lpips_loss,
                "pixel_loss": pixel_loss,
                "pixel_weight": self.pixel_loss_weight
            }
        else:
            return total_loss

class MultiScaleLPIPSLoss(nn.Module):
    """
    多尺度LPIPS损失
    
    在多个尺度上计算LPIPS损失，提供更全面的感知评估
    """
    
    def __init__(
        self,
        scales: List[float] = [1.0, 0.5, 0.25],
        weights: Optional[List[float]] = None,
        net: str = "alex",
        version: str = "0.1",
        use_gpu: bool = True,
        spatial_average: bool = True
    ):
        """
        初始化多尺度LPIPS损失
        
        Args:
            scales: 尺度列表
            weights: 各尺度权重
            net: LPIPS网络
            version: LPIPS版本
            use_gpu: 是否使用GPU
            spatial_average: 是否空间平均
        """
        super().__init__()
        
        self.scales = scales
        self.weights = weights if weights is not None else [1.0] * len(scales)
        
        assert len(self.weights) == len(scales), "权重数量必须与尺度数量匹配"
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # 创建LPIPS损失
        self.lpips_loss = LPIPSLoss(
            net=net,
            version=version,
            use_gpu=use_gpu,
            spatial_average=spatial_average,
            normalize_input=True
        )
        
        logger.info(f"多尺度LPIPS损失初始化: scales={scales}, weights={self.weights}")
    
    def resize_images(
        self,
        images: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """
        调整图像尺寸
        
        Args:
            images: 输入图像
            scale: 缩放比例
            
        Returns:
            调整后的图像
        """
        if scale == 1.0:
            return images
        
        _, _, h, w = images.shape
        new_h, new_w = int(h * scale), int(w * scale)
        
        return F.interpolate(
            images,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多尺度LPIPS损失
        
        Args:
            img1: 第一张图像
            img2: 第二张图像  
            return_components: 是否返回组件
            
        Returns:
            多尺度LPIPS损失
        """
        total_loss = 0.0
        components = {}
        
        for i, (scale, weight) in enumerate(zip(self.scales, self.weights)):
            # 调整图像尺寸
            img1_scaled = self.resize_images(img1, scale)
            img2_scaled = self.resize_images(img2, scale)
            
            # 计算该尺度的LPIPS损失
            scale_loss = self.lpips_loss(img1_scaled, img2_scaled)
            
            # 加权累加
            weighted_loss = weight * scale_loss
            total_loss += weighted_loss
            
            if return_components:
                components[f"scale_{scale}_loss"] = scale_loss
                components[f"scale_{scale}_weight"] = weight
                components[f"scale_{scale}_weighted"] = weighted_loss
        
        if return_components:
            components["total_loss"] = total_loss
            return components
        else:
            return total_loss

def create_lpips_loss(
    net: str = "alex",
    version: str = "0.1",
    use_gpu: bool = True,
    spatial_average: bool = True,
    pixel_loss_weight: float = 0.0,
    multiscale: bool = False,
    scales: Optional[List[float]] = None,
    scale_weights: Optional[List[float]] = None
) -> Union[LPIPSLoss, MultiScaleLPIPSLoss]:
    """
    创建LPIPS损失函数的便捷函数
    
    Args:
        net: LPIPS网络
        version: LPIPS版本
        use_gpu: 是否使用GPU
        spatial_average: 是否空间平均
        pixel_loss_weight: 像素损失权重
        multiscale: 是否使用多尺度
        scales: 多尺度列表
        scale_weights: 多尺度权重
        
    Returns:
        LPIPS损失函数实例
    """
    if multiscale:
        if scales is None:
            scales = [1.0, 0.5, 0.25]
        return MultiScaleLPIPSLoss(
            scales=scales,
            weights=scale_weights,
            net=net,
            version=version,
            use_gpu=use_gpu,
            spatial_average=spatial_average
        )
    else:
        return LPIPSLoss(
            net=net,
            version=version,
            use_gpu=use_gpu,
            spatial_average=spatial_average,
            pixel_loss_weight=pixel_loss_weight
        )

def test_lpips_loss():
    """测试LPIPS损失函数"""
    print("🧪 测试LPIPS损失函数...")
    
    try:
        if not LPIPS_AVAILABLE:
            print("❌ LPIPS库不可用，跳过测试")
            return False
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        
        # 创建测试图像
        img1 = torch.rand(batch_size, 3, 256, 256, device=device, requires_grad=True)
        img2 = torch.rand(batch_size, 3, 256, 256, device=device, requires_grad=True)
        
        print(f"✅ 测试环境: 设备={device}, 批大小={batch_size}")
        
        # 测试基础LPIPS损失
        print("🔮 测试基础LPIPS损失...")
        lpips_loss = create_lpips_loss(
            net="alex",
            use_gpu=(device == "cuda"),
            pixel_loss_weight=0.1
        )
        
        loss_dict = lpips_loss(img1, img2, return_components=True)
        print("✅ 基础LPIPS损失计算成功:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.item():.6f}")
            else:
                print(f"   {key}: {value:.6f}")
        
        # 测试梯度
        print("📈 测试梯度计算...")
        total_loss = loss_dict["total_loss"]
        
        # 清除梯度
        if img1.grad is not None:
            img1.grad.zero_()
        if img2.grad is not None:
            img2.grad.zero_()
        
        total_loss.backward()
        
        img1_grad_norm = img1.grad.norm().item() if img1.grad is not None else 0
        img2_grad_norm = img2.grad.norm().item() if img2.grad is not None else 0
        
        print(f"✅ 梯度计算成功:")
        print(f"   img1梯度范数: {img1_grad_norm:.6f}")
        print(f"   img2梯度范数: {img2_grad_norm:.6f}")
        
        # 测试多尺度LPIPS损失
        print("🔍 测试多尺度LPIPS损失...")
        multiscale_lpips = create_lpips_loss(
            net="alex",
            use_gpu=(device == "cuda"),
            multiscale=True,
            scales=[1.0, 0.5],
            scale_weights=[0.7, 0.3]
        )
        
        # 重新创建图像（避免梯度累积）
        img1_new = torch.rand(batch_size, 3, 256, 256, device=device, requires_grad=True)
        img2_new = torch.rand(batch_size, 3, 256, 256, device=device, requires_grad=True)
        
        multiscale_dict = multiscale_lpips(img1_new, img2_new, return_components=True)
        print("✅ 多尺度LPIPS损失计算成功:")
        for key, value in multiscale_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.item():.6f}")
            else:
                print(f"   {key}: {value:.6f}")
        
        # 测试多尺度梯度
        multiscale_loss = multiscale_dict["total_loss"]
        multiscale_loss.backward()
        
        img1_ms_grad = img1_new.grad.norm().item() if img1_new.grad is not None else 0
        img2_ms_grad = img2_new.grad.norm().item() if img2_new.grad is not None else 0
        
        print(f"✅ 多尺度梯度计算成功:")
        print(f"   img1梯度范数: {img1_ms_grad:.6f}")
        print(f"   img2梯度范数: {img2_ms_grad:.6f}")
        
        print("🎉 LPIPS损失函数测试全部通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_lpips_loss() 