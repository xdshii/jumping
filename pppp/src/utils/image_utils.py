"""
AI图像隐私保护系统 - 图像预处理工具模块

这个模块提供了完整的图像预处理功能，包括：
- 图像格式转换和标准化
- 尺寸调整和裁剪
- PyTorch张量转换
- 图像质量验证
- 批处理支持

作者: AI Privacy Protection Team
创建时间: 2025-01-28
版本: 1.0.0
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Union, List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图像处理器类"""
    
    def __init__(
        self, 
        target_size: int = 512,
        device: str = "cuda:0",
        normalize_method: str = "imagenet"
    ):
        """
        初始化图像处理器
        
        Args:
            target_size: 目标图像尺寸
            device: 计算设备
            normalize_method: 归一化方法 ("imagenet", "zero_one", "neg_one_one")
        """
        self.target_size = target_size
        self.device = torch.device(device)
        self.normalize_method = normalize_method
        
        # 预定义的归一化参数
        self.norm_params = {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "zero_one": {
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0]
            },
            "neg_one_one": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5]
            }
        }
        
        # 创建变换器
        self._setup_transforms()
    
    def _setup_transforms(self):
        """设置图像变换器"""
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor()
        ])
        
        # 归一化变换
        norm_params = self.norm_params[self.normalize_method]
        self.normalize_transform = transforms.Normalize(
            mean=norm_params["mean"],
            std=norm_params["std"]
        )
        
        # 反归一化变换
        self.denormalize_transform = transforms.Normalize(
            mean=[-m/s for m, s in zip(norm_params["mean"], norm_params["std"])],
            std=[1/s for s in norm_params["std"]]
        )
    
    def load_image(
        self, 
        image_path: Union[str, Path],
        convert_rgb: bool = True
    ) -> Image.Image:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
            convert_rgb: 是否转换为RGB格式
            
        Returns:
            PIL图像对象
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的图像格式
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            image = Image.open(image_path)
            
            if convert_rgb and image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.debug(f"成功加载图像: {image_path}, 尺寸: {image.size}, 模式: {image.mode}")
            return image
            
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {e}")
    
    def save_image(
        self, 
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        save_path: Union[str, Path],
        quality: int = 95
    ) -> None:
        """
        保存图像
        
        Args:
            image: 图像数据
            save_path: 保存路径
            quality: JPEG质量 (1-100)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为PIL图像
        if isinstance(image, torch.Tensor):
            image = self.tensor_to_pil(image)
        elif isinstance(image, np.ndarray):
            image = self.numpy_to_pil(image)
        
        # 保存图像
        if save_path.suffix.lower() in ['.jpg', '.jpeg']:
            image.save(save_path, 'JPEG', quality=quality, optimize=True)
        else:
            image.save(save_path, optimize=True)
        
        logger.debug(f"图像已保存到: {save_path}")
    
    def pil_to_tensor(
        self, 
        image: Image.Image,
        normalize: bool = True,
        add_batch_dim: bool = True
    ) -> torch.Tensor:
        """
        将PIL图像转换为PyTorch张量
        
        Args:
            image: PIL图像
            normalize: 是否进行归一化
            add_batch_dim: 是否添加批次维度
            
        Returns:
            PyTorch张量 (B, C, H, W) 或 (C, H, W)
        """
        # 调整尺寸并转换为张量
        tensor = self.base_transform(image)
        
        # 归一化
        if normalize:
            tensor = self.normalize_transform(tensor)
        
        # 添加批次维度
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        
        # 移动到指定设备
        tensor = tensor.to(self.device)
        
        return tensor
    
    def tensor_to_pil(
        self, 
        tensor: torch.Tensor,
        denormalize: bool = True
    ) -> Image.Image:
        """
        将PyTorch张量转换为PIL图像
        
        Args:
            tensor: PyTorch张量 (B, C, H, W) 或 (C, H, W)
            denormalize: 是否进行反归一化
            
        Returns:
            PIL图像
        """
        # 移动到CPU
        tensor = tensor.detach().cpu()
        
        # 移除批次维度
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # 反归一化
        if denormalize and self.normalize_method != "zero_one":
            tensor = self.denormalize_transform(tensor)
        
        # 确保值在[0, 1]范围内
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为PIL图像
        transform = transforms.ToPILImage()
        image = transform(tensor)
        
        return image
    
    def numpy_to_pil(self, array: np.ndarray) -> Image.Image:
        """
        将NumPy数组转换为PIL图像
        
        Args:
            array: NumPy数组 (H, W, C) 或 (H, W)
            
        Returns:
            PIL图像
        """
        # 确保数据类型正确
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        
        # 转换为PIL图像
        if array.ndim == 2:
            return Image.fromarray(array, mode='L')
        elif array.ndim == 3:
            return Image.fromarray(array, mode='RGB')
        else:
            raise ValueError(f"不支持的数组形状: {array.shape}")
    
    def resize_image(
        self, 
        image: Union[Image.Image, torch.Tensor],
        size: Union[int, Tuple[int, int]],
        method: str = "lanczos"
    ) -> Union[Image.Image, torch.Tensor]:
        """
        调整图像尺寸
        
        Args:
            image: 输入图像
            size: 目标尺寸
            method: 插值方法 ("lanczos", "bilinear", "bicubic", "nearest")
            
        Returns:
            调整尺寸后的图像
        """
        if isinstance(size, int):
            size = (size, size)
        
        if isinstance(image, Image.Image):
            # PIL图像调整
            resize_methods = {
                "lanczos": Image.Resampling.LANCZOS,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "nearest": Image.Resampling.NEAREST
            }
            return image.resize(size, resize_methods.get(method, Image.Resampling.LANCZOS))
        
        elif isinstance(image, torch.Tensor):
            # 张量调整
            resize_modes = {
                "lanczos": "bicubic",  # PyTorch没有lanczos，使用bicubic
                "bilinear": "bilinear",
                "bicubic": "bicubic",
                "nearest": "nearest"
            }
            mode = resize_modes.get(method, "bicubic")
            
            return F.interpolate(
                image, 
                size=size, 
                mode=mode, 
                align_corners=False if mode != "nearest" else None
            )
        
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
    
    def crop_center(
        self, 
        image: Union[Image.Image, torch.Tensor],
        crop_size: Union[int, Tuple[int, int]]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        中心裁剪图像
        
        Args:
            image: 输入图像
            crop_size: 裁剪尺寸
            
        Returns:
            裁剪后的图像
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        
        if isinstance(image, Image.Image):
            width, height = image.size
            crop_width, crop_height = crop_size
            
            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            
            return image.crop((left, top, right, bottom))
        
        elif isinstance(image, torch.Tensor):
            _, _, height, width = image.shape
            crop_height, crop_width = crop_size
            
            start_y = (height - crop_height) // 2
            start_x = (width - crop_width) // 2
            
            return image[
                :, :, 
                start_y:start_y + crop_height,
                start_x:start_x + crop_width
            ]
        
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
    
    def enhance_image(
        self, 
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0
    ) -> Image.Image:
        """
        图像增强
        
        Args:
            image: 输入图像
            brightness: 亮度调节 (1.0为原始)
            contrast: 对比度调节 (1.0为原始)
            saturation: 饱和度调节 (1.0为原始)
            sharpness: 锐度调节 (1.0为原始)
            
        Returns:
            增强后的图像
        """
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
        
        return image
    
    def validate_image(
        self, 
        image: Union[Image.Image, torch.Tensor, str, Path]
    ) -> Dict[str, Any]:
        """
        验证图像质量和属性
        
        Args:
            image: 图像数据或路径
            
        Returns:
            包含验证结果的字典
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "properties": {}
        }
        
        try:
            # 加载图像（如果是路径）
            if isinstance(image, (str, Path)):
                image = self.load_image(image)
            
            # 转换为PIL图像进行分析
            if isinstance(image, torch.Tensor):
                image = self.tensor_to_pil(image)
            
            # 基本属性
            result["properties"]["size"] = image.size
            result["properties"]["mode"] = image.mode
            result["properties"]["format"] = getattr(image, 'format', 'Unknown')
            
            # 尺寸检查
            width, height = image.size
            if width < 256 or height < 256:
                result["warnings"].append("图像尺寸可能过小，建议至少256x256")
            
            if width != height:
                result["warnings"].append("图像不是正方形，处理时会进行裁剪")
            
            # 颜色模式检查
            if image.mode not in ['RGB', 'RGBA']:
                result["warnings"].append(f"图像颜色模式为{image.mode}，将转换为RGB")
            
            # 计算图像统计信息
            img_array = np.array(image)
            result["properties"]["mean_brightness"] = np.mean(img_array)
            result["properties"]["std_brightness"] = np.std(img_array)
            
            # 检查是否为纯色图像
            if result["properties"]["std_brightness"] < 5:
                result["warnings"].append("图像可能是纯色或对比度过低")
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"图像验证失败: {e}")
        
        return result
    
    def batch_process(
        self, 
        images: List[Union[Image.Image, str, Path]],
        normalize: bool = True,
        add_batch_dim: bool = False
    ) -> torch.Tensor:
        """
        批量处理图像
        
        Args:
            images: 图像列表
            normalize: 是否归一化
            add_batch_dim: 是否保持批次维度
            
        Returns:
            批量图像张量 (B, C, H, W)
        """
        processed_images = []
        
        for i, image in enumerate(images):
            try:
                # 加载图像（如果是路径）
                if isinstance(image, (str, Path)):
                    image = self.load_image(image)
                
                # 转换为张量
                tensor = self.pil_to_tensor(
                    image, 
                    normalize=normalize, 
                    add_batch_dim=False
                )
                processed_images.append(tensor)
                
            except Exception as e:
                logger.warning(f"处理第{i}张图像时出错: {e}")
                continue
        
        if not processed_images:
            raise ValueError("没有成功处理的图像")
        
        # 堆叠为批次
        batch_tensor = torch.stack(processed_images, dim=0)
        
        if not add_batch_dim and batch_tensor.size(0) == 1:
            batch_tensor = batch_tensor.squeeze(0)
        
        return batch_tensor
    
    def create_image_grid(
        self, 
        images: List[Union[Image.Image, torch.Tensor]],
        grid_size: Optional[Tuple[int, int]] = None,
        padding: int = 2
    ) -> Image.Image:
        """
        创建图像网格
        
        Args:
            images: 图像列表
            grid_size: 网格尺寸 (rows, cols)，None为自动计算
            padding: 图像间距
            
        Returns:
            网格图像
        """
        if not images:
            raise ValueError("图像列表不能为空")
        
        # 转换所有图像为PIL格式
        pil_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                pil_images.append(self.tensor_to_pil(img))
            else:
                pil_images.append(img)
        
        # 计算网格尺寸
        num_images = len(pil_images)
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
        else:
            rows, cols = grid_size
        
        # 获取单个图像尺寸
        img_width, img_height = pil_images[0].size
        
        # 创建网格画布
        grid_width = cols * img_width + (cols - 1) * padding
        grid_height = rows * img_height + (rows - 1) * padding
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # 粘贴图像
        for i, img in enumerate(pil_images):
            if i >= rows * cols:
                break
            
            row = i // cols
            col = i % cols
            
            x = col * (img_width + padding)
            y = row * (img_height + padding)
            
            grid_image.paste(img, (x, y))
        
        return grid_image


# 全局图像处理器实例
default_processor = ImageProcessor()


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """加载图像的便捷函数"""
    return default_processor.load_image(image_path)


def save_image(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    quality: int = 95
) -> None:
    """保存图像的便捷函数"""
    default_processor.save_image(image, save_path, quality)


def pil_to_tensor(
    image: Image.Image,
    normalize: bool = True,
    device: str = "cuda:0"
) -> torch.Tensor:
    """PIL到张量转换的便捷函数"""
    processor = ImageProcessor(device=device)
    return processor.pil_to_tensor(image, normalize)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """张量到PIL转换的便捷函数"""
    return default_processor.tensor_to_pil(tensor)


if __name__ == "__main__":
    # 测试图像处理功能
    print("测试图像处理器...")
    
    # 创建测试图像
    test_image = Image.new('RGB', (256, 256), color='red')
    
    # 测试转换
    processor = ImageProcessor(target_size=512)
    tensor = processor.pil_to_tensor(test_image)
    back_to_pil = processor.tensor_to_pil(tensor)
    
    print(f"原始图像尺寸: {test_image.size}")
    print(f"张量形状: {tensor.shape}")
    print(f"转换回PIL尺寸: {back_to_pil.size}")
    
    # 测试验证
    validation = processor.validate_image(test_image)
    print(f"验证结果: {validation}")
    
    print("图像处理器测试完成！") 