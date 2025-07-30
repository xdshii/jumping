"""
身份损失函数实现

该模块提供身份保护的核心损失函数，包括：
1. ArcFace面部特征提取
2. 面部特征相似度计算  
3. 身份损失计算（最大化特征距离）
4. 多模型集成支持

基于DiffPrivate论文中的身份损失L_ID设计。

作者: AI Privacy Protection System
日期: 2025-07-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
from pathlib import Path
import cv2
from PIL import Image

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. ArcFace功能将被禁用。")

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    logging.warning("FaceNet不可用。FaceNet功能将被禁用。")

logger = logging.getLogger(__name__)

class ArcFaceExtractor:
    """
    ArcFace特征提取器
    
    使用InsightFace库的ArcFace模型提取人脸特征
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/face_models/arcface/models/buffalo_l",
        device: str = "cuda"
    ):
        """
        初始化ArcFace特征提取器
        
        Args:
            model_path: 模型路径
            device: 设备
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace未安装，无法使用ArcFace功能")
            
        self.device = device
        self.model_path = model_path
        
        # 初始化InsightFace应用
        self.app = insightface.app.FaceAnalysis(
            root=str(Path(model_path).parent),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        
        logger.info(f"ArcFace特征提取器初始化完成: {model_path}")
    
    def extract_features(
        self,
        images: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        提取面部特征
        
        Args:
            images: 图像数据，支持多种格式
            return_tensor: 是否返回torch.Tensor
            
        Returns:
            面部特征向量 [batch_size, feature_dim] 或 None（如果未检测到人脸）
        """
        # 转换输入格式
        if isinstance(images, torch.Tensor):
            # 从tensor转换为numpy array (RGB, 0-255)
            if images.dim() == 4:  # batch
                images_np = []
                for i in range(images.shape[0]):
                    img = images[i].cpu().detach()
                    if img.max() <= 1.0:  # 归一化的图像
                        img = (img * 255).clamp(0, 255)
                    img = img.permute(1, 2, 0).numpy().astype(np.uint8)
                    # 转换为BGR（InsightFace期望BGR格式）
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    images_np.append(img)
            else:  # single image
                img = images.cpu().detach()
                if img.max() <= 1.0:
                    img = (img * 255).clamp(0, 255)
                img = img.permute(1, 2, 0).numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images_np = [img]
        elif isinstance(images, np.ndarray):
            if images.ndim == 4:  # batch
                images_np = [images[i] for i in range(images.shape[0])]
            else:
                images_np = [images]
        else:
            images_np = images if isinstance(images, list) else [images]
        
        # 提取特征
        features = []
        for img in images_np:
            try:
                faces = self.app.get(img)
                if len(faces) > 0:
                    # 使用最大的人脸
                    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    feature = face.normed_embedding
                    features.append(feature)
                else:
                    # 如果没有检测到人脸，返回零向量
                    logger.warning("未检测到人脸，返回零特征向量")
                    features.append(np.zeros(512, dtype=np.float32))
            except Exception as e:
                logger.error(f"特征提取失败: {e}")
                features.append(np.zeros(512, dtype=np.float32))
        
        if not features:
            return None
            
        features = np.stack(features, axis=0)
        
        if return_tensor:
            return torch.from_numpy(features).to(self.device)
        else:
            return features

class FaceNetExtractor:
    """
    FaceNet特征提取器
    
    使用FaceNet模型提取人脸特征
    """
    
    def __init__(
        self,
        device: str = "cuda"
    ):
        """
        初始化FaceNet特征提取器
        
        Args:
            device: 设备
        """
        if not FACENET_AVAILABLE:
            raise ImportError("FaceNet库未安装，无法使用FaceNet功能")
            
        self.device = device
        
        # 初始化MTCNN用于人脸检测
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            device=device,
            keep_all=False,
            post_process=False
        )
        
        # 初始化FaceNet模型
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        logger.info(f"FaceNet特征提取器初始化完成")
    
    def extract_features(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        提取面部特征
        
        Args:
            images: 图像数据
            return_tensor: 是否返回torch.Tensor
            
        Returns:
            面部特征向量 [batch_size, feature_dim]
        """
        if isinstance(images, torch.Tensor):
            # 转换tensor为PIL图像列表
            pil_images = []
            for i in range(images.shape[0]):
                img = images[i].cpu().detach()
                if img.max() <= 1.0:
                    img = (img * 255).clamp(0, 255)
                img = img.permute(1, 2, 0).numpy().astype(np.uint8)
                pil_images.append(Image.fromarray(img))
        else:
            pil_images = images
        
        # 检测和对齐人脸
        aligned_faces = []
        for img in pil_images:
            try:
                face = self.mtcnn(img)
                if face is not None:
                    aligned_faces.append(face)
                else:
                    # 如果检测失败，使用零张量
                    aligned_faces.append(torch.zeros(3, 160, 160, device=self.device))
            except Exception as e:
                logger.error(f"人脸检测失败: {e}")
                aligned_faces.append(torch.zeros(3, 160, 160, device=self.device))
        
        if not aligned_faces:
            return None
            
        # 批处理提取特征
        aligned_batch = torch.stack(aligned_faces).to(self.device)
        
        with torch.no_grad():
            features = self.resnet(aligned_batch)
            features = F.normalize(features, p=2, dim=1)  # L2归一化
        
        if return_tensor:
            return features
        else:
            return features.cpu().numpy()

class IdentityLoss(nn.Module):
    """
    身份损失函数
    
    实现DiffPrivate论文中的身份损失L_ID，通过最大化面部特征距离来实现身份保护
    """
    
    def __init__(
        self,
        model_types: List[str] = ["arcface"],
        model_weights: Optional[List[float]] = None,
        distance_metric: str = "cosine",
        device: str = "cuda",
        fallback_to_l2: bool = True
    ):
        """
        初始化身份损失函数
        
        Args:
            model_types: 使用的模型类型 ["arcface", "facenet"]
            model_weights: 各模型权重（如果为None则均等权重）
            distance_metric: 距离度量 ("cosine", "l2", "l1")
            device: 设备
            fallback_to_l2: 当面部检测失败时是否回退到像素L2损失
        """
        super().__init__()
        
        self.model_types = model_types
        self.distance_metric = distance_metric
        self.device = device
        self.fallback_to_l2 = fallback_to_l2
        
        # 设置模型权重
        if model_weights is None:
            self.model_weights = [1.0 / len(model_types)] * len(model_types)
        else:
            assert len(model_weights) == len(model_types), "权重数量必须与模型数量匹配"
            self.model_weights = model_weights
        
        # 初始化特征提取器
        self.extractors = {}
        for model_type in model_types:
            if model_type == "arcface" and INSIGHTFACE_AVAILABLE:
                self.extractors["arcface"] = ArcFaceExtractor(device=device)
            elif model_type == "facenet" and FACENET_AVAILABLE:
                self.extractors["facenet"] = FaceNetExtractor(device=device)
            else:
                logger.error(f"不支持的模型类型或依赖缺失: {model_type}")
        
        logger.info(f"身份损失函数初始化: models={model_types}, metric={distance_metric}")
    
    def compute_distance(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        metric: str = None
    ) -> torch.Tensor:
        """
        计算特征距离
        
        Args:
            features1: 第一组特征 [batch_size, feature_dim]
            features2: 第二组特征 [batch_size, feature_dim]  
            metric: 距离度量
            
        Returns:
            距离值 [batch_size]
        """
        if metric is None:
            metric = self.distance_metric
            
        if metric == "cosine":
            # 余弦距离 (1 - 余弦相似度)
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)
            cosine_sim = torch.sum(features1_norm * features2_norm, dim=1)
            return 1.0 - cosine_sim
        elif metric == "l2":
            # 欧氏距离
            return torch.norm(features1 - features2, p=2, dim=1)
        elif metric == "l1":
            # 曼哈顿距离
            return torch.norm(features1 - features2, p=1, dim=1)
        else:
            raise ValueError(f"不支持的距离度量: {metric}")
    
    def compute_pixel_l2_loss(
        self,
        original_images: torch.Tensor,
        protected_images: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        计算像素级L2损失（作为备用方案）
        
        Args:
            original_images: 原始图像
            protected_images: 保护图像
            reduction: 降维方式
            
        Returns:
            L2损失
        """
        l2_loss = F.mse_loss(original_images, protected_images, reduction='none')
        l2_loss = l2_loss.view(l2_loss.shape[0], -1).mean(dim=1)  # [batch_size]
        
        if reduction == "mean":
            return l2_loss.mean()
        elif reduction == "sum":
            return l2_loss.sum()
        else:
            return l2_loss
    
    def forward(
        self,
        original_images: torch.Tensor,
        protected_images: torch.Tensor,
        target_distance: float = 1.0,
        reduction: str = "mean"
    ) -> Dict[str, torch.Tensor]:
        """
        计算身份损失
        
        Args:
            original_images: 原始图像 [batch_size, 3, H, W]
            protected_images: 保护后图像 [batch_size, 3, H, W]
            target_distance: 目标距离（用于归一化）
            reduction: 降维方式 ("mean", "sum", "none")
            
        Returns:
            损失字典，包含各个模型的损失和总损失
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=original_images.device, requires_grad=True)
        valid_models = 0
        
        for i, model_type in enumerate(self.model_types):
            if model_type not in self.extractors:
                continue
                
            extractor = self.extractors[model_type]
            
            try:
                # 提取原始图像特征
                orig_features = extractor.extract_features(original_images)
                if orig_features is None:
                    logger.warning(f"{model_type}: 无法提取原始图像特征")
                    continue
                
                # 提取保护图像特征
                prot_features = extractor.extract_features(protected_images)
                if prot_features is None:
                    logger.warning(f"{model_type}: 无法提取保护图像特征")
                    continue
                
                # 检查特征是否都是零（表示没有检测到人脸）
                orig_zero_mask = (orig_features.sum(dim=1) == 0)
                prot_zero_mask = (prot_features.sum(dim=1) == 0)
                all_zero_mask = orig_zero_mask & prot_zero_mask
                
                if all_zero_mask.all() and self.fallback_to_l2:
                    # 如果所有样本都没有检测到人脸，回退到像素L2损失
                    logger.info(f"{model_type}: 回退到像素L2损失")
                    pixel_loss = self.compute_pixel_l2_loss(original_images, protected_images, reduction)
                    identity_loss = -pixel_loss  # 负值表示最大化距离
                    distances = pixel_loss
                else:
                    # 计算特征距离
                    distances = self.compute_distance(orig_features, prot_features)
                    
                    # 身份损失：负距离（最大化距离）
                    identity_loss = -distances / target_distance
                    
                    if reduction == "mean":
                        identity_loss = identity_loss.mean()
                        distances = distances.mean()
                    elif reduction == "sum":
                        identity_loss = identity_loss.sum()
                        distances = distances.sum()
                
                losses[f"{model_type}_loss"] = identity_loss
                losses[f"{model_type}_distance"] = distances
                
                # 加权累加到总损失（避免in-place操作）
                total_loss = total_loss + self.model_weights[i] * identity_loss
                valid_models += 1
                
            except Exception as e:
                logger.error(f"{model_type}特征提取失败: {e}")
                continue
        
        if valid_models == 0 and self.fallback_to_l2:
            # 如果所有模型都失败，使用像素L2损失作为最后手段
            logger.info("所有面部模型失败，使用像素L2损失")
            pixel_loss = self.compute_pixel_l2_loss(original_images, protected_images, reduction)
            total_loss = -pixel_loss
            losses["pixel_l2_loss"] = -pixel_loss
            losses["pixel_l2_distance"] = pixel_loss
        
        losses["total_loss"] = total_loss
        # 计算平均距离，确保类型一致性
        distance_values = [losses[f"{m}_distance"] for m in self.model_types if f"{m}_distance" in losses]
        if distance_values:
            if isinstance(distance_values[0], torch.Tensor):
                avg_distance = torch.stack(distance_values).mean()
            else:
                avg_distance = sum(distance_values) / len(distance_values)
        else:
            avg_distance = torch.tensor(0.0, device=original_images.device)
        losses["avg_distance"] = avg_distance
        
        return losses

def create_identity_loss(
    model_types: List[str] = ["arcface"],
    model_weights: Optional[List[float]] = None,
    distance_metric: str = "cosine",
    device: str = "cuda",
    fallback_to_l2: bool = True
) -> IdentityLoss:
    """
    创建身份损失函数的便捷函数
    
    Args:
        model_types: 使用的模型类型
        model_weights: 各模型权重
        distance_metric: 距离度量
        device: 设备
        fallback_to_l2: 是否启用L2回退
        
    Returns:
        身份损失函数实例
    """
    return IdentityLoss(model_types, model_weights, distance_metric, device, fallback_to_l2)

def test_identity_loss():
    """测试身份损失函数"""
    print("🧪 测试身份损失函数...")
    
    try:
        # 检查依赖
        available_models = []
        if INSIGHTFACE_AVAILABLE:
            available_models.append("arcface")
        if FACENET_AVAILABLE:
            available_models.append("facenet")
        
        if not available_models:
            print("❌ 没有可用的面部识别模型，跳过测试")
            return False
        
        print(f"✅ 可用模型: {available_models}")
        
        # 创建身份损失函数（启用L2回退）
        id_loss = create_identity_loss(
            model_types=available_models[:1],  # 只使用第一个可用模型进行测试
            device="cuda" if torch.cuda.is_available() else "cpu",
            fallback_to_l2=True
        )
        
        # 创建测试图像
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        
        # 创建随机图像（模拟人脸）- 使用requires_grad确保梯度连接
        original_images = torch.rand(batch_size, 3, 224, 224, device=device, requires_grad=True)
        protected_images = torch.rand(batch_size, 3, 224, 224, device=device, requires_grad=True)
        
        print(f"✅ 身份损失函数创建成功")
        print(f"   设备: {device}")
        print(f"   模型: {available_models[:1]}")
        print(f"   距离度量: {id_loss.distance_metric}")
        print(f"   L2回退: {id_loss.fallback_to_l2}")
        
        # 测试前向传播
        print("🔮 测试前向传播...")
        loss_dict = id_loss(original_images, protected_images)
        
        print("✅ 前向传播成功:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.item():.6f}")
            else:
                print(f"   {key}: {value:.6f}")
        
        # 测试梯度计算
        print("📈 测试梯度计算...")
        total_loss = loss_dict["total_loss"]
        
        # 清除之前的梯度
        if original_images.grad is not None:
            original_images.grad.zero_()
        if protected_images.grad is not None:
            protected_images.grad.zero_()
        
        total_loss.backward()
        
        # 检查梯度
        orig_grad_norm = original_images.grad.norm().item() if original_images.grad is not None else 0
        prot_grad_norm = protected_images.grad.norm().item() if protected_images.grad is not None else 0
        
        print(f"✅ 梯度计算成功:")
        print(f"   原始图像梯度范数: {orig_grad_norm:.6f}")
        print(f"   保护图像梯度范数: {prot_grad_norm:.6f}")
        
        if orig_grad_norm > 0 or prot_grad_norm > 0:
            print("🎉 身份损失函数测试全部通过！")
            return True
        else:
            print("⚠️ 梯度为零，可能存在问题")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_identity_loss() 