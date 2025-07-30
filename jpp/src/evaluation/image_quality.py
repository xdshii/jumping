"""
图像质量评估模块

该模块提供全面的图像质量评估功能，用于评估隐私保护算法
对图像视觉质量的影响，确保保护效果与图像质量的平衡。

功能包括：
1. 感知质量评估 (LPIPS)
2. 传统质量指标 (PSNR, SSIM, MSE)
3. 结构相似性分析
4. 批量质量评估
5. 质量报告生成和可视化

基于计算机视觉中的标准图像质量评估方法。

作者: AI Privacy Protection System
日期: 2025-07-28
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import json
from datetime import datetime
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.warning("LPIPS库不可用，感知质量评估功能将被禁用")

try:
    from ..utils.image_utils import ImageProcessor
    from ..losses.lpips_loss import create_lpips_loss
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.image_utils import ImageProcessor
    from losses.lpips_loss import create_lpips_loss

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """图像质量指标数据类"""
    lpips: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    structural_similarity: Optional[float] = None
    perceptual_distance: Optional[float] = None
    overall_score: Optional[float] = None

@dataclass
class QualityEvaluationResult:
    """质量评估结果数据类"""
    image_pair_id: str
    metrics: QualityMetrics
    image_size: Tuple[int, int]
    color_channels: int
    metadata: Optional[Dict[str, Any]] = None

class ImageQualityEvaluator:
    """图像质量评估器"""
    
    def __init__(
        self,
        lpips_net: str = "alex",
        lpips_version: str = "0.1",
        device: str = None,
        enable_lpips: bool = True
    ):
        """
        初始化图像质量评估器
        
        Args:
            lpips_net: LPIPS网络类型 ("alex", "vgg", "squeeze")
            lpips_version: LPIPS版本
            device: 设备
            enable_lpips: 是否计算LPIPS
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_lpips = enable_lpips and LPIPS_AVAILABLE
        
        # 初始化工具
        self.image_processor = ImageProcessor()
        
        # 初始化LPIPS
        self.lpips_model = None
        if self.compute_lpips:
            try:
                self.lpips_model = create_lpips_loss(
                    net=lpips_net,
                    version=lpips_version,
                    use_gpu=(self.device == "cuda")
                )
                logger.info(f"LPIPS模型加载成功: {lpips_net} v{lpips_version}")
            except Exception as e:
                logger.warning(f"LPIPS模型加载失败: {e}")
                self.compute_lpips = False
        
        logger.info(f"图像质量评估器初始化: 设备={self.device}, LPIPS={self.compute_lpips}")
    
    def preprocess_images(
        self,
        img1: Union[torch.Tensor, np.ndarray, str],
        img2: Union[torch.Tensor, np.ndarray, str]
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        预处理图像对
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            (img1_np, img2_np, img1_tensor, img2_tensor)
        """
        # 转换为numpy数组（用于传统指标）
        if isinstance(img1, str):
            img1_pil = Image.open(img1).convert('RGB')
            img1_np = np.array(img1_pil)
        elif isinstance(img1, torch.Tensor):
            img1_np = self.image_processor.tensor_to_numpy(img1)
        else:
            img1_np = img1
        
        if isinstance(img2, str):
            img2_pil = Image.open(img2).convert('RGB')
            img2_np = np.array(img2_pil)
        elif isinstance(img2, torch.Tensor):
            img2_np = self.image_processor.tensor_to_numpy(img2)
        else:
            img2_np = img2
        
        # 确保尺寸匹配
        if img1_np.shape != img2_np.shape:
            min_h = min(img1_np.shape[0], img2_np.shape[0])
            min_w = min(img1_np.shape[1], img2_np.shape[1])
            img1_np = cv2.resize(img1_np, (min_w, min_h))
            img2_np = cv2.resize(img2_np, (min_w, min_h))
        
        # 转换为torch张量（用于LPIPS）
        # numpy -> PIL -> tensor 转换链
        img1_pil = self.image_processor.numpy_to_pil(img1_np)
        img2_pil = self.image_processor.numpy_to_pil(img2_np)
        img1_tensor = self.image_processor.pil_to_tensor(img1_pil).to(self.device)
        img2_tensor = self.image_processor.pil_to_tensor(img2_pil).to(self.device)
        
        # 确保张量在[0,1]范围内
        if img1_tensor.max() > 1.1 or img2_tensor.max() > 1.1:
            img1_tensor = img1_tensor / 255.0
            img2_tensor = img2_tensor / 255.0
        
        return img1_np, img2_np, img1_tensor, img2_tensor
    
    def _compute_lpips(
        self,
        img1_tensor: torch.Tensor,
        img2_tensor: torch.Tensor
    ) -> float:
        """
        计算LPIPS感知距离
        
        Args:
            img1_tensor: 第一张图像张量
            img2_tensor: 第二张图像张量
            
        Returns:
            LPIPS距离值
        """
        if not self.compute_lpips or self.lpips_model is None:
            return None
        
        try:
            with torch.no_grad():
                # 确保张量维度正确 [B, C, H, W]
                if img1_tensor.dim() == 3:
                    img1_tensor = img1_tensor.unsqueeze(0)
                if img2_tensor.dim() == 3:
                    img2_tensor = img2_tensor.unsqueeze(0)
                
                lpips_distance = self.lpips_model(img1_tensor, img2_tensor)
                return lpips_distance.item()
        except Exception as e:
            logger.warning(f"LPIPS计算失败: {e}")
            return None
    
    def compute_psnr(
        self,
        img1_np: np.ndarray,
        img2_np: np.ndarray
    ) -> float:
        """
        计算PSNR (Peak Signal-to-Noise Ratio)
        
        Args:
            img1_np: 第一张图像数组
            img2_np: 第二张图像数组
            
        Returns:
            PSNR值 (dB)
        """
        try:
            # 确保数据类型和范围
            if img1_np.dtype != np.uint8:
                img1_np = (img1_np * 255).astype(np.uint8)
            if img2_np.dtype != np.uint8:
                img2_np = (img2_np * 255).astype(np.uint8)
            
            return psnr(img1_np, img2_np, data_range=255)
        except Exception as e:
            logger.warning(f"PSNR计算失败: {e}")
            return None
    
    def compute_ssim(
        self,
        img1_np: np.ndarray,
        img2_np: np.ndarray
    ) -> float:
        """
        计算SSIM (Structural Similarity Index)
        
        Args:
            img1_np: 第一张图像数组
            img2_np: 第二张图像数组
            
        Returns:
            SSIM值 [0, 1]
        """
        try:
            # 确保数据类型和范围
            if img1_np.dtype != np.uint8:
                img1_np = (img1_np * 255).astype(np.uint8)
            if img2_np.dtype != np.uint8:
                img2_np = (img2_np * 255).astype(np.uint8)
            
            # 对于彩色图像，计算每个通道的SSIM然后平均
            if len(img1_np.shape) == 3:
                ssim_values = []
                for channel in range(img1_np.shape[2]):
                    ssim_val = ssim(
                        img1_np[:, :, channel],
                        img2_np[:, :, channel],
                        data_range=255
                    )
                    ssim_values.append(ssim_val)
                return np.mean(ssim_values)
            else:
                return ssim(img1_np, img2_np, data_range=255)
        except Exception as e:
            logger.warning(f"SSIM计算失败: {e}")
            return None
    
    def compute_mse(
        self,
        img1_np: np.ndarray,
        img2_np: np.ndarray
    ) -> float:
        """
        计算MSE (Mean Squared Error)
        
        Args:
            img1_np: 第一张图像数组
            img2_np: 第二张图像数组
            
        Returns:
            MSE值
        """
        try:
            # 转换为浮点数以避免溢出
            img1_float = img1_np.astype(np.float64)
            img2_float = img2_np.astype(np.float64)
            
            return mse(img1_float, img2_float)
        except Exception as e:
            logger.warning(f"MSE计算失败: {e}")
            return None
    
    def compute_mae(
        self,
        img1_np: np.ndarray,
        img2_np: np.ndarray
    ) -> float:
        """
        计算MAE (Mean Absolute Error)
        
        Args:
            img1_np: 第一张图像数组
            img2_np: 第二张图像数组
            
        Returns:
            MAE值
        """
        try:
            # 转换为浮点数
            img1_float = img1_np.astype(np.float64)
            img2_float = img2_np.astype(np.float64)
            
            return np.mean(np.abs(img1_float - img2_float))
        except Exception as e:
            logger.warning(f"MAE计算失败: {e}")
            return None
    
    def compute_overall_score(
        self,
        metrics: QualityMetrics,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        计算综合质量分数
        
        Args:
            metrics: 质量指标
            weights: 各指标权重
            
        Returns:
            综合分数 [0, 1]，越高越好
        """
        if weights is None:
            weights = {
                'lpips': 0.4,    # LPIPS越小越好
                'ssim': 0.3,     # SSIM越大越好
                'psnr': 0.2,     # PSNR越大越好
                'mse': 0.1       # MSE越小越好
            }
        
        score = 0.0
        total_weight = 0.0
        
        # LPIPS分数 (越小越好，转换为越大越好)
        if metrics.lpips is not None and 'lpips' in weights:
            lpips_score = max(0, 1.0 - metrics.lpips)  # 假设LPIPS通常在[0, 1]范围
            score += weights['lpips'] * lpips_score
            total_weight += weights['lpips']
        
        # SSIM分数 (越大越好)
        if metrics.ssim is not None and 'ssim' in weights:
            ssim_score = max(0, min(1, metrics.ssim))
            score += weights['ssim'] * ssim_score
            total_weight += weights['ssim']
        
        # PSNR分数 (越大越好，需要归一化)
        if metrics.psnr is not None and 'psnr' in weights:
            # PSNR通常在10-50dB范围，归一化到[0,1]
            psnr_score = max(0, min(1, (metrics.psnr - 10) / 40))
            score += weights['psnr'] * psnr_score
            total_weight += weights['psnr']
        
        # MSE分数 (越小越好，需要转换)
        if metrics.mse is not None and 'mse' in weights:
            # MSE归一化比较复杂，这里使用简单的指数衰减
            mse_score = np.exp(-metrics.mse / 1000.0)  # 调整衰减因子
            score += weights['mse'] * mse_score
            total_weight += weights['mse']
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def evaluate_single_pair(
        self,
        img1: Union[torch.Tensor, np.ndarray, str],
        img2: Union[torch.Tensor, np.ndarray, str],
        pair_id: str = "unknown"
    ) -> QualityEvaluationResult:
        """
        评估单对图像
        
        Args:
            img1: 第一张图像（通常是原始图像）
            img2: 第二张图像（通常是处理后图像）
            pair_id: 图像对ID
            
        Returns:
            质量评估结果
        """
        # 预处理图像
        img1_np, img2_np, img1_tensor, img2_tensor = self.preprocess_images(img1, img2)
        
        # 计算各项指标
        metrics = QualityMetrics()
        
        # LPIPS
        if self.compute_lpips:
            metrics.lpips = self._compute_lpips(img1_tensor, img2_tensor)
        
        # 传统指标
        metrics.psnr = self.compute_psnr(img1_np, img2_np)
        metrics.ssim = self.compute_ssim(img1_np, img2_np)
        metrics.mse = self.compute_mse(img1_np, img2_np)
        metrics.mae = self.compute_mae(img1_np, img2_np)
        
        # 结构相似性（使用SSIM）
        metrics.structural_similarity = metrics.ssim
        
        # 感知距离（使用LPIPS）
        metrics.perceptual_distance = metrics.lpips
        
        # 综合分数
        metrics.overall_score = self.compute_overall_score(metrics)
        
        # 创建结果
        result = QualityEvaluationResult(
            image_pair_id=pair_id,
            metrics=metrics,
            image_size=(img1_np.shape[1], img1_np.shape[0]),  # (width, height)
            color_channels=img1_np.shape[2] if len(img1_np.shape) == 3 else 1,
            metadata={
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluator_device": self.device,
                "compute_lpips": self.compute_lpips
            }
        )
        
        return result
    
    def evaluate_batch(
        self,
        image_pairs: List[Tuple[Union[torch.Tensor, np.ndarray, str], Union[torch.Tensor, np.ndarray, str]]],
        pair_ids: Optional[List[str]] = None
    ) -> List[QualityEvaluationResult]:
        """
        批量评估图像对
        
        Args:
            image_pairs: 图像对列表
            pair_ids: 图像对ID列表
            
        Returns:
            评估结果列表
        """
        if pair_ids is None:
            pair_ids = [f"pair_{i:03d}" for i in range(len(image_pairs))]
        
        results = []
        
        for i, (img1, img2) in enumerate(image_pairs):
            try:
                result = self.evaluate_single_pair(img1, img2, pair_ids[i])
                results.append(result)
                logger.info(f"评估完成: {pair_ids[i]} - LPIPS={result.metrics.lpips:.4f if result.metrics.lpips else 'N/A'}, SSIM={result.metrics.ssim:.4f if result.metrics.ssim else 'N/A'}")
            except Exception as e:
                logger.error(f"评估失败 {pair_ids[i]}: {e}")
                continue
        
        return results
    
    def analyze_quality_statistics(
        self,
        results: List[QualityEvaluationResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        分析质量统计信息
        
        Args:
            results: 评估结果列表
            
        Returns:
            统计信息字典
        """
        if not results:
            return {}
        
        # 提取各项指标
        metrics_data = {
            'lpips': [r.metrics.lpips for r in results if r.metrics.lpips is not None],
            'psnr': [r.metrics.psnr for r in results if r.metrics.psnr is not None],
            'ssim': [r.metrics.ssim for r in results if r.metrics.ssim is not None],
            'mse': [r.metrics.mse for r in results if r.metrics.mse is not None],
            'mae': [r.metrics.mae for r in results if r.metrics.mae is not None],
            'overall_score': [r.metrics.overall_score for r in results if r.metrics.overall_score is not None]
        }
        
        # 计算统计信息
        statistics = {}
        for metric_name, values in metrics_data.items():
            if values:
                statistics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
            else:
                statistics[metric_name] = {
                    'mean': None, 'std': None, 'min': None,
                    'max': None, 'median': None, 'count': 0
                }
        
        return statistics
    
    def generate_quality_report(
        self,
        results: List[QualityEvaluationResult],
        statistics: Dict[str, Dict[str, float]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成质量评估报告
        
        Args:
            results: 评估结果列表
            statistics: 统计信息
            save_path: 保存路径
            
        Returns:
            完整报告
        """
        if statistics is None:
            statistics = self.analyze_quality_statistics(results)
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "evaluator_config": {
                    "device": self.device,
                    "compute_lpips": self.compute_lpips
                },
                "total_pairs": len(results)
            },
            "statistics": statistics,
            "individual_results": [],
            "summary": {}
        }
        
        # 个别结果
        for result in results:
            report["individual_results"].append({
                "pair_id": result.image_pair_id,
                "metrics": {
                    "lpips": result.metrics.lpips,
                    "psnr": result.metrics.psnr,
                    "ssim": result.metrics.ssim,
                    "mse": result.metrics.mse,
                    "mae": result.metrics.mae,
                    "overall_score": result.metrics.overall_score
                },
                "image_info": {
                    "size": result.image_size,
                    "channels": result.color_channels
                }
            })
        
        # 摘要
        if statistics:
            report["summary"] = {
                "quality_rating": self._rate_quality(statistics),
                "recommendations": self._generate_quality_recommendations(statistics),
                "key_findings": self._extract_key_findings(statistics)
            }
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"质量评估报告已保存到: {save_path}")
        
        return report
    
    def _rate_quality(self, statistics: Dict[str, Dict[str, float]]) -> str:
        """评估整体质量等级"""
        overall_mean = statistics.get('overall_score', {}).get('mean')
        if overall_mean is None:
            return "Unknown"
        
        if overall_mean >= 0.8:
            return "Excellent"
        elif overall_mean >= 0.6:
            return "Good"
        elif overall_mean >= 0.4:
            return "Fair"
        elif overall_mean >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _generate_quality_recommendations(self, statistics: Dict[str, Dict[str, float]]) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        # LPIPS分析
        lpips_mean = statistics.get('lpips', {}).get('mean')
        if lpips_mean is not None:
            if lpips_mean > 0.2:
                recommendations.append("LPIPS感知距离较高，建议降低保护强度或优化算法以改善感知质量")
            elif lpips_mean < 0.05:
                recommendations.append("LPIPS感知质量优秀，可考虑适当增强保护强度")
        
        # SSIM分析
        ssim_mean = statistics.get('ssim', {}).get('mean')
        if ssim_mean is not None:
            if ssim_mean < 0.8:
                recommendations.append("SSIM结构相似性偏低，建议优化算法以保持图像结构")
            elif ssim_mean > 0.95:
                recommendations.append("SSIM结构相似性很高，保护效果可能不足")
        
        # PSNR分析
        psnr_mean = statistics.get('psnr', {}).get('mean')
        if psnr_mean is not None:
            if psnr_mean < 20:
                recommendations.append("PSNR信噪比偏低，建议减少图像失真")
            elif psnr_mean > 40:
                recommendations.append("PSNR信噪比很高，可考虑增强保护强度")
        
        if not recommendations:
            recommendations.append("图像质量指标均在合理范围内")
        
        return recommendations
    
    def _extract_key_findings(self, statistics: Dict[str, Dict[str, float]]) -> List[str]:
        """提取关键发现"""
        findings = []
        
        for metric_name, stats in statistics.items():
            if stats['count'] > 0:
                findings.append(
                    f"{metric_name.upper()}: 平均={stats['mean']:.4f}, "
                    f"标准差={stats['std']:.4f}, 范围=[{stats['min']:.4f}, {stats['max']:.4f}]"
                )
        
        return findings
    
    def plot_quality_metrics(
        self,
        results: List[QualityEvaluationResult],
        save_path: Optional[str] = None
    ):
        """
        绘制质量指标图表
        
        Args:
            results: 评估结果列表
            save_path: 保存路径
        """
        if not results:
            logger.warning("没有结果可绘制")
            return
        
        # 提取指标数据
        metrics_data = {}
        valid_metrics = []
        
        if any(r.metrics.lpips is not None for r in results):
            metrics_data['LPIPS'] = [r.metrics.lpips for r in results if r.metrics.lpips is not None]
            valid_metrics.append('LPIPS')
        
        if any(r.metrics.ssim is not None for r in results):
            metrics_data['SSIM'] = [r.metrics.ssim for r in results if r.metrics.ssim is not None]
            valid_metrics.append('SSIM')
        
        if any(r.metrics.psnr is not None for r in results):
            metrics_data['PSNR'] = [r.metrics.psnr for r in results if r.metrics.psnr is not None]
            valid_metrics.append('PSNR')
        
        if any(r.metrics.overall_score is not None for r in results):
            metrics_data['Overall Score'] = [r.metrics.overall_score for r in results if r.metrics.overall_score is not None]
            valid_metrics.append('Overall Score')
        
        # 创建子图
        n_metrics = len(valid_metrics)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(valid_metrics[:4]):
            if i < len(axes):
                data = metrics_data[metric]
                
                # 直方图
                axes[i].hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'{metric} Distribution')
                axes[i].set_xlabel(metric)
                axes[i].set_ylabel('Frequency')
                
                # 添加统计信息
                mean_val = np.mean(data)
                std_val = np.std(data)
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_val + std_val:.3f}')
                axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_val - std_val:.3f}')
                axes[i].legend()
        
        # 隐藏多余的子图
        for i in range(len(valid_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"质量指标图表已保存到: {save_path}")
        
        plt.close()

def create_image_quality_evaluator(
    lpips_net: str = "alex",
    device: str = None,
    **kwargs
) -> ImageQualityEvaluator:
    """
    创建图像质量评估器的便捷函数
    
    Args:
        lpips_net: LPIPS网络类型
        device: 设备
        **kwargs: 其他参数
        
    Returns:
        质量评估器实例
    """
    return ImageQualityEvaluator(
        lpips_net=lpips_net,
        device=device,
        **kwargs
    )

def test_image_quality_evaluator():
    """测试图像质量评估器"""
    print("🧪 测试图像质量评估器...")
    
    try:
        # 创建评估器
        evaluator = create_image_quality_evaluator()
        
        print(f"✅ 质量评估器创建成功")
        print(f"   设备: {evaluator.device}")
        print(f"   LPIPS支持: {evaluator.compute_lpips}")
        
        # 创建测试图像
        np.random.seed(42)
        
        # 原始图像
        original_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # 模拟保护后的图像（添加一些噪声）
        noise = np.random.normal(0, 10, original_img.shape).astype(np.int16)
        protected_img = np.clip(original_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        print("🔮 测试单对图像评估...")
        
        # 评估单对图像
        result = evaluator.evaluate_single_pair(original_img, protected_img, "test_pair")
        
        print(f"✅ 单对评估成功:")
        if result.metrics.lpips is not None:
            print(f"   LPIPS: {result.metrics.lpips:.4f}")
        if result.metrics.psnr is not None:
            print(f"   PSNR: {result.metrics.psnr:.2f} dB")
        if result.metrics.ssim is not None:
            print(f"   SSIM: {result.metrics.ssim:.4f}")
        if result.metrics.mse is not None:
            print(f"   MSE: {result.metrics.mse:.2f}")
        if result.metrics.overall_score is not None:
            print(f"   综合分数: {result.metrics.overall_score:.4f}")
        
        # 测试批量评估
        print("📊 测试批量评估...")
        
        # 创建多个测试图像对
        image_pairs = []
        for i in range(5):
            # 生成不同程度的失真
            noise_level = (i + 1) * 5
            noise = np.random.normal(0, noise_level, original_img.shape).astype(np.int16)
            distorted_img = np.clip(original_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image_pairs.append((original_img, distorted_img))
        
        batch_results = evaluator.evaluate_batch(image_pairs)
        
        print(f"✅ 批量评估成功: {len(batch_results)}对图像")
        
        # 测试统计分析
        print("📈 测试统计分析...")
        statistics = evaluator.analyze_quality_statistics(batch_results)
        
        for metric_name, stats in statistics.items():
            if stats['count'] > 0:
                print(f"   {metric_name}: 平均={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
        
        # 测试报告生成
        print("📋 测试报告生成...")
        report = evaluator.generate_quality_report(batch_results, statistics)
        
        print(f"✅ 报告生成成功:")
        print(f"   评估对数: {report['metadata']['total_pairs']}")
        print(f"   质量评级: {report['summary']['quality_rating']}")
        print(f"   建议数量: {len(report['summary']['recommendations'])}")
        
        print("🎉 图像质量评估器测试全部通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_image_quality_evaluator() 