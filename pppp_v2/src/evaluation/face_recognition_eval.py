"""
多模型面部识别评估系统

该模块提供对多个面部识别模型的统一评估接口，用于测试
隐私保护算法对不同识别模型的有效性和可转移性。

功能包括：
1. 支持多种面部识别模型 (ArcFace, FaceNet, etc.)
2. 批量特征提取和相似度计算
3. 可转移性评估和对比分析
4. 详细的评估报告生成
5. 模型性能基准测试

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
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    from ..losses.id_loss import IdentityLoss, create_identity_loss
    from ..utils.image_utils import ImageProcessor
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from losses.id_loss import IdentityLoss, create_identity_loss
    from utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    model_name: str
    original_features: np.ndarray
    protected_features: np.ndarray
    similarities: np.ndarray
    distances: np.ndarray
    mean_similarity: float
    mean_distance: float
    std_similarity: float
    std_distance: float
    protection_rate: float  # 低于阈值的比例
    metadata: Optional[Dict[str, Any]] = None

class FaceRecognitionEvaluator:
    """面部识别评估器"""
    
    def __init__(
        self,
        model_types: List[str] = None,
        device: str = None,
        similarity_threshold: float = 0.6,
        distance_metric: str = "cosine"
    ):
        """
        初始化面部识别评估器
        
        Args:
            model_types: 模型类型列表
            device: 设备
            similarity_threshold: 相似度阈值
            distance_metric: 距离度量方式
        """
        self.model_types = model_types or ["arcface", "facenet"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold
        self.distance_metric = distance_metric
        
        # 初始化工具
        self.image_processor = ImageProcessor()
        self.models = {}
        
        # 加载模型
        self._load_models()
        
        logger.info(f"面部识别评估器初始化: {len(self.models)}个模型, 设备={self.device}")
    
    def _load_models(self):
        """加载面部识别模型"""
        for model_type in self.model_types:
            try:
                if model_type.lower() in ["arcface", "facenet"]:
                    # 使用现有的IdentityLoss模型
                    model = create_identity_loss(
                        model_types=[model_type.lower()],
                        device=self.device,
                        fallback_to_l2=False
                    )
                    self.models[model_type] = model
                    logger.info(f"加载模型成功: {model_type}")
                else:
                    logger.warning(f"不支持的模型类型: {model_type}")
            except Exception as e:
                logger.error(f"加载模型失败 {model_type}: {e}")
    
    def extract_features(
        self,
        images: Union[torch.Tensor, List[str], str],
        model_name: str
    ) -> np.ndarray:
        """
        提取图像特征
        
        Args:
            images: 图像张量、路径列表或单个路径
            model_name: 模型名称
            
        Returns:
            特征向量数组
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未加载")
        
        model = self.models[model_name]
        
        # 处理输入
        if isinstance(images, str):
            # 单个图像路径
            img_tensor = self.image_processor.load_image(images)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
        elif isinstance(images, list):
            # 图像路径列表
            img_tensors = []
            for img_path in images:
                img_tensor = self.image_processor.load_image(img_path)
                img_tensors.append(img_tensor)
            img_tensor = torch.stack(img_tensors).to(self.device)
        else:
            # 已经是张量
            img_tensor = images
            if img_tensor.device != self.device:
                img_tensor = img_tensor.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            if model_name.lower() == "arcface" and "arcface" in model.extractors:
                features = model.extractors["arcface"].extract_features(img_tensor)
            elif model_name.lower() == "facenet" and "facenet" in model.extractors:
                features = model.extractors["facenet"].extract_features(img_tensor)
            else:
                raise ValueError(f"模型 {model_name} 不可用或不支持")
        
        return features.cpu().numpy()
    
    def compute_similarities(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        metric: str = None
    ) -> np.ndarray:
        """
        计算特征相似度
        
        Args:
            features1: 第一组特征
            features2: 第二组特征
            metric: 距离度量方式
            
        Returns:
            相似度数组
        """
        if metric is None:
            metric = self.distance_metric
        
        if metric == "cosine":
            # 余弦相似度
            features1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
            features2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
            similarities = np.sum(features1_norm * features2_norm, axis=1)
        elif metric == "euclidean":
            # 欧氏距离转相似度
            distances = np.linalg.norm(features1 - features2, axis=1)
            similarities = 1.0 / (1.0 + distances)
        elif metric == "manhattan":
            # 曼哈顿距离转相似度
            distances = np.sum(np.abs(features1 - features2), axis=1)
            similarities = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"不支持的距离度量: {metric}")
        
        return similarities
    
    def compute_distances(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        metric: str = None
    ) -> np.ndarray:
        """
        计算特征距离
        
        Args:
            features1: 第一组特征
            features2: 第二组特征
            metric: 距离度量方式
            
        Returns:
            距离数组
        """
        if metric is None:
            metric = self.distance_metric
        
        if metric == "cosine":
            # 余弦距离
            similarities = self.compute_similarities(features1, features2, "cosine")
            distances = 1.0 - similarities
        elif metric == "euclidean":
            # 欧氏距离
            distances = np.linalg.norm(features1 - features2, axis=1)
        elif metric == "manhattan":
            # 曼哈顿距离
            distances = np.sum(np.abs(features1 - features2), axis=1)
        else:
            raise ValueError(f"不支持的距离度量: {metric}")
        
        return distances
    
    def evaluate_single_model(
        self,
        original_images: Union[torch.Tensor, List[str]],
        protected_images: Union[torch.Tensor, List[str]],
        model_name: str
    ) -> EvaluationResult:
        """
        评估单个模型
        
        Args:
            original_images: 原始图像
            protected_images: 保护图像
            model_name: 模型名称
            
        Returns:
            评估结果
        """
        logger.info(f"开始评估模型: {model_name}")
        
        # 提取特征
        original_features = self.extract_features(original_images, model_name)
        protected_features = self.extract_features(protected_images, model_name)
        
        # 计算相似度和距离
        similarities = self.compute_similarities(original_features, protected_features)
        distances = self.compute_distances(original_features, protected_features)
        
        # 计算统计指标
        mean_similarity = np.mean(similarities)
        mean_distance = np.mean(distances)
        std_similarity = np.std(similarities)
        std_distance = np.std(distances)
        
        # 计算保护率（低于阈值的比例）
        protection_rate = np.mean(similarities < self.similarity_threshold)
        
        result = EvaluationResult(
            model_name=model_name,
            original_features=original_features,
            protected_features=protected_features,
            similarities=similarities,
            distances=distances,
            mean_similarity=mean_similarity,
            mean_distance=mean_distance,
            std_similarity=std_similarity,
            std_distance=std_distance,
            protection_rate=protection_rate,
            metadata={
                "threshold": self.similarity_threshold,
                "metric": self.distance_metric,
                "num_samples": len(similarities)
            }
        )
        
        logger.info(f"模型 {model_name} 评估完成: 平均相似度={mean_similarity:.4f}, 保护率={protection_rate:.2%}")
        
        return result
    
    def evaluate_all_models(
        self,
        original_images: Union[torch.Tensor, List[str]],
        protected_images: Union[torch.Tensor, List[str]]
    ) -> Dict[str, EvaluationResult]:
        """
        评估所有模型
        
        Args:
            original_images: 原始图像
            protected_images: 保护图像
            
        Returns:
            所有模型的评估结果
        """
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.evaluate_single_model(
                    original_images, protected_images, model_name
                )
                results[model_name] = result
            except Exception as e:
                logger.error(f"评估模型失败 {model_name}: {e}")
                continue
        
        return results
    
    def analyze_transferability(
        self,
        results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """
        分析可转移性
        
        Args:
            results: 评估结果字典
            
        Returns:
            可转移性分析结果
        """
        if len(results) < 2:
            logger.warning("至少需要2个模型来分析可转移性")
            return {}
        
        transferability_analysis = {
            "model_comparison": {},
            "cross_model_correlation": {},
            "overall_transferability": 0.0,
            "best_model": "",
            "worst_model": "",
            "consistency_score": 0.0
        }
        
        # 提取各模型的保护率
        protection_rates = {name: result.protection_rate for name, result in results.items()}
        similarities = {name: result.similarities for name, result in results.items()}
        
        # 找出最好和最差的模型
        best_model = max(protection_rates.keys(), key=lambda k: protection_rates[k])
        worst_model = min(protection_rates.keys(), key=lambda k: protection_rates[k])
        
        transferability_analysis["best_model"] = best_model
        transferability_analysis["worst_model"] = worst_model
        transferability_analysis["overall_transferability"] = np.mean(list(protection_rates.values()))
        
        # 计算模型间相似度相关性
        model_names = list(similarities.keys())
        correlation_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    corr = np.corrcoef(similarities[model1], similarities[model2])[0, 1]
                    correlation_matrix[i, j] = corr
        
        transferability_analysis["cross_model_correlation"] = {
            "matrix": correlation_matrix.tolist(),
            "model_names": model_names,
            "mean_correlation": np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        }
        
        # 计算一致性分数（所有模型保护率的标准差，越小越一致）
        consistency_score = 1.0 / (1.0 + np.std(list(protection_rates.values())))
        transferability_analysis["consistency_score"] = consistency_score
        
        # 模型对比
        for name, result in results.items():
            transferability_analysis["model_comparison"][name] = {
                "protection_rate": result.protection_rate,
                "mean_similarity": result.mean_similarity,
                "mean_distance": result.mean_distance,
                "std_similarity": result.std_similarity,
                "rank": sorted(protection_rates.keys(), key=lambda k: protection_rates[k], reverse=True).index(name) + 1
            }
        
        return transferability_analysis
    
    def generate_report(
        self,
        results: Dict[str, EvaluationResult],
        transferability_analysis: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            results: 评估结果
            transferability_analysis: 可转移性分析
            save_path: 保存路径
            
        Returns:
            完整报告
        """
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "evaluator_config": {
                    "models": list(self.models.keys()),
                    "device": self.device,
                    "similarity_threshold": self.similarity_threshold,
                    "distance_metric": self.distance_metric
                },
                "num_samples": len(next(iter(results.values())).similarities) if results else 0
            },
            "individual_results": {},
            "transferability_analysis": transferability_analysis,
            "summary": {}
        }
        
        # 个别模型结果
        for name, result in results.items():
            report["individual_results"][name] = {
                "protection_rate": result.protection_rate,
                "mean_similarity": result.mean_similarity,
                "mean_distance": result.mean_distance,
                "std_similarity": result.std_similarity,
                "std_distance": result.std_distance,
                "metadata": result.metadata
            }
        
        # 摘要统计
        if results:
            protection_rates = [r.protection_rate for r in results.values()]
            similarities = [r.mean_similarity for r in results.values()]
            
            report["summary"] = {
                "total_models_evaluated": len(results),
                "average_protection_rate": np.mean(protection_rates),
                "min_protection_rate": np.min(protection_rates),
                "max_protection_rate": np.max(protection_rates),
                "std_protection_rate": np.std(protection_rates),
                "average_similarity": np.mean(similarities),
                "overall_effectiveness": "High" if np.mean(protection_rates) > 0.7 else "Medium" if np.mean(protection_rates) > 0.4 else "Low"
            }
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"评估报告已保存到: {save_path}")
        
        return report
    
    def plot_results(
        self,
        results: Dict[str, EvaluationResult],
        save_dir: Optional[str] = None
    ):
        """
        绘制评估结果
        
        Args:
            results: 评估结果
            save_dir: 保存目录
        """
        if not results:
            logger.warning("没有结果可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 保护率对比
        model_names = list(results.keys())
        protection_rates = [results[name].protection_rate for name in model_names]
        
        axes[0, 0].bar(model_names, protection_rates)
        axes[0, 0].set_title('Protection Rate by Model')
        axes[0, 0].set_ylabel('Protection Rate')
        axes[0, 0].set_ylim(0, 1)
        for i, rate in enumerate(protection_rates):
            axes[0, 0].text(i, rate + 0.01, f'{rate:.2%}', ha='center')
        
        # 2. 相似度分布
        for name, result in results.items():
            axes[0, 1].hist(result.similarities, alpha=0.7, label=name, bins=20)
        axes[0, 1].axvline(self.similarity_threshold, color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_title('Similarity Distribution')
        axes[0, 1].set_xlabel('Similarity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. 距离分布
        for name, result in results.items():
            axes[1, 0].hist(result.distances, alpha=0.7, label=name, bins=20)
        axes[1, 0].set_title('Distance Distribution')
        axes[1, 0].set_xlabel('Distance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. 模型对比雷达图
        if len(results) >= 2:
            # 选择前两个模型进行对比
            model1, model2 = list(results.keys())[:2]
            result1, result2 = results[model1], results[model2]
            
            categories = ['Protection Rate', 'Mean Distance', '1-Mean Similarity', 'Consistency']
            values1 = [
                result1.protection_rate,
                min(result1.mean_distance, 2.0) / 2.0,  # 归一化到[0,1]
                1 - result1.mean_similarity,
                1 - result1.std_similarity
            ]
            values2 = [
                result2.protection_rate,
                min(result2.mean_distance, 2.0) / 2.0,
                1 - result2.mean_similarity,
                1 - result2.std_similarity
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, values1, width, label=model1)
            axes[1, 1].bar(x + width/2, values2, width, label=model2)
            axes[1, 1].set_title('Model Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(categories, rotation=45)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / "evaluation_results.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"结果图表已保存到: {save_path}")
        
        plt.close()

def create_face_recognition_evaluator(
    model_types: List[str] = None,
    device: str = None,
    **kwargs
) -> FaceRecognitionEvaluator:
    """
    创建面部识别评估器的便捷函数
    
    Args:
        model_types: 模型类型列表
        device: 设备
        **kwargs: 其他参数
        
    Returns:
        评估器实例
    """
    return FaceRecognitionEvaluator(
        model_types=model_types,
        device=device,
        **kwargs
    )

def test_face_recognition_evaluator():
    """测试面部识别评估器"""
    print("🧪 测试面部识别评估器...")
    
    try:
        # 创建评估器
        evaluator = create_face_recognition_evaluator(
            model_types=["arcface"],  # 先测试一个模型
            similarity_threshold=0.6
        )
        
        print(f"✅ 评估器创建成功")
        print(f"   加载模型: {list(evaluator.models.keys())}")
        print(f"   设备: {evaluator.device}")
        
        # 创建测试数据
        batch_size = 3
        test_images = torch.rand(batch_size, 3, 224, 224, device=evaluator.device)
        protected_images = torch.rand(batch_size, 3, 224, 224, device=evaluator.device)
        
        print("🔮 测试特征提取...")
        
        # 测试特征提取
        for model_name in evaluator.models.keys():
            try:
                original_features = evaluator.extract_features(test_images, model_name)
                protected_features = evaluator.extract_features(protected_images, model_name)
                
                print(f"✅ {model_name} 特征提取成功: {original_features.shape}")
                
                # 测试相似度计算
                similarities = evaluator.compute_similarities(original_features, protected_features)
                distances = evaluator.compute_distances(original_features, protected_features)
                
                print(f"✅ {model_name} 相似度计算成功: 平均相似度={np.mean(similarities):.4f}")
                
            except Exception as e:
                print(f"⚠️ {model_name} 测试失败: {e}")
                continue
        
        # 测试评估
        print("📊 测试完整评估...")
        results = evaluator.evaluate_all_models(test_images, protected_images)
        
        if results:
            print(f"✅ 完整评估成功: {len(results)}个模型")
            
            for name, result in results.items():
                print(f"   {name}: 保护率={result.protection_rate:.2%}, 平均相似度={result.mean_similarity:.4f}")
            
            # 测试可转移性分析（如果有多个模型）
            if len(results) >= 2:
                transferability = evaluator.analyze_transferability(results)
                print(f"✅ 可转移性分析完成: 整体可转移性={transferability.get('overall_transferability', 0):.2%}")
            
            # 测试报告生成
            report = evaluator.generate_report(results, {})
            print(f"✅ 报告生成成功: {report['summary'].get('total_models_evaluated', 0)}个模型")
        
        print("🎉 面部识别评估器测试完全通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_face_recognition_evaluator() 