"""
身份保护率(PPR)计算模块

该模块实现隐私保护率(Privacy Protection Rate)的计算，
这是评估面部隐私保护算法效果的核心指标。

功能包括：
1. 标准PPR计算 (基于相似度阈值)
2. 多阈值PPR分析
3. 模型间PPR对比
4. 时间序列PPR追踪
5. PPR可视化和报告生成

基于DiffPrivate论文中的评估指标设计。

作者: AI Privacy Protection System
日期: 2025-07-28
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

try:
    from ..evaluation.face_recognition_eval import FaceRecognitionEvaluator, EvaluationResult
    from ..utils.image_utils import ImageProcessor
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from evaluation.face_recognition_eval import FaceRecognitionEvaluator, EvaluationResult
    from utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)

@dataclass
class PPRResult:
    """PPR计算结果数据类"""
    model_name: str
    threshold: float
    ppr_value: float
    num_protected: int
    total_samples: int
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_metrics: Optional[Dict[str, Any]] = None

class ProtectionRateCalculator:
    """身份保护率计算器"""
    
    def __init__(
        self,
        default_threshold: float = 0.6,
        confidence_level: float = 0.95
    ):
        """
        初始化保护率计算器
        
        Args:
            default_threshold: 默认相似度阈值
            confidence_level: 置信区间水平
        """
        self.default_threshold = default_threshold
        self.confidence_level = confidence_level
        
        # 初始化工具
        self.image_processor = ImageProcessor()
        
        logger.info(f"保护率计算器初始化: 阈值={default_threshold}, 置信度={confidence_level}")
    
    def calculate_ppr(
        self,
        similarities: np.ndarray,
        threshold: float = None,
        compute_confidence: bool = True
    ) -> PPRResult:
        """
        计算单个模型的PPR
        
        Args:
            similarities: 相似度数组
            threshold: 相似度阈值
            compute_confidence: 是否计算置信区间
            
        Returns:
            PPR结果
        """
        if threshold is None:
            threshold = self.default_threshold
        
        # 过滤NaN值
        valid_similarities = similarities[~np.isnan(similarities)]
        if len(valid_similarities) == 0:
            logger.warning("所有相似度值都是NaN")
            return PPRResult(
                model_name="unknown",
                threshold=threshold,
                ppr_value=0.0,
                num_protected=0,
                total_samples=len(similarities)
            )
        
        # 计算保护的样本数（相似度低于阈值）
        protected_mask = valid_similarities < threshold
        num_protected = np.sum(protected_mask)
        total_samples = len(valid_similarities)
        
        # 计算PPR
        ppr_value = num_protected / total_samples if total_samples > 0 else 0.0
        
        # 计算置信区间
        confidence_interval = None
        if compute_confidence and total_samples > 0:
            confidence_interval = self._compute_confidence_interval(
                ppr_value, total_samples, self.confidence_level
            )
        
        # 计算附加指标
        additional_metrics = {
            "mean_similarity": np.mean(valid_similarities),
            "std_similarity": np.std(valid_similarities),
            "median_similarity": np.median(valid_similarities),
            "min_similarity": np.min(valid_similarities),
            "max_similarity": np.max(valid_similarities),
            "num_nan": len(similarities) - len(valid_similarities)
        }
        
        return PPRResult(
            model_name="unknown",
            threshold=threshold,
            ppr_value=ppr_value,
            num_protected=num_protected,
            total_samples=total_samples,
            confidence_interval=confidence_interval,
            additional_metrics=additional_metrics
        )
    
    def _compute_confidence_interval(
        self,
        ppr: float,
        n: int,
        confidence_level: float
    ) -> Tuple[float, float]:
        """
        计算PPR的置信区间（使用二项分布的正态近似）
        
        Args:
            ppr: PPR值
            n: 样本数量
            confidence_level: 置信水平
            
        Returns:
            (下界, 上界)
        """
        if n <= 0:
            return (0.0, 0.0)
        
        # 计算标准误差
        se = np.sqrt(ppr * (1 - ppr) / n)
        
        # Z分数（对于95%置信区间，z=1.96）
        if confidence_level == 0.95:
            z = 1.96
        elif confidence_level == 0.99:
            z = 2.576
        elif confidence_level == 0.90:
            z = 1.645
        else:
            # 近似计算
            from scipy.stats import norm
            z = norm.ppf((1 + confidence_level) / 2)
        
        # 计算置信区间
        margin = z * se
        lower = max(0.0, ppr - margin)
        upper = min(1.0, ppr + margin)
        
        return (lower, upper)
    
    def calculate_multi_threshold_ppr(
        self,
        similarities: np.ndarray,
        thresholds: List[float] = None,
        model_name: str = "unknown"
    ) -> List[PPRResult]:
        """
        计算多个阈值下的PPR
        
        Args:
            similarities: 相似度数组
            thresholds: 阈值列表
            model_name: 模型名称
            
        Returns:
            PPR结果列表
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = []
        for threshold in thresholds:
            result = self.calculate_ppr(similarities, threshold)
            result.model_name = model_name
            results.append(result)
        
        return results
    
    def calculate_multi_model_ppr(
        self,
        evaluation_results: Dict[str, EvaluationResult],
        threshold: float = None
    ) -> Dict[str, PPRResult]:
        """
        计算多个模型的PPR
        
        Args:
            evaluation_results: 评估结果字典
            threshold: 相似度阈值
            
        Returns:
            各模型的PPR结果
        """
        if threshold is None:
            threshold = self.default_threshold
        
        ppr_results = {}
        
        for model_name, eval_result in evaluation_results.items():
            ppr_result = self.calculate_ppr(eval_result.similarities, threshold)
            ppr_result.model_name = model_name
            ppr_results[model_name] = ppr_result
        
        return ppr_results
    
    def analyze_ppr_trends(
        self,
        ppr_history: List[Dict[str, PPRResult]],
        time_points: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        分析PPR趋势
        
        Args:
            ppr_history: PPR历史记录列表
            time_points: 时间点列表
            
        Returns:
            趋势分析结果
        """
        if not ppr_history:
            return {}
        
        # 提取所有模型名称
        all_models = set()
        for ppr_dict in ppr_history:
            all_models.update(ppr_dict.keys())
        
        trends = {}
        
        for model_name in all_models:
            model_pprs = []
            model_times = []
            
            for i, ppr_dict in enumerate(ppr_history):
                if model_name in ppr_dict:
                    model_pprs.append(ppr_dict[model_name].ppr_value)
                    model_times.append(time_points[i] if time_points else i)
            
            if len(model_pprs) >= 2:
                # 计算趋势
                ppr_array = np.array(model_pprs)
                
                # 线性回归斜率（简单趋势）
                x = np.arange(len(ppr_array))
                slope = np.polyfit(x, ppr_array, 1)[0] if len(ppr_array) > 1 else 0
                
                trends[model_name] = {
                    "values": model_pprs,
                    "times": model_times,
                    "initial_ppr": model_pprs[0],
                    "final_ppr": model_pprs[-1],
                    "change": model_pprs[-1] - model_pprs[0],
                    "slope": slope,
                    "trend": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
                    "max_ppr": np.max(ppr_array),
                    "min_ppr": np.min(ppr_array),
                    "mean_ppr": np.mean(ppr_array),
                    "std_ppr": np.std(ppr_array)
                }
        
        return trends
    
    def compare_models(
        self,
        ppr_results: Dict[str, PPRResult]
    ) -> Dict[str, Any]:
        """
        比较不同模型的PPR
        
        Args:
            ppr_results: PPR结果字典
            
        Returns:
            模型比较结果
        """
        if not ppr_results:
            return {}
        
        # 提取PPR值
        model_names = list(ppr_results.keys())
        ppr_values = [ppr_results[name].ppr_value for name in model_names]
        
        # 排序
        sorted_indices = np.argsort(ppr_values)[::-1]  # 降序
        
        comparison = {
            "best_model": model_names[sorted_indices[0]] if model_names else "",
            "worst_model": model_names[sorted_indices[-1]] if model_names else "",
            "rankings": [model_names[i] for i in sorted_indices],
            "ppr_values": [ppr_values[i] for i in sorted_indices],
            "mean_ppr": np.mean(ppr_values),
            "std_ppr": np.std(ppr_values),
            "range_ppr": np.max(ppr_values) - np.min(ppr_values) if ppr_values else 0,
            "consistency": 1.0 / (1.0 + np.std(ppr_values)) if ppr_values else 0  # 一致性分数
        }
        
        # 统计分析
        if len(ppr_values) >= 2:
            comparison["statistical_significance"] = self._test_significance(ppr_results)
        
        return comparison
    
    def _test_significance(
        self,
        ppr_results: Dict[str, PPRResult]
    ) -> Dict[str, Any]:
        """
        测试PPR差异的统计显著性
        
        Args:
            ppr_results: PPR结果字典
            
        Returns:
            显著性测试结果
        """
        # 简化的卡方检验
        models = list(ppr_results.keys())
        significance_tests = {}
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                result1 = ppr_results[model1]
                result2 = ppr_results[model2]
                
                # 构建2x2列联表
                protected1 = result1.num_protected
                unprotected1 = result1.total_samples - result1.num_protected
                protected2 = result2.num_protected
                unprotected2 = result2.total_samples - result2.num_protected
                
                # 简单的差异检验
                if result1.total_samples > 0 and result2.total_samples > 0:
                    diff = abs(result1.ppr_value - result2.ppr_value)
                    significance_tests[f"{model1}_vs_{model2}"] = {
                        "difference": result1.ppr_value - result2.ppr_value,
                        "abs_difference": diff,
                        "is_significant": diff > 0.05,  # 简化的显著性判断
                        "better_model": model1 if result1.ppr_value > result2.ppr_value else model2
                    }
        
        return significance_tests
    
    def generate_ppr_report(
        self,
        ppr_results: Dict[str, PPRResult],
        comparison: Dict[str, Any] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成PPR评估报告
        
        Args:
            ppr_results: PPR结果字典
            comparison: 模型比较结果
            save_path: 保存路径
            
        Returns:
            完整报告
        """
        if comparison is None:
            comparison = self.compare_models(ppr_results)
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "calculator_config": {
                    "default_threshold": self.default_threshold,
                    "confidence_level": self.confidence_level
                },
                "total_models": len(ppr_results)
            },
            "individual_results": {},
            "comparison": comparison,
            "summary": {}
        }
        
        # 个别模型结果
        for model_name, ppr_result in ppr_results.items():
            report["individual_results"][model_name] = {
                "ppr_value": ppr_result.ppr_value,
                "threshold": ppr_result.threshold,
                "num_protected": ppr_result.num_protected,
                "total_samples": ppr_result.total_samples,
                "confidence_interval": ppr_result.confidence_interval,
                "additional_metrics": ppr_result.additional_metrics
            }
        
        # 摘要统计
        if ppr_results:
            ppr_values = [r.ppr_value for r in ppr_results.values()]
            report["summary"] = {
                "overall_ppr": np.mean(ppr_values),
                "best_ppr": np.max(ppr_values),
                "worst_ppr": np.min(ppr_values),
                "ppr_std": np.std(ppr_values),
                "effectiveness_rating": self._rate_effectiveness(np.mean(ppr_values)),
                "recommendation": self._generate_recommendation(comparison)
            }
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"PPR报告已保存到: {save_path}")
        
        return report
    
    def _rate_effectiveness(self, ppr: float) -> str:
        """评估保护效果等级"""
        if ppr >= 0.85:
            return "Excellent"
        elif ppr >= 0.70:
            return "Good"
        elif ppr >= 0.50:
            return "Fair"
        elif ppr >= 0.30:
            return "Poor"
        else:
            return "Very Poor"
    
    def _generate_recommendation(self, comparison: Dict[str, Any]) -> str:
        """生成改进建议"""
        if not comparison:
            return "无足够数据生成建议"
        
        mean_ppr = comparison.get("mean_ppr", 0)
        consistency = comparison.get("consistency", 0)
        
        if mean_ppr >= 0.8 and consistency >= 0.8:
            return "保护效果优秀且一致，建议保持当前配置"
        elif mean_ppr >= 0.7:
            return "保护效果良好，可考虑优化一致性"
        elif mean_ppr >= 0.5:
            return "保护效果中等，建议增强保护强度或优化算法参数"
        else:
            return "保护效果较差，建议重新评估算法设计或参数配置"
    
    def plot_ppr_comparison(
        self,
        ppr_results: Dict[str, PPRResult],
        save_path: Optional[str] = None,
        show_confidence: bool = True
    ):
        """
        绘制PPR对比图
        
        Args:
            ppr_results: PPR结果字典
            save_path: 保存路径
            show_confidence: 是否显示置信区间
        """
        if not ppr_results:
            logger.warning("没有PPR结果可绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = list(ppr_results.keys())
        ppr_values = [ppr_results[name].ppr_value for name in model_names]
        
        # 1. PPR柱状图
        bars = ax1.bar(model_names, ppr_values)
        ax1.set_title('Privacy Protection Rate by Model')
        ax1.set_ylabel('PPR')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for i, (bar, ppr) in enumerate(zip(bars, ppr_values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ppr:.2%}', ha='center', va='bottom')
            
            # 添加置信区间
            if show_confidence:
                ppr_result = ppr_results[model_names[i]]
                if ppr_result.confidence_interval:
                    lower, upper = ppr_result.confidence_interval
                    ax1.errorbar(bar.get_x() + bar.get_width()/2, ppr,
                               yerr=[[ppr - lower], [upper - ppr]],
                               fmt='none', capsize=3, color='black')
        
        # 2. 阈值敏感性分析（如果有多阈值数据）
        # 这里显示样本数量分布
        sample_counts = [ppr_results[name].total_samples for name in model_names]
        
        ax2.bar(model_names, sample_counts, alpha=0.7, color='lightblue')
        ax2.set_title('Sample Count by Model')
        ax2.set_ylabel('Number of Samples')
        
        for i, count in enumerate(sample_counts):
            ax2.text(i, count + max(sample_counts) * 0.01, str(count),
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PPR对比图已保存到: {save_path}")
        
        plt.close()

def create_protection_rate_calculator(
    default_threshold: float = 0.6,
    **kwargs
) -> ProtectionRateCalculator:
    """
    创建保护率计算器的便捷函数
    
    Args:
        default_threshold: 默认阈值
        **kwargs: 其他参数
        
    Returns:
        保护率计算器实例
    """
    return ProtectionRateCalculator(
        default_threshold=default_threshold,
        **kwargs
    )

def test_protection_rate_calculator():
    """测试保护率计算器"""
    print("🧪 测试身份保护率计算器...")
    
    try:
        # 创建计算器
        calculator = create_protection_rate_calculator(default_threshold=0.6)
        
        print("✅ 保护率计算器创建成功")
        print(f"   默认阈值: {calculator.default_threshold}")
        print(f"   置信水平: {calculator.confidence_level}")
        
        # 创建测试数据
        np.random.seed(42)  # 固定随机种子以获得一致结果
        
        # 模拟相似度数据（保护后应该更低）
        original_similarities = np.random.uniform(0.7, 0.95, 100)  # 高相似度
        protected_similarities = np.random.uniform(0.2, 0.6, 100)   # 低相似度
        
        print("🔮 测试单个PPR计算...")
        
        # 测试原始图像PPR（应该很低）
        original_ppr = calculator.calculate_ppr(original_similarities)
        print(f"✅ 原始图像PPR: {original_ppr.ppr_value:.2%} ({original_ppr.num_protected}/{original_ppr.total_samples})")
        
        # 测试保护图像PPR（应该较高）
        protected_ppr = calculator.calculate_ppr(protected_similarities)
        print(f"✅ 保护图像PPR: {protected_ppr.ppr_value:.2%} ({protected_ppr.num_protected}/{protected_ppr.total_samples})")
        
        # 测试多阈值PPR
        print("📊 测试多阈值PPR...")
        multi_threshold_results = calculator.calculate_multi_threshold_ppr(
            protected_similarities, 
            thresholds=[0.3, 0.5, 0.7, 0.9],
            model_name="test_model"
        )
        
        for result in multi_threshold_results:
            print(f"   阈值{result.threshold}: PPR={result.ppr_value:.2%}")
        
        # 测试模型比较
        print("🏆 测试模型比较...")
        
        # 模拟多个模型的PPR结果
        model_pprs = {
            "arcface": PPRResult("arcface", 0.6, 0.85, 85, 100),
            "facenet": PPRResult("facenet", 0.6, 0.78, 78, 100),
            "dummy": PPRResult("dummy", 0.6, 0.65, 65, 100)
        }
        
        comparison = calculator.compare_models(model_pprs)
        print(f"✅ 最佳模型: {comparison['best_model']} (PPR: {max(comparison['ppr_values']):.2%})")
        print(f"✅ 最差模型: {comparison['worst_model']} (PPR: {min(comparison['ppr_values']):.2%})")
        print(f"✅ 平均PPR: {comparison['mean_ppr']:.2%}")
        print(f"✅ 一致性分数: {comparison['consistency']:.3f}")
        
        # 测试报告生成
        print("📋 测试报告生成...")
        report = calculator.generate_ppr_report(model_pprs, comparison)
        
        print(f"✅ 报告生成成功:")
        print(f"   评估模型数: {report['metadata']['total_models']}")
        print(f"   整体PPR: {report['summary']['overall_ppr']:.2%}")
        print(f"   效果评级: {report['summary']['effectiveness_rating']}")
        print(f"   建议: {report['summary']['recommendation']}")
        
        print("🎉 身份保护率计算器测试全部通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_protection_rate_calculator() 