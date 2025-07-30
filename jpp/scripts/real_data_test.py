"""
真实数据端到端测试脚本

使用真实人脸照片进行完整的隐私保护算法测试，包括：
1. 图像预处理和验证
2. 对抗性隐私保护
3. 多模型评估
4. PPR计算
5. 图像质量评估
6. 完整报告生成

这是V0.9版本的关键验证测试。

作者: AI Privacy Protection System
日期: 2025-07-28
"""

import os
import sys
import time
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from datetime import datetime
import json
from tqdm import tqdm

# 启用PyTorch CUDA内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入我们的模块
from src.config.config import ConfigManager, ProtectionStrength
from src.config.protection_strength import get_strength_mapper, ProtectionLevel
from src.utils.image_utils import ImageProcessor
from src.utils.progress_utils import create_progress_manager
from src.models.sd_loader import StableDiffusionLoader
from src.optimization.adversarial_loop import AdversarialOptimizer
from src.evaluation.face_recognition_eval import create_face_recognition_evaluator
from src.evaluation.protection_rate import create_protection_rate_calculator
from src.evaluation.image_quality import create_image_quality_evaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        return f"分配: {allocated:.2f}GB, 保留: {reserved:.2f}GB"
    return "GPU不可用"

class RealDataTester:
    """真实数据测试器"""
    
    def __init__(
        self,
        test_data_dir: str = "data/test_faces/raw",
        output_dir: str = "experiments/real_data_test",
        protection_level: ProtectionLevel = ProtectionLevel.MEDIUM,
        max_images: int = 10,  # 限制处理图像数量以节省时间
        device: str = None
    ):
        """
        初始化真实数据测试器
        
        Args:
            test_data_dir: 测试数据目录
            output_dir: 输出目录
            protection_level: 保护级别
            max_images: 最大处理图像数量
            device: 设备
        """
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.protection_level = protection_level
        self.max_images = max_images
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.protected_dir = self.output_dir / "protected_images"
        self.protected_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self._initialize_components()
        
        logger.info(f"真实数据测试器初始化完成: {max_images}张图像, 保护级别={protection_level.value}")
    
    def _initialize_components(self):
        """初始化各个组件"""
        logger.info("初始化系统组件...")
        logger.info(f"初始GPU内存状态: {get_gpu_memory_info()}")
        
        # 配置管理
        self.config_manager = ConfigManager()
        
        # 保护强度映射
        self.strength_mapper = get_strength_mapper()
        self.protection_config = self.strength_mapper.get_weights(self.protection_level)
        
        # 图像处理
        self.image_processor = ImageProcessor()
        
        # Stable Diffusion加载器（延迟加载）
        self.sd_loader = None
        
        # 评估器（延迟加载，避免重复模型初始化）
        self.face_evaluator = None
        self.ppr_calculator = None
        self.quality_evaluator = None
        
        logger.info("系统组件初始化完成（延迟加载模式）")
    
    def _ensure_sd_loader(self):
        """按需加载Stable Diffusion模型"""
        if self.sd_loader is None:
            try:
                logger.info("按需加载Stable Diffusion模型...")
                clear_gpu_memory()  # 清理内存
                self.sd_loader = StableDiffusionLoader(device=self.device)
                logger.info(f"SD模型加载后GPU内存: {get_gpu_memory_info()}")
            except Exception as e:
                logger.error(f"Stable Diffusion模型加载失败: {e}")
                raise e
    
    def _ensure_evaluators(self):
        """按需加载评估器"""
        if self.face_evaluator is None:
            logger.info("按需加载人脸识别评估器...")
            clear_gpu_memory()
            self.face_evaluator = create_face_recognition_evaluator(
                model_types=["arcface"],
                device=self.device
            )
            logger.info(f"人脸评估器加载后GPU内存: {get_gpu_memory_info()}")
        
        if self.ppr_calculator is None:
            self.ppr_calculator = create_protection_rate_calculator(
                default_threshold=0.6
            )
        
        if self.quality_evaluator is None:
            logger.info("按需加载图像质量评估器...")
            clear_gpu_memory()
            self.quality_evaluator = create_image_quality_evaluator(
                device=self.device
            )
            logger.info(f"质量评估器加载后GPU内存: {get_gpu_memory_info()}")
    
    def load_test_images(self) -> List[Tuple[str, str]]:
        """
        加载测试图像
        
        Returns:
            (图像路径, 图像ID)列表
        """
        logger.info(f"从 {self.test_data_dir} 加载测试图像...")
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.test_data_dir.glob(f'*{ext}'))
            image_files.extend(self.test_data_dir.glob(f'*{ext.upper()}'))
        
        # 排序并限制数量
        image_files = sorted(image_files)[:self.max_images]
        
        # 创建(路径, ID)对
        image_pairs = []
        for img_path in image_files:
            img_id = img_path.stem  # 文件名（不含扩展名）
            image_pairs.append((str(img_path), img_id))
        
        logger.info(f"加载了 {len(image_pairs)} 张测试图像")
        return image_pairs
    
    def preprocess_and_validate_images(
        self, 
        image_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, torch.Tensor]]:
        """
        预处理和验证图像
        
        Args:
            image_pairs: (图像路径, 图像ID)列表
            
        Returns:
            (图像路径, 图像ID, 图像张量)列表
        """
        logger.info("预处理和验证图像...")
        
        valid_images = []
        
        for img_path, img_id in tqdm(image_pairs, desc="预处理图像"):
            try:
                # 加载和验证图像
                img_pil = self.image_processor.load_image(img_path)
                img_tensor = self.image_processor.pil_to_tensor(img_pil)
                
                # 检查图像有效性
                self.image_processor.validate_image(img_pil)
                valid_images.append((img_path, img_id, img_tensor))
                logger.debug(f"图像验证成功: {img_id}")
                    
            except Exception as e:
                logger.warning(f"图像预处理失败 {img_id}: {e}")
                continue
        
        logger.info(f"预处理完成: {len(valid_images)}/{len(image_pairs)} 张图像有效")
        return valid_images
    
    def apply_privacy_protection(
        self, 
        valid_images: List[Tuple[str, str, torch.Tensor]]
    ) -> List[Tuple[str, str, torch.Tensor, torch.Tensor]]:
        """
        应用隐私保护
        
        Args:
            valid_images: 有效图像列表
            
        Returns:
            (原图路径, 图像ID, 原图张量, 保护图张量)列表
        """
        logger.info("开始应用隐私保护...")
        
        # 确保SD模型已加载
        try:
            self._ensure_sd_loader()
        except Exception as e:
            logger.error(f"无法加载Stable Diffusion模型: {e}")
            return []
        
        # 创建对抗优化器
        try:
            # 创建基础配置
            from src.config.config import PrivacyProtectionConfig, load_default_config
            config = load_default_config()
            
            adversarial_optimizer = AdversarialOptimizer(
                sd_loader=self.sd_loader,
                config=config,
                device=self.device
            )
            logger.info("对抗优化器创建成功")
        except Exception as e:
            logger.error(f"对抗优化器创建失败: {e}")
            return []
        
        protected_results = []
        
        # 创建进度管理器
        progress_manager = create_progress_manager(
            total_iterations=len(valid_images) * self.protection_config.max_iterations,
            save_dir=str(self.output_dir / "progress"),
            experiment_name="real_data_protection"
        )
        
        for img_path, img_id, original_tensor in valid_images:
            try:
                logger.info(f"处理图像: {img_id}")
                logger.info(f"处理前GPU内存: {get_gpu_memory_info()}")
                
                # 应用保护
                protected_tensor = self._protect_single_image(
                    original_tensor,
                    adversarial_optimizer,
                    img_id
                )
                
                if protected_tensor is not None:
                    # 保存保护后的图像
                    protected_path = self.protected_dir / f"{img_id}_protected.png"
                    self.image_processor.save_image(protected_tensor, str(protected_path))
                    
                    protected_results.append((img_path, img_id, original_tensor, protected_tensor))
                    logger.info(f"图像保护成功: {img_id}")
                else:
                    logger.warning(f"图像保护失败: {img_id}")
                    
            except Exception as e:
                logger.error(f"图像保护异常 {img_id}: {e}")
                # 清理内存后继续
                clear_gpu_memory()
                continue
            finally:
                # 每个图像处理完成后清理内存
                clear_gpu_memory()
                logger.info(f"处理后GPU内存: {get_gpu_memory_info()}")
        
        logger.info(f"隐私保护完成: {len(protected_results)}/{len(valid_images)} 张图像")
        return protected_results
    
    def _protect_single_image(
        self,
        original_tensor: torch.Tensor,
        adversarial_optimizer: AdversarialOptimizer,
        img_id: str
    ) -> torch.Tensor:
        """
        保护单张图像
        
        Args:
            original_tensor: 原始图像张量
            adversarial_optimizer: 对抗优化器
            img_id: 图像ID
            
        Returns:
            保护后的图像张量
        """
        try:
            # 导入必要的枚举
            from src.config.config import ProtectionStrength
            
            # 映射保护级别到枚举
            level_to_enum = {
                ProtectionLevel.LIGHT: ProtectionStrength.LIGHT,
                ProtectionLevel.MEDIUM: ProtectionStrength.MEDIUM,
                ProtectionLevel.STRONG: ProtectionStrength.STRONG
            }
            
            strength_enum = level_to_enum.get(self.protection_level, ProtectionStrength.MEDIUM)
            
            # 执行优化
            result = adversarial_optimizer.protect_image(
                original_tensor,
                prompt="",
                strength=strength_enum,
                optimize_uncond=False  # 跳过无条件优化，直接进行核心保护
            )
            
            return result.protected_image
            
        except Exception as e:
            logger.error(f"单图像保护失败 {img_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_protection_effectiveness(
        self,
        protected_results: List[Tuple[str, str, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        """
        评估保护效果
        
        Args:
            protected_results: 保护结果列表
            
        Returns:
            评估结果字典
        """
        logger.info("开始评估保护效果...")
        
        evaluation_results = {
            'face_recognition_eval': {},
            'ppr_results': {},
            'quality_results': {},
            'summary': {}
        }
        
        if not protected_results:
            logger.warning("没有保护结果可评估")
            return evaluation_results
            
        # 确保评估器已加载
        try:
            self._ensure_evaluators()
        except Exception as e:
            logger.error(f"评估器加载失败: {e}")
            return evaluation_results
        
        # 准备图像对
        original_images = [result[2] for result in protected_results]  # 原图张量
        protected_images = [result[3] for result in protected_results]  # 保护图张量
        image_ids = [result[1] for result in protected_results]  # 图像ID
        
        try:
            # 1. 人脸识别评估
            logger.info("进行人脸识别评估...")
            logger.info(f"评估前GPU内存: {get_gpu_memory_info()}")
            fr_results = self.face_evaluator.evaluate_all_models(
                original_images, protected_images
            )
            evaluation_results['face_recognition_eval'] = fr_results
            clear_gpu_memory()  # 清理内存
            
            # 2. PPR计算
            logger.info("计算身份保护率...")
            ppr_results = self.ppr_calculator.calculate_multi_model_ppr(fr_results)
            evaluation_results['ppr_results'] = ppr_results
            clear_gpu_memory()  # 清理内存
            
            # 3. 图像质量评估
            logger.info("进行图像质量评估...")
            logger.info(f"质量评估前GPU内存: {get_gpu_memory_info()}")
            image_pairs = list(zip(original_images, protected_images))
            quality_results = self.quality_evaluator.evaluate_batch(image_pairs, image_ids)
            evaluation_results['quality_results'] = quality_results
            clear_gpu_memory()  # 清理内存
            
            # 4. 生成摘要
            evaluation_results['summary'] = self._generate_evaluation_summary(
                fr_results, ppr_results, quality_results
            )
            
            logger.info("保护效果评估完成")
            logger.info(f"评估完成后GPU内存: {get_gpu_memory_info()}")
            
        except Exception as e:
            logger.error(f"评估过程出错: {e}")
            import traceback
            traceback.print_exc()
        
        return evaluation_results
    
    def _generate_evaluation_summary(
        self,
        fr_results: Dict[str, Any],
        ppr_results: Dict[str, Any],
        quality_results: List[Any]
    ) -> Dict[str, Any]:
        """生成评估摘要"""
        summary = {
            'total_images': len(quality_results),
            'protection_effectiveness': {},
            'quality_metrics': {},
            'overall_rating': 'Unknown'
        }
        
        try:
            # PPR摘要
            if ppr_results:
                ppr_values = [result.ppr_value for result in ppr_results.values()]
                summary['protection_effectiveness'] = {
                    'mean_ppr': np.mean(ppr_values),
                    'min_ppr': np.min(ppr_values),
                    'max_ppr': np.max(ppr_values),
                    'models_evaluated': list(ppr_results.keys())
                }
            
            # 质量摘要
            if quality_results:
                lpips_values = [r.metrics.lpips for r in quality_results if r.metrics.lpips is not None]
                ssim_values = [r.metrics.ssim for r in quality_results if r.metrics.ssim is not None]
                
                if lpips_values:
                    summary['quality_metrics']['mean_lpips'] = np.mean(lpips_values)
                if ssim_values:
                    summary['quality_metrics']['mean_ssim'] = np.mean(ssim_values)
            
            # 整体评级
            mean_ppr = summary['protection_effectiveness'].get('mean_ppr', 0)
            mean_lpips = summary['quality_metrics'].get('mean_lpips', 1.0)
            
            if mean_ppr >= 0.7 and mean_lpips <= 0.15:
                summary['overall_rating'] = 'Excellent'
            elif mean_ppr >= 0.5 and mean_lpips <= 0.25:
                summary['overall_rating'] = 'Good'
            elif mean_ppr >= 0.3:
                summary['overall_rating'] = 'Fair'
            else:
                summary['overall_rating'] = 'Poor'
                
        except Exception as e:
            logger.error(f"生成摘要时出错: {e}")
        
        return summary
    
    def generate_comprehensive_report(
        self,
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成综合报告
        
        Args:
            evaluation_results: 评估结果
            
        Returns:
            综合报告
        """
        logger.info("生成综合测试报告...")
        
        report = {
            'metadata': {
                'test_timestamp': datetime.now().isoformat(),
                'protection_level': self.protection_level.value,
                'device': self.device,
                'total_images_processed': evaluation_results.get('summary', {}).get('total_images', 0),
                'test_data_source': str(self.test_data_dir)
            },
            'protection_config': {
                'lambda_id': self.protection_config.lambda_id,
                'lambda_lpips': self.protection_config.lambda_lpips,
                'lambda_self': self.protection_config.lambda_self,
                'max_iterations': self.protection_config.max_iterations,
                'learning_rate': self.protection_config.learning_rate
            },
            'evaluation_results': evaluation_results,
            'recommendations': self._generate_recommendations(evaluation_results)
        }
        
        # 保存报告
        report_path = self.output_dir / "comprehensive_test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"综合报告已保存到: {report_path}")
        return report
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        summary = evaluation_results.get('summary', {})
        
        # 基于PPR的建议
        mean_ppr = summary.get('protection_effectiveness', {}).get('mean_ppr', 0)
        if mean_ppr < 0.5:
            recommendations.append("保护率偏低，建议增加λ_ID权重或增加迭代次数")
        elif mean_ppr > 0.9:
            recommendations.append("保护率很高，可考虑降低保护强度以改善图像质量")
        
        # 基于LPIPS的建议
        mean_lpips = summary.get('quality_metrics', {}).get('mean_lpips', 0)
        if mean_lpips > 0.2:
            recommendations.append("感知质量下降较多，建议增加λ_LPIPS权重")
        
        # 整体建议
        overall_rating = summary.get('overall_rating', 'Unknown')
        if overall_rating == 'Poor':
            recommendations.append("整体效果较差，建议重新调整算法参数或保护策略")
        elif overall_rating == 'Excellent':
            recommendations.append("效果优秀，当前配置可用于生产环境")
        
        return recommendations if recommendations else ["当前配置表现良好，建议保持"]
    
    def run_complete_test(self) -> Dict[str, Any]:
        """
        运行完整测试
        
        Returns:
            完整测试报告
        """
        logger.info("开始真实数据完整测试...")
        start_time = time.time()
        
        try:
            # 1. 加载测试图像
            image_pairs = self.load_test_images()
            if not image_pairs:
                raise RuntimeError("没有找到测试图像")
            
            # 2. 预处理和验证
            valid_images = self.preprocess_and_validate_images(image_pairs)
            if not valid_images:
                raise RuntimeError("没有有效的测试图像")
            
            # 3. 应用隐私保护
            protected_results = self.apply_privacy_protection(valid_images)
            if not protected_results:
                raise RuntimeError("隐私保护处理失败")
            
            # 4. 评估保护效果
            evaluation_results = self.evaluate_protection_effectiveness(protected_results)
            
            # 5. 生成综合报告
            comprehensive_report = self.generate_comprehensive_report(evaluation_results)
            
            # 测试完成
            total_time = time.time() - start_time
            logger.info(f"真实数据测试完成！总耗时: {total_time:.2f}秒")
            
            # 打印关键结果
            self._print_key_results(comprehensive_report)
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"完整测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _print_key_results(self, report: Dict[str, Any]):
        """打印关键结果"""
        print("\n" + "="*80)
        print("真实数据测试关键结果")
        print("="*80)
        
        metadata = report.get('metadata', {})
        summary = report.get('evaluation_results', {}).get('summary', {})
        
        print(f"测试概览:")
        print(f"   处理图像数: {metadata.get('total_images_processed', 0)}")
        print(f"   保护级别: {metadata.get('protection_level', 'Unknown')}")
        print(f"   设备: {metadata.get('device', 'Unknown')}")
        
        protection_effectiveness = summary.get('protection_effectiveness', {})
        if protection_effectiveness:
            print(f"\n保护效果:")
            print(f"   平均PPR: {protection_effectiveness.get('mean_ppr', 0):.2%}")
            print(f"   PPR范围: {protection_effectiveness.get('min_ppr', 0):.2%} - {protection_effectiveness.get('max_ppr', 0):.2%}")
        
        quality_metrics = summary.get('quality_metrics', {})
        if quality_metrics:
            print(f"\n图像质量:")
            if 'mean_lpips' in quality_metrics:
                print(f"   平均LPIPS: {quality_metrics['mean_lpips']:.4f}")
            if 'mean_ssim' in quality_metrics:
                print(f"   平均SSIM: {quality_metrics['mean_ssim']:.4f}")
        
        print(f"\n整体评级: {summary.get('overall_rating', 'Unknown')}")
        
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n改进建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("="*80 + "\n")

def main():
    """主函数"""
    print("真实数据端到端测试")
    print("=" * 50)
    
    # 初始内存状态
    print(f"启动时GPU内存状态: {get_gpu_memory_info()}")
    clear_gpu_memory()  # 清理启动时的内存
    
    # 创建测试器
    tester = RealDataTester(
        test_data_dir="data/test_faces/processed",
        output_dir="experiments/real_data_test",
        protection_level=ProtectionLevel.MEDIUM,
        max_images=1,  # 先测试5张图像
        device="cuda"
    )
    
    print(f"测试器创建后GPU内存状态: {get_gpu_memory_info()}")
    
    # 运行完整测试
    report = tester.run_complete_test()
    
    if "error" not in report:
        print("真实数据测试成功完成！")
        print(f"详细报告已保存到: {tester.output_dir}/comprehensive_test_report.json")
    else:
        print("真实数据测试失败")
        print(f"错误: {report['error']}")
    
    # 最终内存清理
    clear_gpu_memory()
    print(f"测试完成后GPU内存状态: {get_gpu_memory_info()}")

if __name__ == "__main__":
    main() 