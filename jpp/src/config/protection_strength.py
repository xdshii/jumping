"""
保护强度映射系统

该模块提供隐私保护的强度配置映射，支持轻度、中度、强度三个级别，
每个级别对应不同的损失函数权重组合，实现个性化的保护效果。

功能包括：
1. 三档保护强度配置 (轻度/中度/强度)
2. 动态权重映射和验证
3. 用户友好的强度描述
4. 性能和效果的平衡建议

作者: AI Privacy Protection System
日期: 2025-07-28
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class ProtectionLevel(Enum):
    """保护级别枚举"""
    LIGHT = "light"      # 轻度保护
    MEDIUM = "medium"    # 中度保护  
    STRONG = "strong"    # 强度保护

@dataclass
class StrengthWeights:
    """保护强度权重配置"""
    lambda_id: float        # 身份损失权重
    lambda_lpips: float     # LPIPS感知损失权重
    lambda_self: float      # 自注意力损失权重
    max_iterations: int     # 最大迭代次数
    learning_rate: float    # 学习率
    
    def __post_init__(self):
        """验证权重配置的合理性"""
        if self.lambda_id < 0 or self.lambda_lpips < 0 or self.lambda_self < 0:
            raise ValueError("损失权重不能为负数")
        if self.max_iterations <= 0:
            raise ValueError("最大迭代次数必须大于0")
        if self.learning_rate <= 0:
            raise ValueError("学习率必须大于0")

@dataclass  
class ProtectionProfile:
    """保护配置档案"""
    level: ProtectionLevel
    name: str
    description: str
    weights: StrengthWeights
    expected_ppr: float     # 预期身份保护率
    expected_lpips: float   # 预期LPIPS值
    processing_time: str    # 预期处理时间
    use_case: str          # 使用场景

class ProtectionStrengthMapper:
    """保护强度映射器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化保护强度映射器
        
        Args:
            config_path: 配置文件路径，如果None则使用默认配置
        """
        self.config_path = config_path
        self.profiles = self._load_default_profiles()
        
        if config_path and Path(config_path).exists():
            self._load_custom_profiles(config_path)
        
        logger.info(f"保护强度映射器初始化: {len(self.profiles)}个配置档案")
    
    def _load_default_profiles(self) -> Dict[ProtectionLevel, ProtectionProfile]:
        """加载默认保护配置档案"""
        
        profiles = {}
        
        # 轻度保护：注重速度和图像质量，适合日常使用
        profiles[ProtectionLevel.LIGHT] = ProtectionProfile(
            level=ProtectionLevel.LIGHT,
            name="轻度保护",
            description="适合日常使用，在保持图像质量的同时提供基础隐私保护",
            weights=StrengthWeights(
                lambda_id=0.5,      # 较低的身份损失权重
                lambda_lpips=0.8,   # 较高的感知质量权重
                lambda_self=0.3,    # 较低的结构损失权重
                max_iterations=30,  # 较少迭代次数
                learning_rate=0.015 # 稍高学习率加速收敛
            ),
            expected_ppr=60.0,      # 预期60%保护率
            expected_lpips=0.08,    # 预期较好的感知质量
            processing_time="30-60秒",
            use_case="社交媒体、日常分享"
        )
        
        # 中度保护：平衡效果和质量，推荐配置
        profiles[ProtectionLevel.MEDIUM] = ProtectionProfile(
            level=ProtectionLevel.MEDIUM,
            name="中度保护",
            description="推荐配置，在保护效果和图像质量间取得最佳平衡",
            weights=StrengthWeights(
                lambda_id=1.0,      # 标准身份损失权重
                lambda_lpips=0.6,   # 平衡的感知质量权重
                lambda_self=0.4,    # 适中的结构损失权重
                max_iterations=50,  # 标准迭代次数
                learning_rate=0.01  # 标准学习率
            ),
            expected_ppr=75.0,      # 预期75%保护率
            expected_lpips=0.12,    # 预期适中的感知质量
            processing_time="1-2分钟",
            use_case="一般隐私保护、商业用途"
        )
        
        # 强度保护：最大化保护效果，适合高敏感场景
        profiles[ProtectionLevel.STRONG] = ProtectionProfile(
            level=ProtectionLevel.STRONG,
            name="强度保护",
            description="最大化隐私保护效果，适用于高敏感和安全要求的场景",
            weights=StrengthWeights(
                lambda_id=1.5,      # 高身份损失权重
                lambda_lpips=0.4,   # 较低的感知质量权重
                lambda_self=0.6,    # 较高的结构损失权重
                max_iterations=80,  # 更多迭代次数
                learning_rate=0.008 # 较低学习率精细优化
            ),
            expected_ppr=85.0,      # 预期85%保护率
            expected_lpips=0.18,    # 预期感知质量下降
            processing_time="2-4分钟",
            use_case="高敏感内容、安全关键应用"
        )
        
        return profiles
    
    def _load_custom_profiles(self, config_path: str):
        """从配置文件加载自定义档案"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            custom_profiles = config.get('protection_profiles', {})
            
            for level_str, profile_config in custom_profiles.items():
                try:
                    level = ProtectionLevel(level_str)
                    
                    # 更新现有档案或创建新档案
                    if level in self.profiles:
                        # 更新权重
                        weights_config = profile_config.get('weights', {})
                        current_weights = self.profiles[level].weights
                        
                        updated_weights = StrengthWeights(
                            lambda_id=weights_config.get('lambda_id', current_weights.lambda_id),
                            lambda_lpips=weights_config.get('lambda_lpips', current_weights.lambda_lpips),
                            lambda_self=weights_config.get('lambda_self', current_weights.lambda_self),
                            max_iterations=weights_config.get('max_iterations', current_weights.max_iterations),
                            learning_rate=weights_config.get('learning_rate', current_weights.learning_rate)
                        )
                        
                        self.profiles[level].weights = updated_weights
                        logger.info(f"更新保护档案: {level.value}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"跳过无效的保护档案配置 {level_str}: {e}")
                    
        except Exception as e:
            logger.error(f"加载自定义保护档案失败: {e}")
    
    def get_profile(self, level: ProtectionLevel) -> ProtectionProfile:
        """
        获取指定级别的保护档案
        
        Args:
            level: 保护级别
            
        Returns:
            保护档案
        """
        if level not in self.profiles:
            raise ValueError(f"不支持的保护级别: {level}")
        
        return self.profiles[level]
    
    def get_weights(self, level: ProtectionLevel) -> StrengthWeights:
        """
        获取指定级别的权重配置
        
        Args:
            level: 保护级别
            
        Returns:
            权重配置
        """
        return self.get_profile(level).weights
    
    def get_weights_dict(self, level: ProtectionLevel) -> Dict[str, float]:
        """
        获取权重字典格式（用于兼容现有代码）
        
        Args:
            level: 保护级别
            
        Returns:
            权重字典
        """
        weights = self.get_weights(level)
        return {
            'lambda_id': weights.lambda_id,
            'lambda_lpips': weights.lambda_lpips,
            'lambda_self': weights.lambda_self
        }
    
    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有保护档案的摘要信息
        
        Returns:
            档案摘要字典
        """
        summary = {}
        
        for level, profile in self.profiles.items():
            summary[level.value] = {
                'name': profile.name,
                'description': profile.description,
                'expected_ppr': profile.expected_ppr,
                'expected_lpips': profile.expected_lpips,
                'processing_time': profile.processing_time,
                'use_case': profile.use_case,
                'weights': {
                    'lambda_id': profile.weights.lambda_id,
                    'lambda_lpips': profile.weights.lambda_lpips,
                    'lambda_self': profile.weights.lambda_self,
                    'max_iterations': profile.weights.max_iterations,
                    'learning_rate': profile.weights.learning_rate
                }
            }
        
        return summary
    
    def recommend_profile(
        self,
        priority: str = "balanced",
        time_budget: Optional[str] = None,
        quality_requirement: str = "medium"
    ) -> Tuple[ProtectionLevel, str]:
        """
        基于用户需求推荐保护档案
        
        Args:
            priority: 优先级 ("speed", "protection", "quality", "balanced")
            time_budget: 时间预算 ("fast", "medium", "slow")
            quality_requirement: 质量要求 ("low", "medium", "high")
            
        Returns:
            (推荐级别, 推荐理由)
        """
        
        if priority == "speed" or time_budget == "fast":
            return ProtectionLevel.LIGHT, "优先考虑处理速度，选择轻度保护"
        
        elif priority == "protection":
            return ProtectionLevel.STRONG, "优先考虑保护效果，选择强度保护"
        
        elif priority == "quality" and quality_requirement == "high":
            return ProtectionLevel.LIGHT, "优先考虑图像质量，选择轻度保护"
        
        elif time_budget == "slow" and quality_requirement == "low":
            return ProtectionLevel.STRONG, "有充足时间且质量要求不高，选择强度保护"
        
        else:
            return ProtectionLevel.MEDIUM, "平衡各方面需求，选择中度保护（推荐）"
    
    def save_profiles(self, output_path: str):
        """
        保存当前档案配置到文件
        
        Args:
            output_path: 输出文件路径
        """
        config = {
            'protection_profiles': {}
        }
        
        for level, profile in self.profiles.items():
            config['protection_profiles'][level.value] = {
                'name': profile.name,
                'description': profile.description,
                'weights': {
                    'lambda_id': profile.weights.lambda_id,
                    'lambda_lpips': profile.weights.lambda_lpips,
                    'lambda_self': profile.weights.lambda_self,
                    'max_iterations': profile.weights.max_iterations,
                    'learning_rate': profile.weights.learning_rate
                },
                'expected_ppr': profile.expected_ppr,
                'expected_lpips': profile.expected_lpips,
                'processing_time': profile.processing_time,
                'use_case': profile.use_case
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"保护档案配置已保存到: {output_path}")

# 全局实例
_global_mapper = None

def get_strength_mapper(config_path: Optional[str] = None) -> ProtectionStrengthMapper:
    """
    获取全局保护强度映射器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        映射器实例
    """
    global _global_mapper
    
    if _global_mapper is None:
        _global_mapper = ProtectionStrengthMapper(config_path)
    
    return _global_mapper

def create_strength_mapper(config_path: Optional[str] = None) -> ProtectionStrengthMapper:
    """
    创建新的保护强度映射器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        映射器实例
    """
    return ProtectionStrengthMapper(config_path)

def test_protection_strength():
    """测试保护强度映射系统"""
    print("🧪 测试保护强度映射系统...")
    
    try:
        # 创建映射器
        mapper = create_strength_mapper()
        
        print("✅ 保护强度映射器创建成功")
        
        # 测试档案列表
        profiles = mapper.list_profiles()
        print(f"✅ 加载了 {len(profiles)} 个保护档案:")
        
        for level, info in profiles.items():
            print(f"   📋 {level}: {info['name']}")
            print(f"      描述: {info['description']}")
            print(f"      权重: λ_ID={info['weights']['lambda_id']}, λ_LPIPS={info['weights']['lambda_lpips']}, λ_self={info['weights']['lambda_self']}")
            print(f"      预期PPR: {info['expected_ppr']}%, LPIPS: {info['expected_lpips']}")
            print(f"      处理时间: {info['processing_time']}")
            print()
        
        # 测试权重获取
        print("🔮 测试权重获取...")
        for level in ProtectionLevel:
            weights = mapper.get_weights(level)
            weights_dict = mapper.get_weights_dict(level)
            print(f"✅ {level.value}: 权重对象={weights}, 字典={weights_dict}")
        
        # 测试推荐系统
        print("🎯 测试推荐系统...")
        test_cases = [
            {"priority": "speed", "desc": "速度优先"},
            {"priority": "protection", "desc": "保护优先"},
            {"priority": "quality", "quality_requirement": "high", "desc": "质量优先"},
            {"priority": "balanced", "desc": "平衡需求"}
        ]
        
        for case in test_cases:
            desc = case.pop("desc")
            level, reason = mapper.recommend_profile(**case)
            print(f"✅ {desc}: 推荐 {level.value} - {reason}")
        
        # 测试配置保存
        print("💾 测试配置保存...")
        test_config_path = "test_protection_profiles.yaml"
        mapper.save_profiles(test_config_path)
        print(f"✅ 配置已保存到: {test_config_path}")
        
        # 清理测试文件
        import os
        if os.path.exists(test_config_path):
            os.remove(test_config_path)
            print("✅ 测试文件已清理")
        
        print("🎉 保护强度映射系统测试完全通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_protection_strength() 