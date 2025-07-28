"""
AI图像隐私保护系统 - 配置管理模块

这个模块提供了完整的配置管理功能，包括：
- YAML配置文件的加载和保存
- 配置参数的验证和类型检查
- 默认配置和用户配置的合并
- 运行时配置的动态更新

作者: AI Privacy Protection Team
创建时间: 2025-01-28
版本: 1.0.0
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import torch

logger = logging.getLogger(__name__)


class ProtectionStrength(Enum):
    """保护强度枚举"""
    LIGHT = "light"      # 轻度保护
    MEDIUM = "medium"    # 中度保护  
    STRONG = "strong"    # 强度保护


class ModelType(Enum):
    """模型类型枚举"""
    STABLE_DIFFUSION = "stable_diffusion"
    FLUX = "flux"


@dataclass
class OptimizationConfig:
    """优化配置参数"""
    
    # 基础优化参数
    learning_rate: float = 0.01
    max_iterations: int = 250
    convergence_threshold: float = 1e-4
    
    # 损失函数权重 - 默认为中度保护
    identity_loss_weight: float = 1.0
    lpips_loss_weight: float = 0.6
    cross_attn_loss_weight: float = 10000.0
    self_attn_loss_weight: float = 100.0
    
    # 强度映射配置
    strength_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "light": {
            "identity": 0.5,
            "lpips": 1.0,
            "cross_attn": 5000.0,
            "self_attn": 50.0,
            "max_iterations": 100
        },
        "medium": {
            "identity": 1.0,
            "lpips": 0.6,
            "cross_attn": 10000.0,
            "self_attn": 100.0,
            "max_iterations": 250
        },
        "strong": {
            "identity": 1.5,
            "lpips": 0.4,
            "cross_attn": 15000.0,
            "self_attn": 150.0,
            "max_iterations": 350
        }
    })
    
    def __post_init__(self):
        """配置验证"""
        if self.learning_rate <= 0:
            raise ValueError("学习率必须大于0")
        if self.max_iterations <= 0:
            raise ValueError("最大迭代次数必须大于0")


@dataclass
class ModelConfig:
    """模型配置参数"""
    
    # 模型路径配置
    stable_diffusion_path: str = "checkpoints/sd2"
    flux_path: str = "checkpoints/flux1"
    face_models_path: str = "checkpoints/face_models"
    
    # Stable Diffusion参数
    sd_model_id: str = "stabilityai/stable-diffusion-2-base"
    sd_guidance_scale: float = 7.5
    sd_num_inference_steps: int = 50
    
    # FLUX.1参数
    flux_model_id: str = "black-forest-labs/FLUX.1-dev"
    flux_guidance_scale: float = 1.0
    flux_num_inference_steps: int = 28
    
    # 面部识别模型配置
    arcface_model: str = "buffalo_l"
    facenet_model: str = "vggface2"
    
    def validate_paths(self) -> None:
        """验证模型路径"""
        paths = [
            self.stable_diffusion_path, 
            self.flux_path, 
            self.face_models_path
        ]
        for path in paths:
            if not os.path.exists(path):
                logger.warning(f"模型路径不存在: {path}")


@dataclass
class SystemConfig:
    """系统配置参数"""
    
    # 设备配置
    device: str = "cuda:0"
    use_fp16: bool = True
    use_xformers: bool = True
    
    # 内存配置
    max_batch_size: int = 1
    gradient_checkpointing: bool = True
    low_mem: bool = False
    
    # 图像参数
    image_size: int = 512
    output_format: str = "PNG"
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # 实验配置
    experiment_name: str = "privacy_protection"
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        """系统配置验证"""
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA不可用，自动切换到CPU")
            self.device = "cpu"
            self.use_fp16 = False


@dataclass
class PrivacyProtectionConfig:
    """隐私保护系统完整配置"""
    
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig) 
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def get_strength_config(self, strength: ProtectionStrength) -> Dict[str, Any]:
        """根据保护强度获取配置参数"""
        strength_key = strength.value
        if strength_key not in self.optimization.strength_weights:
            raise ValueError(f"不支持的保护强度: {strength_key}")
        
        return self.optimization.strength_weights[strength_key]
    
    def update_for_strength(self, strength: ProtectionStrength) -> None:
        """更新配置以适应指定的保护强度"""
        strength_config = self.get_strength_config(strength)
        
        self.optimization.identity_loss_weight = strength_config["identity"]
        self.optimization.lpips_loss_weight = strength_config["lpips"]
        self.optimization.cross_attn_loss_weight = strength_config["cross_attn"]
        self.optimization.self_attn_loss_weight = strength_config["self_attn"]
        self.optimization.max_iterations = strength_config["max_iterations"]


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_config(
        self, 
        config_file: str = "default.yaml",
        config_class: type = PrivacyProtectionConfig
    ) -> PrivacyProtectionConfig:
        """
        从YAML文件加载配置
        
        Args:
            config_file: 配置文件名
            config_class: 配置类
            
        Returns:
            配置实例
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            logger.info(f"配置文件不存在，创建默认配置: {config_path}")
            default_config = config_class()
            self.save_config(default_config, config_file)
            return default_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 递归创建配置对象
            return self._dict_to_config(config_data, config_class)
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.info("使用默认配置")
            return config_class()
    
    def save_config(
        self, 
        config: PrivacyProtectionConfig, 
        config_file: str = "default.yaml"
    ) -> None:
        """
        保存配置到YAML文件
        
        Args:
            config: 配置实例
            config_file: 配置文件名
        """
        config_path = self.config_dir / config_file
        
        try:
            config_dict = self._config_to_dict(config)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict, 
                    f, 
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )
            
            logger.info(f"配置已保存到: {config_path}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any], config_class: type):
        """将字典转换为配置对象"""
        if not isinstance(config_dict, dict):
            return config_dict
        
        # 处理嵌套的配置类
        if config_class == PrivacyProtectionConfig:
            return PrivacyProtectionConfig(
                optimization=self._dict_to_config(
                    config_dict.get('optimization', {}), 
                    OptimizationConfig
                ),
                model=self._dict_to_config(
                    config_dict.get('model', {}), 
                    ModelConfig
                ),
                system=self._dict_to_config(
                    config_dict.get('system', {}), 
                    SystemConfig
                )
            )
        else:
            # 使用字典中的值更新默认配置
            default_config = config_class()
            for key, value in config_dict.items():
                if hasattr(default_config, key):
                    setattr(default_config, key, value)
            return default_config
    
    def _config_to_dict(self, config) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        if hasattr(config, '__dataclass_fields__'):
            result = {}
            for field_name, field_value in config.__dict__.items():
                if hasattr(field_value, '__dataclass_fields__'):
                    result[field_name] = self._config_to_dict(field_value)
                else:
                    result[field_name] = field_value
            return result
        else:
            return config
    
    def create_experiment_config(
        self, 
        base_config: PrivacyProtectionConfig,
        experiment_name: str,
        **overrides
    ) -> PrivacyProtectionConfig:
        """
        创建实验配置
        
        Args:
            base_config: 基础配置
            experiment_name: 实验名称
            **overrides: 需要覆盖的配置参数
            
        Returns:
            实验配置
        """
        # 深拷贝基础配置
        import copy
        exp_config = copy.deepcopy(base_config)
        
        # 更新实验名称
        exp_config.system.experiment_name = experiment_name
        
        # 应用覆盖参数
        for key, value in overrides.items():
            if '.' in key:
                # 支持嵌套属性设置，如 "optimization.learning_rate"
                keys = key.split('.')
                obj = exp_config
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                setattr(obj, keys[-1], value)
            else:
                setattr(exp_config, key, value)
        
        return exp_config


# 全局配置管理器实例
config_manager = ConfigManager()


def load_default_config() -> PrivacyProtectionConfig:
    """加载默认配置"""
    return config_manager.load_config()


def save_default_config(config: PrivacyProtectionConfig) -> None:
    """保存默认配置"""
    config_manager.save_config(config)


if __name__ == "__main__":
    # 测试配置系统
    import torch
    
    # 创建并保存默认配置
    config = PrivacyProtectionConfig()
    config_manager.save_config(config, "test_config.yaml")
    
    # 加载配置
    loaded_config = config_manager.load_config("test_config.yaml")
    
    # 测试强度配置
    loaded_config.update_for_strength(ProtectionStrength.STRONG)
    print(f"强度保护配置 - Identity权重: {loaded_config.optimization.identity_loss_weight}")
    
    # 创建实验配置
    exp_config = config_manager.create_experiment_config(
        loaded_config,
        "test_experiment",
        **{"optimization.learning_rate": 0.005}
    )
    
    print("配置系统测试完成！") 