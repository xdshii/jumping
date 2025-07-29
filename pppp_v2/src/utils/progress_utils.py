"""
进度条显示和中间结果保存工具

该模块提供优化过程中的进度可视化和中间结果保存功能，
帮助用户跟踪优化进度并保存关键的中间状态。

功能包括：
1. 自定义进度条显示（支持多种信息显示）
2. 中间结果保存（图像、损失历史、模型状态）
3. 优化过程可视化（实时损失曲线）
4. 结果对比和分析工具
5. 异常情况的恢复机制

作者: AI Privacy Protection System
日期: 2025-07-28
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import pickle
import json
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import threading

try:
    from ..utils.image_utils import ImageProcessor
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)

@dataclass
class ProgressSnapshot:
    """进度快照数据类"""
    iteration: int
    timestamp: float
    losses: Dict[str, float]
    learning_rate: float
    protected_image: Optional[torch.Tensor] = None
    protected_latents: Optional[torch.Tensor] = None
    additional_data: Optional[Dict[str, Any]] = None

class AdvancedProgressBar:
    """高级进度条管理器"""
    
    def __init__(
        self,
        total_iterations: int,
        description: str = "优化进度",
        save_dir: Optional[str] = None,
        save_frequency: int = 10,
        display_metrics: List[str] = None,
        smoothing_window: int = 5
    ):
        """
        初始化进度条管理器
        
        Args:
            total_iterations: 总迭代次数
            description: 进度描述
            save_dir: 保存目录
            save_frequency: 保存频率（每N次迭代）
            display_metrics: 显示的指标列表
            smoothing_window: 平滑窗口大小
        """
        self.total_iterations = total_iterations
        self.description = description
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_frequency = save_frequency
        self.display_metrics = display_metrics or ["total_loss", "id_loss", "lpips_loss"]
        self.smoothing_window = smoothing_window
        
        # 初始化保存目录
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir = self.save_dir / "images"
            self.data_dir = self.save_dir / "data"
            self.plots_dir = self.save_dir / "plots"
            
            for directory in [self.images_dir, self.data_dir, self.plots_dir]:
                directory.mkdir(exist_ok=True)
        
        # 进度跟踪
        self.pbar = None
        self.start_time = None
        self.loss_history = defaultdict(list)
        self.snapshots = []
        self.smoothed_losses = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.image_processor = ImageProcessor()
        
        # 实时绘图（可选）
        self.live_plotting = False
        self.plot_thread = None
        self.plot_lock = threading.Lock()
        
        logger.info(f"进度条管理器初始化: {total_iterations}次迭代, 保存到 {save_dir}")
    
    def start(self):
        """开始进度跟踪"""
        self.start_time = time.time()
        self.pbar = tqdm(
            total=self.total_iterations,
            desc=self.description,
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        logger.info("开始进度跟踪")
    
    def update(
        self,
        iteration: int,
        losses: Dict[str, float],
        learning_rate: float,
        protected_image: Optional[torch.Tensor] = None,
        protected_latents: Optional[torch.Tensor] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """
        更新进度
        
        Args:
            iteration: 当前迭代次数
            losses: 损失字典
            learning_rate: 学习率
            protected_image: 保护后的图像
            protected_latents: 保护后的潜空间
            additional_data: 附加数据
        """
        if self.pbar is None:
            raise RuntimeError("请先调用start()方法")
        
        # 更新损失历史
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.loss_history[key].append(value)
            self.smoothed_losses[key].append(value)
        
        # 计算平滑损失
        smoothed_metrics = {}
        for key in self.display_metrics:
            if key in self.smoothed_losses and len(self.smoothed_losses[key]) > 0:
                smoothed_metrics[key] = np.mean(list(self.smoothed_losses[key]))
        
        # 更新进度条显示
        postfix_dict = {}
        for key, value in smoothed_metrics.items():
            if 'loss' in key.lower():
                postfix_dict[key] = f"{value:.6f}"
            else:
                postfix_dict[key] = f"{value:.4f}"
        
        postfix_dict['lr'] = f"{learning_rate:.2e}"
        
        # 计算ETA
        if iteration > 0:
            elapsed_time = time.time() - self.start_time
            time_per_iter = elapsed_time / iteration
            remaining_iters = self.total_iterations - iteration
            eta_seconds = time_per_iter * remaining_iters
            eta_formatted = self._format_time(eta_seconds)
            postfix_dict['ETA'] = eta_formatted
        
        self.pbar.set_postfix(**postfix_dict)
        self.pbar.update(1)
        
        # 保存中间结果
        if self.save_dir and iteration % self.save_frequency == 0:
            self._save_intermediate_results(
                iteration, losses, learning_rate, 
                protected_image, protected_latents, additional_data
            )
        
        # 创建快照
        snapshot = ProgressSnapshot(
            iteration=iteration,
            timestamp=time.time(),
            losses=losses.copy(),
            learning_rate=learning_rate,
            protected_image=protected_image.clone().detach() if protected_image is not None else None,
            protected_latents=protected_latents.clone().detach() if protected_latents is not None else None,
            additional_data=additional_data
        )
        self.snapshots.append(snapshot)
    
    def _save_intermediate_results(
        self,
        iteration: int,
        losses: Dict[str, float],
        learning_rate: float,
        protected_image: Optional[torch.Tensor],
        protected_latents: Optional[torch.Tensor],
        additional_data: Optional[Dict[str, Any]]
    ):
        """保存中间结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存图像（如果提供）
            if protected_image is not None:
                image_path = self.images_dir / f"iter_{iteration:04d}_{timestamp}.png"
                self.image_processor.save_image(protected_image, str(image_path))
            
            # 保存潜空间（如果提供）
            if protected_latents is not None:
                latents_path = self.data_dir / f"latents_{iteration:04d}_{timestamp}.pt"
                torch.save(protected_latents, latents_path)
            
            # 保存损失和元数据
            metadata = {
                'iteration': iteration,
                'timestamp': timestamp,
                'losses': {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in losses.items()},
                'learning_rate': learning_rate,
                'additional_data': additional_data
            }
            
            metadata_path = self.data_dir / f"metadata_{iteration:04d}_{timestamp}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"保存中间结果失败 (迭代 {iteration}): {e}")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m{seconds%60:.0f}s"
        else:
            return f"{seconds/3600:.0f}h{(seconds%3600)/60:.0f}m"
    
    def finish(self, final_losses: Optional[Dict[str, float]] = None):
        """完成进度跟踪"""
        if self.pbar:
            self.pbar.close()
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        logger.info(f"优化完成: 总时间 {self._format_time(total_time)}")
        
        # 保存最终结果摘要
        if self.save_dir:
            self._save_final_summary(final_losses, total_time)
    
    def _save_final_summary(self, final_losses: Optional[Dict[str, float]], total_time: float):
        """保存最终摘要"""
        try:
            summary = {
                'total_iterations': len(self.snapshots),
                'total_time_seconds': total_time,
                'total_time_formatted': self._format_time(total_time),
                'final_losses': final_losses or {},
                'loss_history': {k: list(v) for k, v in self.loss_history.items()},
                'timestamp': datetime.now().isoformat(),
                'snapshots_count': len(self.snapshots)
            }
            
            summary_path = self.save_dir / "optimization_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # 保存快照数据
            snapshots_path = self.save_dir / "snapshots.pkl"
            with open(snapshots_path, 'wb') as f:
                pickle.dump(self.snapshots, f)
            
            logger.info(f"最终摘要已保存到: {summary_path}")
            
        except Exception as e:
            logger.error(f"保存最终摘要失败: {e}")
    
    def plot_loss_curves(self, save_path: Optional[str] = None, show: bool = False):
        """绘制损失曲线"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 绘制主要损失
            for i, (loss_name, values) in enumerate(self.loss_history.items()):
                if 'total' in loss_name.lower() or len(self.loss_history) <= 5:
                    plt.subplot(2, 2, min(i + 1, 4))
                    plt.plot(values, label=loss_name)
                    plt.title(f"{loss_name} 变化曲线")
                    plt.xlabel("迭代次数")
                    plt.ylabel("损失值")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"损失曲线已保存到: {save_path}")
            
            if show:
                plt.show()
            
            if not show:
                plt.close()
                
        except Exception as e:
            logger.error(f"绘制损失曲线失败: {e}")
    
    def get_best_iteration(self, metric: str = "total_loss", minimize: bool = True) -> Optional[int]:
        """
        获取最佳迭代次数
        
        Args:
            metric: 评估指标
            minimize: 是否最小化指标
            
        Returns:
            最佳迭代次数
        """
        if metric not in self.loss_history or len(self.loss_history[metric]) == 0:
            return None
        
        values = self.loss_history[metric]
        if minimize:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return best_idx
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[ProgressSnapshot]:
        """
        从检查点加载进度状态
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            进度快照
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if isinstance(checkpoint, ProgressSnapshot):
                return checkpoint
            else:
                logger.warning("检查点文件格式不正确")
                return None
                
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None

class OptimizationLogger:
    """优化过程日志记录器"""
    
    def __init__(self, log_dir: str, experiment_name: str = "optimization"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建唯一的实验ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        self.experiment_dir = self.log_dir / self.experiment_id
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 设置日志文件
        log_file = self.experiment_dir / "optimization.log"
        self.logger = logging.getLogger(f"optimization_{self.experiment_id}")
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"优化日志记录器初始化: {self.experiment_id}")
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置信息"""
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"配置已记录: {config}")
    
    def log_iteration(self, iteration: int, losses: Dict[str, float], **kwargs):
        """记录迭代信息"""
        loss_str = ", ".join([f"{k}={v:.6f}" for k, v in losses.items()])
        extra_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        
        message = f"Iter {iteration}: {loss_str}"
        if extra_str:
            message += f", {extra_str}"
        
        self.logger.info(message)
    
    def log_milestone(self, message: str):
        """记录里程碑事件"""
        self.logger.info(f"MILESTONE: {message}")
    
    def get_experiment_dir(self) -> Path:
        """获取实验目录"""
        return self.experiment_dir

def create_progress_manager(
    total_iterations: int,
    save_dir: Optional[str] = None,
    experiment_name: str = "optimization",
    save_frequency: int = 10,
    **kwargs
) -> AdvancedProgressBar:
    """
    创建进度管理器的便捷函数
    
    Args:
        total_iterations: 总迭代次数
        save_dir: 保存目录
        experiment_name: 实验名称
        save_frequency: 保存频率
        **kwargs: 其他参数
        
    Returns:
        进度管理器实例
    """
    if save_dir is None:
        save_dir = f"experiments/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return AdvancedProgressBar(
        total_iterations=total_iterations,
        save_dir=save_dir,
        save_frequency=save_frequency,
        **kwargs
    )

def test_progress_utils():
    """测试进度工具"""
    print("🧪 测试进度工具...")
    
    try:
        # 创建测试目录
        test_dir = "test_progress"
        
        # 测试进度条管理器
        progress_manager = create_progress_manager(
            total_iterations=20,
            save_dir=test_dir,
            experiment_name="test_exp",
            save_frequency=5
        )
        
        print("✅ 进度管理器创建成功")
        
        # 模拟优化过程
        progress_manager.start()
        
        for i in range(20):
            # 模拟损失
            losses = {
                'total_loss': 1.0 - i * 0.04 + np.random.normal(0, 0.01),
                'id_loss': 0.5 - i * 0.02 + np.random.normal(0, 0.005),
                'lpips_loss': 0.3 - i * 0.01 + np.random.normal(0, 0.003)
            }
            
            learning_rate = 0.01 * (0.95 ** (i // 5))
            
            # 模拟图像数据
            if i % 5 == 0:
                fake_image = torch.rand(1, 3, 64, 64)
                fake_latents = torch.rand(1, 4, 8, 8)
            else:
                fake_image = None
                fake_latents = None
            
            progress_manager.update(
                iteration=i,
                losses=losses,
                learning_rate=learning_rate,
                protected_image=fake_image,
                protected_latents=fake_latents,
                additional_data={'step_type': 'normal'}
            )
            
            time.sleep(0.05)  # 模拟处理时间
        
        progress_manager.finish({'final_total_loss': 0.1})
        
        print("✅ 优化过程模拟完成")
        
        # 测试最佳迭代查找
        best_iter = progress_manager.get_best_iteration('total_loss')
        print(f"✅ 最佳迭代: {best_iter}")
        
        # 测试损失曲线绘制
        plot_path = f"{test_dir}/loss_curves.png"
        progress_manager.plot_loss_curves(save_path=plot_path)
        print(f"✅ 损失曲线已保存: {plot_path}")
        
        # 测试日志记录器
        logger = OptimizationLogger(test_dir, "test_logging")
        logger.log_config({"learning_rate": 0.01, "batch_size": 1})
        logger.log_iteration(0, {'loss': 1.0}, lr=0.01)
        logger.log_milestone("测试完成")
        
        print("✅ 日志记录器测试完成")
        
        # 清理测试文件
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("✅ 测试文件已清理")
        
        print("🎉 进度工具测试全部通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_progress_utils() 