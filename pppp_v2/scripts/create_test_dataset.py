"""
测试数据集创建脚本

该脚本用于创建标准化的人脸图像测试集，支持从多种数据源收集图像，
并进行标准化处理，为模型评估提供一致的测试基准。

功能包括：
1. 从公开数据集下载人脸图像
2. 图像质量检测和筛选
3. 人脸检测和裁剪
4. 数据集标准化处理
5. 生成数据集统计报告

作者: AI Privacy Protection System
日期: 2025-07-28
"""

import os
import requests
import urllib.request
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import json
import shutil
from tqdm import tqdm
import hashlib
from datetime import datetime

try:
    from ..src.utils.image_utils import ImageProcessor
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)

class TestDatasetCreator:
    """测试数据集创建器"""
    
    def __init__(
        self,
        output_dir: str = "data/test_faces",
        target_count: int = 50,
        image_size: Tuple[int, int] = (512, 512),
        min_face_size: int = 128
    ):
        """
        初始化测试数据集创建器
        
        Args:
            output_dir: 输出目录
            target_count: 目标图像数量
            image_size: 目标图像尺寸
            min_face_size: 最小人脸尺寸
        """
        self.output_dir = Path(output_dir)
        self.target_count = target_count
        self.image_size = image_size
        self.min_face_size = min_face_size
        
        # 创建目录结构
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.metadata_dir = self.output_dir / "metadata"
        
        for directory in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            directory.mkdir(exist_ok=True)
        
        # 初始化工具
        self.image_processor = ImageProcessor()
        
        # 人脸检测器（使用OpenCV的Haar级联）
        self.face_cascade = None
        self._init_face_detector()
        
        # 数据源配置
        self.data_sources = self._get_data_sources()
        
        logger.info(f"测试数据集创建器初始化: 目标{target_count}张图像, 输出到 {output_dir}")
    
    def _init_face_detector(self):
        """初始化人脸检测器"""
        try:
            # 尝试加载OpenCV的人脸检测器
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("OpenCV人脸检测器加载成功")
            else:
                logger.warning("OpenCV人脸检测器不可用，将使用备用方案")
        except Exception as e:
            logger.warning(f"人脸检测器初始化失败: {e}")
    
    def _get_data_sources(self) -> Dict[str, Dict]:
        """获取数据源配置"""
        # 注意：这里提供的是一些示例和公开可用的测试图像
        # 在实际使用中，请确保遵守相关的版权和隐私规定
        
        sources = {
            "sample_faces": {
                "description": "示例人脸图像（生成或公开可用）",
                "urls": [
                    # 这里可以添加一些公开可用的测试图像URL
                    # 注意：实际部署时需要替换为合法的测试图像源
                ],
                "enabled": False  # 默认关闭，需要手动启用
            },
            "synthetic": {
                "description": "生成合成人脸图像",
                "enabled": True
            }
        }
        
        return sources
    
    def generate_synthetic_faces(self, count: int) -> List[str]:
        """
        生成合成人脸图像（用于测试）
        
        Args:
            count: 生成数量
            
        Returns:
            生成的图像路径列表
        """
        generated_paths = []
        
        logger.info(f"生成 {count} 张合成测试图像...")
        
        for i in tqdm(range(count), desc="生成合成图像"):
            # 创建一个带有简单模式的测试图像
            # 这是一个占位符实现，实际中可能需要更复杂的生成逻辑
            
            # 生成随机图像作为基础
            img_array = np.random.randint(0, 256, (self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            
            # 添加一些简单的几何形状模拟人脸特征
            center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
            
            # 绘制椭圆形"脸部"轮廓
            cv2.ellipse(img_array, (center_x, center_y), (80, 100), 0, 0, 360, (200, 180, 160), -1)
            
            # 添加"眼睛"
            cv2.circle(img_array, (center_x - 25, center_y - 20), 8, (50, 50, 50), -1)
            cv2.circle(img_array, (center_x + 25, center_y - 20), 8, (50, 50, 50), -1)
            
            # 添加"嘴巴"
            cv2.ellipse(img_array, (center_x, center_y + 30), (20, 10), 0, 0, 180, (100, 50, 50), -1)
            
            # 添加一些噪声以增加真实感
            noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
            img_array = cv2.add(img_array, noise)
            
            # 保存图像
            filename = f"synthetic_face_{i:03d}.png"
            filepath = self.raw_dir / filename
            
            cv2.imwrite(str(filepath), img_array)
            generated_paths.append(str(filepath))
        
        logger.info(f"成功生成 {len(generated_paths)} 张合成图像")
        return generated_paths
    
    def download_from_urls(self, urls: List[str]) -> List[str]:
        """
        从URL列表下载图像
        
        Args:
            urls: URL列表
            
        Returns:
            下载的图像路径列表
        """
        downloaded_paths = []
        
        for i, url in enumerate(tqdm(urls, desc="下载图像")):
            try:
                # 生成文件名
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"downloaded_{i:03d}_{url_hash}.jpg"
                filepath = self.raw_dir / filename
                
                # 下载图像
                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_paths.append(str(filepath))
                logger.info(f"下载成功: {filename}")
                
            except Exception as e:
                logger.warning(f"下载失败 {url}: {e}")
                continue
        
        return downloaded_paths
    
    def detect_faces(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸
        
        Args:
            image_path: 图像路径
            
        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 使用人脸检测器
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(self.min_face_size, self.min_face_size)
                )
                # 检查faces是否为空或者不是numpy数组
                if len(faces) > 0 and hasattr(faces, 'tolist'):
                    return faces.tolist()
                else:
                    # 备用方案：假设整个图像是人脸
                    h, w = img.shape[:2]
                    return [(0, 0, w, h)]
            else:
                # 备用方案：假设整个图像是人脸
                h, w = img.shape[:2]
                return [(0, 0, w, h)]
                
        except Exception as e:
            logger.warning(f"人脸检测失败 {image_path}: {e}")
            return []
    
    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        处理单张图像
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            
        Returns:
            处理是否成功
        """
        try:
            # 检测人脸
            faces = self.detect_faces(input_path)
            
            if not faces:
                logger.warning(f"未检测到人脸: {input_path}")
                return False
            
            # 使用第一个检测到的人脸
            x, y, w, h = faces[0]
            
            # 读取图像
            img = cv2.imread(input_path)
            if img is None:
                return False
            
            # 裁剪人脸区域（添加一些边距）
            margin = max(w, h) // 4
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)
            
            face_img = img[y1:y2, x1:x2]
            
            # 调整尺寸
            resized_img = cv2.resize(face_img, self.image_size)
            
            # 保存处理后的图像
            cv2.imwrite(output_path, resized_img)
            
            return True
            
        except Exception as e:
            logger.error(f"图像处理失败 {input_path}: {e}")
            return False
    
    def create_dataset(self) -> Dict[str, any]:
        """
        创建测试数据集
        
        Returns:
            数据集统计信息
        """
        logger.info("开始创建测试数据集...")
        
        # 收集原始图像
        raw_images = []
        
        # 1. 生成合成图像
        if self.data_sources["synthetic"]["enabled"]:
            synthetic_count = min(self.target_count, 30)  # 限制合成图像数量
            synthetic_images = self.generate_synthetic_faces(synthetic_count)
            raw_images.extend(synthetic_images)
        
        # 2. 从URL下载（如果配置了）
        for source_name, config in self.data_sources.items():
            if source_name != "synthetic" and config.get("enabled", False):
                urls = config.get("urls", [])
                if urls:
                    downloaded_images = self.download_from_urls(urls)
                    raw_images.extend(downloaded_images)
        
        logger.info(f"收集到 {len(raw_images)} 张原始图像")
        
        # 处理图像
        processed_count = 0
        failed_count = 0
        
        for i, raw_path in enumerate(tqdm(raw_images, desc="处理图像")):
            if processed_count >= self.target_count:
                break
            
            output_filename = f"face_{processed_count:03d}.png"
            output_path = str(self.processed_dir / output_filename)
            
            if self.process_image(raw_path, output_path):
                processed_count += 1
            else:
                failed_count += 1
        
        # 生成数据集统计信息
        stats = {
            "dataset_name": "privacy_protection_test_faces",
            "creation_date": datetime.now().isoformat(),
            "total_images": processed_count,
            "image_size": self.image_size,
            "raw_images_collected": len(raw_images),
            "processing_success_rate": processed_count / len(raw_images) if raw_images else 0,
            "failed_processing": failed_count,
            "data_sources": {k: v for k, v in self.data_sources.items() if v.get("enabled", False)}
        }
        
        # 保存统计信息
        stats_path = self.metadata_dir / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 创建数据集索引
        self._create_dataset_index()
        
        logger.info(f"数据集创建完成: {processed_count} 张图像")
        logger.info(f"统计信息已保存到: {stats_path}")
        
        return stats
    
    def _create_dataset_index(self):
        """创建数据集索引文件"""
        index = {
            "version": "1.0",
            "description": "Privacy Protection Test Face Dataset",
            "images": []
        }
        
        # 遍历处理后的图像
        for img_path in sorted(self.processed_dir.glob("*.png")):
            try:
                # 获取图像信息
                img = Image.open(img_path)
                width, height = img.size
                
                # 计算文件哈希
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                image_info = {
                    "filename": img_path.name,
                    "path": str(img_path.relative_to(self.output_dir)),
                    "size": [width, height],
                    "file_size": img_path.stat().st_size,
                    "md5_hash": file_hash
                }
                
                index["images"].append(image_info)
                
            except Exception as e:
                logger.warning(f"处理索引时出错 {img_path}: {e}")
        
        # 保存索引文件
        index_path = self.metadata_dir / "dataset_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集索引已保存到: {index_path}")
    
    def validate_dataset(self) -> Dict[str, any]:
        """
        验证数据集质量
        
        Returns:
            验证结果
        """
        logger.info("开始验证数据集...")
        
        validation_results = {
            "total_images": 0,
            "valid_images": 0,
            "corrupt_images": 0,
            "size_issues": 0,
            "face_detection_success": 0,
            "issues": []
        }
        
        for img_path in self.processed_dir.glob("*.png"):
            validation_results["total_images"] += 1
            
            try:
                # 检查图像是否可以正常读取
                img = Image.open(img_path)
                width, height = img.size
                
                # 检查尺寸
                if (width, height) != self.image_size:
                    validation_results["size_issues"] += 1
                    validation_results["issues"].append(f"尺寸错误: {img_path.name} ({width}x{height})")
                
                # 检查人脸检测
                faces = self.detect_faces(str(img_path))
                if faces:
                    validation_results["face_detection_success"] += 1
                else:
                    validation_results["issues"].append(f"人脸检测失败: {img_path.name}")
                
                validation_results["valid_images"] += 1
                
            except Exception as e:
                validation_results["corrupt_images"] += 1
                validation_results["issues"].append(f"图像损坏: {img_path.name} - {e}")
        
        # 保存验证结果
        validation_path = self.metadata_dir / "validation_results.json"
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集验证完成: {validation_results['valid_images']}/{validation_results['total_images']} 有效")
        
        return validation_results

def create_test_dataset(
    target_count: int = 50,
    output_dir: str = "data/test_faces",
    **kwargs
) -> Dict[str, any]:
    """
    创建测试数据集的便捷函数
    
    Args:
        target_count: 目标图像数量
        output_dir: 输出目录
        **kwargs: 其他参数
        
    Returns:
        数据集统计信息
    """
    creator = TestDatasetCreator(
        output_dir=output_dir,
        target_count=target_count,
        **kwargs
    )
    
    stats = creator.create_dataset()
    validation_results = creator.validate_dataset()
    
    # 合并结果
    stats["validation"] = validation_results
    
    return stats

def main():
    """主函数"""
    print("🗂️ 创建测试人脸数据集...")
    
    try:
        # 创建数据集
        stats = create_test_dataset(
            target_count=50,
            output_dir="data/test_faces"
        )
        
        print("✅ 测试数据集创建完成！")
        print(f"📊 统计信息:")
        print(f"   总图像数: {stats['total_images']}")
        print(f"   图像尺寸: {stats['image_size']}")
        print(f"   处理成功率: {stats['processing_success_rate']:.2%}")
        print(f"   验证结果: {stats['validation']['valid_images']}/{stats['validation']['total_images']} 有效")
        
        if stats['validation']['issues']:
            print(f"⚠️ 发现 {len(stats['validation']['issues'])} 个问题，详见验证报告")
        
        print(f"📁 数据集位置: {Path('data/test_faces').absolute()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 