"""
数据集加载与处理模块
参考 RARE-UNet + LB-UNet 的设计，适用于 DR²-UNet 项目
支持 ISIC2018 皮肤病灶数据集
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict, Any, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class ISICDataset(Dataset):
    """
    ISIC 皮肤病灶分割数据集
    """

    def __init__(
            self,
            data_dir: str,
            phase: str = "train",
            transform: Optional[Callable] = None,
            target_size: Tuple[int, int] = (256, 256),
            normalize: bool = True,
    ):
        """
        Args:
            data_dir: 数据根目录 (例如 ../数据集/isic2018)
            phase: 数据集阶段 ('train', 'val', 'test')
            transform: 数据增强 transform
            target_size: 目标图像尺寸（最大尺寸）
            normalize: 是否归一化
        """
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.transform = transform
        self.target_size = target_size  # 最大分辨率
        self.normalize = normalize

        # 1. 拼接路径: base_path + phase + images/masks
        self.images_dir = self.data_dir / phase / "images"
        self.masks_dir = self.data_dir / phase / "masks"

        # 2. 检查路径
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # 3. 获取文件列表 (使用 self.image_files 和 self.mask_files)
        self.image_files = sorted(
            [
                f
                for f in os.listdir(self.images_dir)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        self.mask_files = sorted(
            [f for f in os.listdir(self.masks_dir) if f.endswith(".png")]
        )

        # 4. 匹配图像和掩码
        self.samples = self._match_samples()

        logger.info(f"Loaded {len(self.samples)} samples for {phase} phase")

    def _match_samples(self) -> List[Tuple[str, str]]:
        """匹配图像和掩码文件"""
        samples = []

        # 创建掩码文件名到文件名的映射 (处理可能的后缀差异，如 _segmentation)
        # key: 文件名无后缀, value: 原始文件名
        mask_map = {}
        for f in self.mask_files:
            name = os.path.splitext(f)[0]
            # 去除常见的掩码后缀以便匹配
            clean_name = name.replace("_segmentation", "").replace("_seg", "").replace("_mask", "")
            mask_map[clean_name] = f

        for img_file in self.image_files:
            img_name = os.path.splitext(img_file)[0]

            # 查找对应的掩码
            if img_name in mask_map:
                mask_file = mask_map[img_name]
                samples.append((img_file, mask_file))
            else:
                logger.warning(f"No mask found for image: {img_file}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据样本
        Returns:
            image: [C, H, W] float tensor
            mask: [H, W] float tensor
        """
        img_file, mask_file = self.samples[idx]

        # 加载图像
        img_path = self.images_dir / img_file
        img = np.array(Image.open(img_path).convert("RGB"))

        # 加载掩码
        mask_path = self.masks_dir / mask_file
        mask = np.array(Image.open(mask_path).convert("L"))

        # 二值化掩码 (0 或 1)
        mask = (mask > 127).astype(np.float32)

        target_size = self.target_size

        # Resize 到目标分辨率
        img = self._resize_image(img, target_size)
        mask = self._resize_mask(mask, target_size)

        # 转为 Tensor
        # Image: [H, W, C] -> [C, H, W], float32, 0-255
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # Mask: [H, W], float32
        mask = torch.from_numpy(mask).float()

        # 归一化图像 (ToTensor + Normalize)
        if self.normalize:
            # 转换到 [0, 1]
            img = img / 255.0
            # 标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std

        # 数据增强（仅训练阶段）
        if self.transform is not None and self.phase == "train":
            img, mask = self.transform(img, mask)

        return img, mask

    def _resize_image(self, img: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if target_size is None:
            target_size = self.target_size
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(target_size, Image.BILINEAR)
        return np.array(pil_img)

    def _resize_mask(self, mask: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if target_size is None:
            target_size = self.target_size
        # 掩码使用最近邻插值，避免产生新的灰度值
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        pil_mask = pil_mask.resize(target_size, Image.NEAREST)
        return (np.array(pil_mask) / 255.0).astype(np.float32)


class ISICDatasetFromFolder(ISICDataset):
    """
    ISIC 数据集（从文件夹加载，与 ISICDataset 相同接口）
    保留此别名以保持兼容性
    """

    pass


def build_transforms(cfg, phase: str = "train") -> Optional[Callable]:
    """
    构建数据增强 transform

    Args:
        cfg: 数据集配置
        phase: 数据集阶段

    Returns:
        数据增强 transform 函数
    """
    if phase != "train":
        # 验证/测试阶段不做增强
        return None

    # 从配置读取增强参数
    aug_config = cfg.get("augmentation", {})

    if not aug_config.get("enabled", False):
        return None

    transforms_list = []

    for transform_config in aug_config.get("transforms", []):
        transform_name = list(transform_config.keys())[0]
        transform_params = transform_config[transform_name]

        if transform_name == "RandomHorizontalFlip":
            p = transform_params.get("p", 0.5)
            transforms_list.append(RandomHorizontalFlip(p=p))

        elif transform_name == "RandomVerticalFlip":
            p = transform_params.get("p", 0.5)
            transforms_list.append(RandomVerticalFlip(p=p))

        elif transform_name == "RandomRotation":
            degrees = transform_params.get("degrees", [0, 360])
            p = transform_params.get("p", 0.5)
            transforms_list.append(RandomRotation(degrees=degrees, p=p))

        elif transform_name == "ColorJitter":
            brightness = transform_params.get("brightness", 0.2)
            contrast = transform_params.get("contrast", 0.2)
            p = transform_params.get("p", 0.3)
            transforms_list.append(
                ColorJitter(brightness=brightness, contrast=contrast, p=p)
            )

    if transforms_list:
        return Compose(transforms_list)
    return None


class RandomHorizontalFlip:
    """随机水平翻转"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        if torch.rand(1).item() < self.p:
            return img.flip(-1), mask.flip(-1)
        return img, mask


class RandomVerticalFlip:
    """随机垂直翻转"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        if torch.rand(1).item() < self.p:
            return img.flip(-2), mask.flip(-2)
        return img, mask


class RandomRotation:
    """随机旋转（90度倍数）"""

    def __init__(self, degrees: List[int] = [0, 90, 180, 270], p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        if torch.rand(1).item() < self.p:
            k = torch.randint(0, len(self.degrees), (1,)).item()
            if k > 0:
                img = torch.rot90(img, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[0, 1])
        return img, mask


class ColorJitter:
    """颜色抖动"""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        if torch.rand(1).item() < self.p:
            # Brightness
            if self.brightness > 0:
                factor = 1.0 + torch.rand(1).item() * self.brightness
                img = img * factor

            # Contrast
            if self.contrast > 0:
                factor = 1.0 + torch.rand(1).item() * self.contrast
                mean = img.mean(dim=(1, 2), keepdim=True)
                img = (img - mean) * factor + mean

        return img, mask


class Compose:
    """组合多个数据增强"""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


def build_dataset(cfg, phase: str = "train") -> Dataset:
    """
    数据集工厂函数

    Args:
        cfg: Hydra 配置对象
        phase: 数据集阶段

    Returns:
        数据集实例
    """
    # 获取配置参数
    data_dir = cfg.dataset.base_path
    target_size = tuple(cfg.dataset.target_shape)

    # 构建数据增强
    transform = build_transforms(cfg.dataset, phase)

    # 创建数据集
    dataset = ISICDataset(
        data_dir=data_dir,
        phase=phase,
        transform=transform,
        target_size=target_size,
        normalize=True,
    )

    return dataset


def get_train_val_split(
    data_dir: str, train_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    划分训练集和验证集

    Args:
        data_dir: 数据根目录
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        train_files: 训练文件列表
        val_files: 验证文件列表
    """
    # 获取所有图像文件
    train_images_dir = Path(data_dir) / "train" / "images"

    if not train_images_dir.exists():
        raise FileNotFoundError(f"Train images directory not found: {train_images_dir}")

    image_files = sorted(
        [
            f
            for f in os.listdir(train_images_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    # 随机划分
    np.random.seed(seed)
    indices = np.random.permutation(len(image_files))
    split_idx = int(len(image_files) * train_ratio)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_files = [image_files[i] for i in train_indices]
    val_files = [image_files[i] for i in val_indices]

    logger.info(
        f"Split {len(image_files)} images into {len(train_files)} train and {len(val_files)} val"
    )

    return train_files, val_files
