"""
ISIC2018 数据预处理脚本
功能：
1. 从原始 ISIC2018 数据（images + masks）划分出 train/val/test
2. 组织成代码所需的目录结构

使用方法：
    python data/preprocess_data.py --input-dir <原始数据路径> --output-dir <输出路径>

示例：
    python data/preprocess_data.py --input-dir ../数据集/ISIC2018_raw --output-dir ../数据集/ISIC2018
"""

import os
import argparse
import shutil
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ISIC2018 数据预处理：划分 train/val/test"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="原始数据目录（包含 images 和 masks 子目录）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="输出目录（将创建 train/val/test 子目录）",
    )
    parser.add_argument(
        "--train-ratio",
        "-tr",
        type=float,
        default=0.8,
        help="训练集比例 (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio", "-vr", type=float, default=0.1, help="验证集比例 (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio", "-te", type=float, default=0.1, help="测试集比例 (default: 0.1)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="随机种子 (default: 42)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="1",
        choices=["1", "2"],
        help="ISIC 任务类型: 1=Segmentation, 2=Attribute detection (default: 1)",
    )

    args = parser.parse_args()

    # 检查比例之和
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"比例之和必须为 1.0，当前为 {total}")

    return args


def find_image_mask_pairs(
    images_dir: Path, masks_dir: Path, task: str = "1"
) -> List[Tuple[str, str]]:
    """
    查找图像和掩码的配对

    Args:
        images_dir: 图像目录
        masks_dir: 掩码目录
        task: ISIC 任务类型

    Returns:
        [(image_filename, mask_filename), ...] 列表
    """
    # 获取图像文件
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [
        f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)
    ]

    # 获取掩码文件
    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(".png")]

    # 创建掩码文件名到完整名的映射（去除扩展名）
    mask_names = {os.path.splitext(f)[0]: f for f in mask_files}

    # 匹配图像和掩码
    pairs = []
    unmatched_images = []

    for img_file in sorted(image_files):
        img_name = os.path.splitext(img_file)[0]

        # 尝试多种匹配策略
        mask_name = None

        # 1. 精确匹配
        if img_name in mask_names:
            mask_name = mask_names[img_name]
        else:
            # 2. 前缀匹配（ISIC2018 中图像可能有 _SEG 之类的后缀）
            for m in mask_names:
                if m.startswith(img_name) or img_name.startswith(m):
                    mask_name = mask_names[m]
                    break

        if mask_name:
            pairs.append((img_file, mask_name))
        else:
            unmatched_images.append(img_file)

    if unmatched_images:
        logger.warning(f"找到 {len(unmatched_images)} 张图像没有对应的掩码")

    logger.info(f"成功匹配 {len(pairs)} 对图像-掩码")
    return pairs


def split_dataset(
    pairs: List[Tuple[str, str]], train_ratio: float, val_ratio: float, seed: int = 42
) -> Tuple[List, List, List]:
    """
    划分数据集

    Args:
        pairs: 图像-掩码对列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(pairs))

    n_train = int(len(pairs) * train_ratio)
    n_val = int(len(pairs) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]
    test_pairs = [pairs[i] for i in test_indices]

    logger.info(
        f"划分结果: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}"
    )

    return train_pairs, val_pairs, test_pairs


def copy_files(
    pairs: List[Tuple[str, str]],
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    phase: str,
):
    """
    复制文件到指定目录

    Args:
        pairs: 图像-掩码对列表
        images_dir: 原始图像目录
        masks_dir: 原始掩码目录
        output_dir: 输出根目录
        phase: 阶段名称 (train/val/test)
    """
    phase_images_dir = output_dir / phase / "images"
    phase_masks_dir = output_dir / phase / "masks"

    phase_images_dir.mkdir(parents=True, exist_ok=True)
    phase_masks_dir.mkdir(parents=True, exist_ok=True)

    for img_file, mask_file in tqdm(pairs, desc=f"Copying {phase}"):
        # 复制图像
        src_img = images_dir / img_file
        dst_img = phase_images_dir / img_file
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # 复制掩码
        src_mask = masks_dir / mask_file
        dst_mask = phase_masks_dir / mask_file
        if not dst_mask.exists():
            shutil.copy2(src_mask, dst_mask)


def validate_output(output_dir: Path) -> bool:
    """
    验证输出目录结构

    Args:
        output_dir: 输出目录

    Returns:
        是否验证通过
    """
    required_dirs = [
        "train/images",
        "train/masks",
        "val/images",
        "val/masks",
        "test/images",
        "test/masks",
    ]

    all_valid = True
    for dir_path in required_dirs:
        full_path = output_dir / dir_path
        if not full_path.exists():
            logger.error(f"缺少目录: {full_path}")
            all_valid = False
        else:
            # 统计文件数
            files = list(full_path.iterdir())
            logger.info(f"  {dir_path}: {len(files)} files")

    return all_valid


def main():
    """主函数"""
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 50)
    logger.info("ISIC2018 数据预处理")
    logger.info("=" * 50)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(
        f"划分比例: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}"
    )
    logger.info(f"随机种子: {args.seed}")
    logger.info("=" * 50)

    # 检查输入目录
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 查找图像和掩码目录
    # ISIC2018 可能的目录结构：
    # 1. input_dir/ISIC2018_Task1_Training_Input/ + ISIC2018_Task1_Training_GroundTruth/
    # 2. input_dir/images/ + input_dir/masks/
    # 3. input_dir/ 直接包含图像和掩码

    # 尝试自动检测目录结构
    images_dir = None
    masks_dir = None

    # 方案1: 直接子目录
    if (input_dir / "images").exists() and (input_dir / "masks").exists():
        images_dir = input_dir / "images"
        masks_dir = input_dir / "masks"
    else:
        # 方案2: 搜索可能的子目录
        possible_image_dirs = [
            "ISIC2018_Task1_Training_Input",
            "ISIC2018_Task1-2_Training_Input",
            "ISIC2018_Training_Input",
            "train",
            "input",
        ]
        possible_mask_dirs = [
            "ISIC2018_Task1_Training_GroundTruth",
            "ISIC2018_Task1-2_Training_GroundTruth",
            "ISIC2018_Training_GroundTruth",
            "train_mask",
            "mask",
            "ground_truth",
        ]

        for img_d in possible_image_dirs:
            for mask_d in possible_mask_dirs:
                if (input_dir / img_d).exists() and (input_dir / mask_d).exists():
                    images_dir = input_dir / img_d
                    masks_dir = input_dir / mask_d
                    logger.info(f"自动检测到目录结构: {img_d} + {mask_d}")
                    break
            if images_dir:
                break

    if images_dir is None or masks_dir is None:
        raise FileNotFoundError(
            f"无法找到图像和掩码目录。\n"
            f"请确保输入目录包含以下结构之一:\n"
            f"  1. input/images/ + input/masks/\n"
            f"  2. input/ISIC2018_Task1_Training_Input/ + input/ISIC2018_Task1_Training_GroundTruth/"
        )

    logger.info(f"图像目录: {images_dir}")
    logger.info(f"掩码目录: {masks_dir}")

    # 查找图像-掩码对
    pairs = find_image_mask_pairs(images_dir, masks_dir, args.task)

    if len(pairs) == 0:
        raise ValueError("未找到任何图像-掩码对，请检查数据目录结构")

    # 划分数据集
    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs, args.train_ratio, args.val_ratio, args.seed
    )

    # 创建输出目录并复制文件
    logger.info("开始复制文件...")
    copy_files(train_pairs, images_dir, masks_dir, output_dir, "train")
    copy_files(val_pairs, images_dir, masks_dir, output_dir, "val")
    copy_files(test_pairs, images_dir, masks_dir, output_dir, "test")

    # 验证输出
    logger.info("验证输出目录...")
    if validate_output(output_dir):
        logger.info("=" * 50)
        logger.info("预处理完成！")
        logger.info("=" * 50)
    else:
        logger.error("输出验证失败")
        return 1

    # 打印统计信息
    total = len(train_pairs) + len(val_pairs) + len(test_pairs)
    logger.info(f"总计: {total} 样本")
    logger.info(f"  train: {len(train_pairs)} ({len(train_pairs) / total * 100:.1f}%)")
    logger.info(f"  val:   {len(val_pairs)} ({len(val_pairs) / total * 100:.1f}%)")
    logger.info(f"  test:  {len(test_pairs)} ({len(test_pairs) / total * 100:.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
