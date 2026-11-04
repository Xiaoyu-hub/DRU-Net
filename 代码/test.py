#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试集推理脚本
功能：加载模型，在测试集上推理，输出评估指标和结果
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from omegaconf import OmegaConf
from data.datasets import ISICDataset
from models.dr_unet import DRUNet
from models.unet import UNet2D
from utils.metrics import dice_coefficient, iou_score
from utils.postprocess import PostProcessor


def build_test_loader(cfg):
    """构建测试集 DataLoader"""
    dataset = ISICDataset(
        data_dir=cfg.dataset.base_path,
        phase="test",
        target_size=tuple(cfg.dataset.target_shape),
        normalize=True,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    return test_loader, len(dataset)


def load_model(cfg, checkpoint_path, model_type="dr_unet"):
    """加载模型"""
    if model_type == "dr_unet":
        model = DRUNet(cfg, mode="inference")
    else:
        # Baseline U-Net 也使用配置初始化
        from models.unet import UNet2D

        model = UNet2D(cfg)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return model


def evaluate_model(model, test_loader, device, inference_mode="dynamic", cfg=None):
    """在测试集上评估模型"""
    model.eval()
    model.to(device)

    all_dice = []
    all_iou = []
    all_preds = []
    all_targets = []

    # 路由统计
    router_easy = 0
    router_hard = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)

            # 模型推理
            outputs = model(images)

            # 获取最终输出
            if isinstance(outputs, dict):
                final_output = outputs.get("final_output", outputs.get("light_output"))

                # 路由统计
                if "is_hard" in outputs and outputs["is_hard"] is not None:
                    is_hard = outputs["is_hard"].cpu().numpy()
                    router_easy += (is_hard == 0).sum()
                    router_hard += (is_hard == 1).sum()
            else:
                final_output = outputs

            # 计算指标
            preds = torch.sigmoid(final_output)
            preds = (preds > 0.5).float()

            # 批量计算 Dice 和 IoU (使用torch)
            pred_tensor = preds.cpu()  # [B, 1, H, W]
            target_tensor = targets.cpu()  # [B, H, W]
            
            # 手动计算二分类 Dice 和 IoU
            pred_binary = (pred_tensor > 0.5).squeeze(1).float()  # [B, H, W]
            target_binary = target_tensor.float()  # [B, H, W]
            
            # Dice: 2*|A∩B| / (|A|+|B|)
            intersection = (pred_binary * target_binary).sum(dim=(1, 2))
            union = pred_binary.sum(dim=(1, 2)) + target_binary.sum(dim=(1, 2))
            dice_scores = (2.0 * intersection + 1e-5) / (union + 1e-5)
            
            # IoU: |A∩B| / |A∪B|
            iou_scores = (intersection + 1e-5) / (union - intersection + 1e-5)
            
            # 收集每个样本的分数
            for i in range(preds.shape[0]):
                all_dice.append(dice_scores[i].item())
                all_iou.append(iou_scores[i].item())
                all_preds.append(pred_binary[i].cpu().numpy())
                all_targets.append(target_binary[i].cpu().numpy())

            total_samples += images.shape[0]

    # 汇总结果
    results = {
        "dice": float(np.mean(all_dice)),
        "dice_std": float(np.std(all_dice)),
        "iou": float(np.mean(all_iou)),
        "iou_std": float(np.std(all_iou)),
        "num_samples": total_samples,
    }

    # 路由统计
    if router_easy + router_hard > 0:
        results["router_easy_ratio"] = router_easy / total_samples
        results["router_hard_ratio"] = router_hard / total_samples

    return results, all_preds, all_targets


def save_results(results, output_dir, model_name, test_dataset_size):
    """保存结果到文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 JSON 结果
    results_file = output_dir / f"{model_name}_test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存文本报告
    report_file = output_dir / f"{model_name}_test_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"Test Results Report - {model_name}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Test Dataset Size: {test_dataset_size}\n")
        f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n--- Main Metrics ---\n")
        f.write(f"Dice Score: {results['dice']:.4f} ± {results['dice_std']:.4f}\n")
        f.write(f"IoU Score:  {results['iou']:.4f} ± {results['iou_std']:.4f}\n")

        if "router_easy_ratio" in results:
            f.write(f"\n--- Router Statistics ---\n")
            f.write(
                f"Easy Samples (Light Branch): {results['router_easy_ratio'] * 100:.2f}%\n"
            )
            f.write(
                f"Hard Samples (Heavy Branch): {results['router_hard_ratio'] * 100:.2f}%\n"
            )

        f.write(f"\n{'=' * 60}\n")

    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {report_file}")

    return results_file, report_file


def main():
    parser = argparse.ArgumentParser(description="Test Set Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dr_unet",
        choices=["dr_unet", "unet"],
        help="Model type",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=["light", "heavy", "dynamic"],
        help="Inference mode (only for DR-UNet)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/test_results", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    args = parser.parse_args()

    # 加载配置 - 使用 Hydra 方式加载（支持 defaults）
    from hydra import compose, initialize_config_dir
    import os

    config_dir = os.path.abspath("config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="base.yaml")

    # 如果是 DR-UNet，设置推理模式
    if args.model == "dr_unet":
        if "inference" in cfg.architecture:
            cfg.architecture.inference.mode = args.mode
        else:
            cfg.architecture.inference = {"mode": args.mode}

    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Inference Mode: {args.mode}")
    print(f"=" * 60)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print("Loading model...")
    model = load_model(cfg, args.checkpoint, args.model)

    # 构建测试集 DataLoader
    print("Loading test dataset...")
    try:
        test_loader, test_size = build_test_loader(cfg)
        print(f"Test samples: {test_size}")
    except Exception as e:
        print(f"Error loading test set: {e}")
        print(
            "Note: Create 'test' folder in your dataset directory with 'images' and 'masks' subfolders"
        )
        return

    # 评估
    print("Evaluating on test set...")
    results, preds, targets = evaluate_model(model, test_loader, device, args.mode, cfg)

    # 保存结果
    model_name = f"{args.model}_{args.mode}" if args.model == "dr_unet" else args.model
    save_results(results, args.output, model_name, test_size)

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"Test Results:")
    print(f"  Dice: {results['dice']:.4f} ± {results['dice_std']:.4f}")
    print(f"  IoU:  {results['iou']:.4f} ± {results['iou_std']:.4f}")
    if "router_easy_ratio" in results:
        print(f"  Early Exit Rate: {results['router_easy_ratio'] * 100:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
