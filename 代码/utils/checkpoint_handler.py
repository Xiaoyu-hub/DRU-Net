"""
模型检查点管理模块
参考 RARE-UNet 的设计
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointHandler:
    """
    检查点管理器

    负责保存和加载模型检查点
    """

    def __init__(
        self,
        base_dir: str = "trained_models",
        model_name: str = "model",
    ):
        """
        Args:
            base_dir: 模型保存根目录
            model_name: 模型名称
        """
        self.base_dir = Path(base_dir)
        self.model_name = model_name

        # 创建目录
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 最佳/最新的检查点路径
        self.best_path = self.base_dir / f"{model_name}_best.pth"
        self.last_path = self.base_dir / f"{model_name}_last.pth"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        best_metric_val: float = 0.0,
        is_best: bool = False,
    ) -> str:
        """
        保存检查点

        Args:
            model: PyTorch 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前 epoch
            best_metric_val: 最佳指标值
            is_best: 是否是最佳模型

        Returns:
            保存路径
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric_val,
        }

        # 添加优化器状态
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # 添加调度器状态
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # 保存最新检查点
        torch.save(checkpoint, self.last_path)
        logger.info(f"Saved last checkpoint: {self.last_path}")

        # 如果是最佳模型，保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.best_path)
            logger.info(f"Saved best checkpoint: {self.best_path}")

        return str(self.last_path)

    def load(
        self,
        checkpoint_path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
            model: PyTorch 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备

        Returns:
            检查点字典
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 加载模型权重
        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model state from {checkpoint_path}")

        # 加载优化器状态
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded optimizer state")

        # 加载调度器状态
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Loaded scheduler state")

        return checkpoint

    def load_best(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        加载最佳检查点

        Args:
            model: PyTorch 模型
            device: 设备

        Returns:
            检查点字典
        """
        return self.load(self.best_path, model=model, device=device)

    def load_last(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        加载最新检查点

        Args:
            model: PyTorch 模型
            device: 设备

        Returns:
            检查点字典
        """
        return self.load(self.last_path, model=model, device=device)
