"""
基础训练器模块
参考 RARE-UNet 的设计，适用于 DR²-UNet 项目
支持训练循环、验证、模型保存、早停等功能
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.metrics import dice_coefficient, iou_score, pixel_accuracy
from utils.checkpoint_handler import CheckpointHandler

logger = logging.getLogger(__name__)


def _get_predictions(output: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    从模型输出获取预测类别索引

    Args:
        output: 模型输出 [B, C, H, W]
        num_classes: 输出通道数 (out_channels)

    Returns:
        preds: 预测类别索引 [B, H, W]
    """
    # 使用 out_channels 判断而不是 num_classes
    # 二分类任务: out_channels=1, 使用 sigmoid + 阈值
    # 多分类任务: out_channels>=2, 使用 argmax
    if num_classes == 1:
        # 二分类：使用 sigmoid + 阈值
        prob = torch.sigmoid(output.squeeze(1))
        return (prob > 0.5).long()
    else:
        # 多分类：使用 argmax
        return output.argmax(dim=1)


class BaseTrainer:
    """
    基础训练器类

    支持：
    - 训练循环
    - 验证
    - 模型保存（best/latest）
    - 早停
    - AMP 混合精度
    - 断点续训
    """

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_handler: Optional[CheckpointHandler] = None,
    ):
        """
        Args:
            cfg: 配置对象
            model: PyTorch 模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 计算设备
            checkpoint_handler: 检查点管理器
        """
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # 检查点管理
        self.ckpt_handler = checkpoint_handler

        # 训练参数
        self.num_epochs = cfg.training.num_epochs
        # 使用 out_channels 判断而不是 num_classes
        # out_channels 才是真正决定模型输出形式的参数
        self.num_classes = cfg.architecture.out_channels
        self.ignore_index = cfg.dataset.get("ignore_index", None)

        # 梯度裁剪
        self.max_norm = cfg.training.get("grad_clip_norm", 1.0)

        # AMP 混合精度
        self.use_amp = cfg.get("use_amp", False) and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using AMP mixed precision training")

        # 早停
        self.early_stopper = None
        early_cfg = cfg.training.get("early_stopper", None)
        if early_cfg and early_cfg.get("enabled", False):
            from trainers.early_stopping import EarlyStopping

            self.early_stopper = EarlyStopping(
                patience=early_cfg.get("patience", 50),
                delta=early_cfg.get("delta", 1e-4),
                criterion=early_cfg.get("criterion", "dice"),
                verbose=early_cfg.get("verbose", True),
            )
            logger.info(f"Early stopping enabled: patience={early_cfg.get('patience')}")

        # 训练状态
        self.start_epoch = 0
        self.best_metric = (
            0.0 if self.early_stopper is None else self.early_stopper.best_score
        )
        self.current_epoch = 0

        # 断点续训
        resume_checkpoint = cfg.get("resume_checkpoint", None)
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            self._resume_from_checkpoint(resume_checkpoint)

        logger.info(f"BaseTrainer initialized for {self.num_epochs} epochs")

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """从检查点恢复训练"""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        if self.ckpt_handler is not None:
            checkpoint = self.ckpt_handler.load(checkpoint_path, model=self.model)
            self.start_epoch = checkpoint.get("epoch", 0)
            self.best_metric = checkpoint.get("best_metric", 0.0)

            # 恢复优化器和调度器
            if "optimizer" in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            logger.info(
                f"Resumed from epoch {self.start_epoch}, best metric: {self.best_metric:.4f}"
            )

    def train(self):
        """主训练循环"""
        start_time = time.time()

        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch

            # 训练一个 epoch
            epoch_start = time.time()
            train_metrics = self.train_one_epoch(epoch)
            train_time = time.time() - epoch_start

            # 验证
            val_start = time.time()
            val_metrics = self.validate(epoch)
            val_time = time.time() - val_start

            # 学习率调度
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    # ReduceLROnPlateau 需要传入指标
                    if hasattr(self.scheduler, "metric"):
                        self.scheduler.step(
                            val_metrics.get(
                                self.scheduler.metric, val_metrics.get("dice", 0)
                            )
                        )
                    else:
                        self.scheduler.step()

            # 记录学习率
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_metrics["learning_rate"] = current_lr

            # 打印训练信息
            self._log_epoch(epoch, train_metrics, val_metrics, train_time, val_time)

            # 保存检查点
            self._save_checkpoint(epoch, val_metrics)

            # 早停检查
            if self.early_stopper is not None:
                stop_early, improved = self.early_stopper(
                    val_metrics.get("loss", 0),
                    val_metrics.get(self.early_stopper.criterion, 0),
                )

                if stop_early:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.2f} minutes")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch 编号

        Returns:
            训练指标字典
        """
        self.model.train()

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_acc = 0.0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1} [Train]")

        for batch in pbar:
            images, masks = batch
            
            # 移动到设备
            images = images.to(self.device)
            masks = masks.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad(set_to_none=True)

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # 反向传播（AMP）
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            # 计算指标
            with torch.no_grad():
                preds = _get_predictions(outputs, self.num_classes)
                batch_dice = dice_coefficient(
                    preds, masks, self.num_classes, ignore_index=self.ignore_index
                ).item()
                batch_iou = iou_score(
                    preds, masks, self.num_classes, ignore_index=self.ignore_index
                ).item()
                batch_acc = pixel_accuracy(preds, masks).item()

            # 累计
            total_loss += loss.item()
            total_dice += batch_dice
            total_iou += batch_iou
            total_acc += batch_acc
            num_batches += 1

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{batch_dice:.4f}",
                }
            )

        # 平均指标
        avg_metrics = {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
            "accuracy": total_acc / num_batches,
        }

        return avg_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        验证

        Args:
            epoch: 当前 epoch 编号

        Returns:
            验证指标字典
        """
        self.model.eval()

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_acc = 0.0
        num_batches = 0

        pbar = tqdm(self.val_dataloader, desc=f"Epoch {epoch + 1} [Val]")

        with torch.no_grad():
            for images, masks in pbar:
                # 移动到设备
                images = images.to(self.device)
                masks = masks.to(self.device)

                # 前向传播
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # 计算指标
                preds = _get_predictions(outputs, self.num_classes)
                batch_dice = dice_coefficient(
                    preds, masks, self.num_classes, ignore_index=self.ignore_index
                ).item()
                batch_iou = iou_score(
                    preds, masks, self.num_classes, ignore_index=self.ignore_index
                ).item()
                batch_acc = pixel_accuracy(preds, masks).item()

                # 累计
                total_loss += loss.item()
                total_dice += batch_dice
                total_iou += batch_iou
                total_acc += batch_acc
                num_batches += 1

                # 更新进度条
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "dice": f"{batch_dice:.4f}",
                    }
                )

        # 平均指标
        avg_metrics = {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
            "accuracy": total_acc / num_batches,
        }

        return avg_metrics

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        train_time: float,
        val_time: float,
    ):
        """打印 epoch 训练信息"""
        logger.info("=" * 50)
        logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
        logger.info(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"Dice: {train_metrics['dice']:.4f}, "
            f"IoU: {train_metrics['iou']:.4f}, "
            f"Time: {train_time:.2f}s"
        )
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Dice: {val_metrics['dice']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}, "
            f"Time: {val_time:.2f}s"
        )
        logger.info(f"  LR: {train_metrics['learning_rate']:.6f}")
        logger.info("=" * 50)

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        """保存检查点"""
        if self.ckpt_handler is None:
            return

        # 判断是否是最佳模型
        is_best = val_metrics.get("dice", 0) > self.best_metric
        if is_best:
            self.best_metric = val_metrics.get("dice", 0)

        # 保存检查点
        self.ckpt_handler.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch + 1,
            best_metric_val=self.best_metric,
            is_best=is_best,
        )


def build_trainer(
    cfg: Any,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
) -> BaseTrainer:
    """
    训练器工厂函数

    Args:
        cfg: 配置对象
        model: PyTorch 模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器

    Returns:
        训练器实例
    """
    # 创建检查点管理器
    checkpoint_handler = CheckpointHandler(
        base_dir=cfg.trained_models.base_dir,
        model_name=cfg.architecture.name,
    )

    # 创建训练器
    trainer = BaseTrainer(
        cfg=cfg,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_handler=checkpoint_handler,
    )

    return trainer
