"""
DR²-UNet 训练器模块
支持多损失训练（轻分支 + 重分支 + 路由 + 一致性）
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from trainers.trainer import BaseTrainer
from utils.metrics import dice_coefficient, iou_score, pixel_accuracy
from utils.utils import resize_masks_to

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


class DRTrainer(BaseTrainer):
    """
    DR²-UNet 训练器

    支持：
    - 轻分支损失
    - 重分支损失
    - 路由器损失
    - 一致性损失
    - 动态权重更新
    """

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        train_dataloader: Any,
        val_dataloader: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_handler: Optional[Any] = None,
    ):
        # 继承 BaseTrainer
        super().__init__(
            cfg,
            model,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            scheduler,
            device,
            checkpoint_handler,
        )

        # DR²-UNet 特有参数
        arch_cfg = cfg.architecture

        # 损失权重
        self.light_weight = (
            arch_cfg.loss_weights.light_weight
            if hasattr(arch_cfg, "loss_weights")
            else 1.0
        )
        self.heavy_weight = (
            arch_cfg.loss_weights.heavy_weight
            if hasattr(arch_cfg, "loss_weights")
            else 1.5
        )
        self.router_weight = (
            arch_cfg.loss_weights.router_weight
            if hasattr(arch_cfg, "loss_weights")
            else 0.5
        )
        self.consistency_weight = (
            arch_cfg.loss_weights.consistency_weight
            if hasattr(arch_cfg, "loss_weights")
            else 0.3
        )

        # Router 损失函数
        self.router_criterion = nn.BCEWithLogitsLoss()

        # 是否使用动态权重
        self.use_dynamic_weights = True

        # 记录每个损失的均值（用于动态权重）
        self.loss_history = {"light": [], "heavy": [], "router": [], "consistency": []}

        logger.info(
            f"DRTrainer initialized: light={self.light_weight}, "
            f"heavy={self.heavy_weight}, router={self.router_weight}, "
            f"consistency={self.consistency_weight}"
        )

    def _compute_loss(
        self, outputs: Dict[str, torch.Tensor], masks: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 DR²-UNet 的总损失
        """
        total_loss = 0.0
        loss_dict = {}

        # 1. 轻分支损失
        if "light_output" in outputs and outputs["light_output"] is not None:
            light_loss = self.criterion(outputs["light_output"], masks)
            total_loss += self.light_weight * light_loss
            loss_dict["light"] = light_loss.item()

        # 2. 重分支损失（单输出）
        if "heavy_output" in outputs and outputs["heavy_output"] is not None:
            heavy_loss = self.criterion(outputs["heavy_output"], masks)
            total_loss += self.heavy_weight * heavy_loss
            loss_dict["heavy"] = heavy_loss.item()

        # 3. 路由器损失
        if "router_logits" in outputs and outputs["router_logits"] is not None:
            if "light_output" in outputs and "heavy_output" in outputs:
                if outputs["light_output"] is not None and outputs["heavy_output"] is not None:
                    router_targets = self._compute_router_targets(
                        outputs["light_output"], outputs["heavy_output"], masks
                    )
                    router_loss = self.router_criterion(
                        outputs["router_logits"], router_targets
                    )
                    total_loss += self.router_weight * router_loss
                    loss_dict["router"] = router_loss.item()

        # 4. 轻/重分支输出一致性损失
        if "light_output" in outputs and "heavy_output" in outputs:
            if (
                outputs["light_output"] is not None
                and outputs["heavy_output"] is not None
            ):
                consistency_loss = self._compute_consistency_loss(
                    outputs["light_output"], outputs["heavy_output"]
                )
                total_loss += self.consistency_weight * consistency_loss
                loss_dict["consistency"] = consistency_loss.item()

        # 记录损失历史
        for key, value in loss_dict.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)

        return total_loss, loss_dict

    def _compute_router_targets(
        self, 
        light_output: torch.Tensor, 
        heavy_output: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        根据轻分支和重分支输出的差异计算 Router 的目标标签
        
        差异小（简单样本）-> is_hard=0（早退）
        差异大（难样本）-> is_hard=1（走重分支）
        
        Args:
            light_output: 轻分支输出 [B, 1, H, W]
            heavy_output: 重分支输出 [B, 1, H, W]
            masks: 目标掩码 [B, H, W]
            
        Returns:
            is_hard: 难度标签 [B, 1]
        """
        light_prob = torch.sigmoid(light_output)
        heavy_prob = torch.sigmoid(heavy_output)
        
        diff = torch.abs(light_prob - heavy_prob)
        avg_diff = diff.mean(dim=[1, 2, 3])
        
        threshold = 0.1
        is_hard = (avg_diff > threshold).float().unsqueeze(1)
        
        return is_hard

    def _compute_consistency_loss(
        self, light_output: torch.Tensor, heavy_output: torch.Tensor
    ) -> torch.Tensor:
        """
        计算一致性损失（轻分支和重分支输出的对齐）
        二分类（out_channels=1）用 sigmoid，多分类用 softmax。
        """
        if self.num_classes == 1:
            light_prob = torch.sigmoid(light_output)
            heavy_prob = torch.sigmoid(heavy_output)
        else:
            light_prob = torch.softmax(light_output, dim=1)
            heavy_prob = torch.softmax(heavy_output, dim=1)

        # MSE 损失
        return F.mse_loss(light_prob, heavy_prob)

    def _compute_metrics(
        self, outputs: Dict[str, torch.Tensor], masks: torch.Tensor
    ) -> Dict[str, Any]:
        """
        计算评估指标

        使用 final_output 计算指标
        """
        # 获取最终输出
        if "final_output" in outputs:
            final_output = outputs["final_output"]
        elif "heavy_output" in outputs:
            final_output = outputs["heavy_output"]
        else:
            final_output = outputs["light_output"]

        # 获取预测 - 使用新的辅助函数处理单通道输出
        preds = _get_predictions(final_output, self.num_classes)

        # 计算指标
        dice = dice_coefficient(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        ).item()

        iou = iou_score(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        ).item()

        acc = pixel_accuracy(preds, masks).item()

        return {
            "dice": dice,
            "iou": iou,
            "accuracy": acc,
        }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个 epoch
        """
        self.model.train()

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_acc = 0.0
        total_hard_ratio = 0.0
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
                    loss, loss_dict = self._compute_loss(outputs, masks)

                # 反向传播（AMP）
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss, loss_dict = self._compute_loss(outputs, masks)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            # 计算指标
            with torch.no_grad():
                metrics = self._compute_metrics(outputs, masks)
                
                # 收集额外统计信息
                batch_size = images.shape[0]
                
                # is_hard 统计
                if "is_hard" in outputs and outputs["is_hard"] is not None:
                    is_hard_count = outputs["is_hard"].sum().item()
                    hard_ratio = is_hard_count / batch_size
                else:
                    hard_ratio = 0.0
            
            # 累计
            total_loss += loss.item()
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            total_acc += metrics["accuracy"]
            total_hard_ratio += hard_ratio
            num_batches += 1
            
            # 记录每个batch的损失组成
            for key, value in loss_dict.items():
                if f"loss_{key}" not in self.loss_history:
                    self.loss_history[f"loss_{key}"] = []
                self.loss_history[f"loss_{key}"].append(value)

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{metrics['dice']:.4f}",
                    "hard": f"{hard_ratio:.2f}",
                }
            )

        # 平均指标
        avg_metrics = {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
            "accuracy": total_acc / num_batches,
            "hard_ratio": total_hard_ratio / num_batches,
        }
        
        # 详细损失组件日志（最后10个batch的均值）
        avg_loss_components = {}
        for key in self.loss_history:
            if key.startswith("loss_") and len(self.loss_history[key]) > 0:
                avg_loss_components[key] = sum(self.loss_history[key][-10:]) / min(10, len(self.loss_history[key]))
        
        # 路由器决策统计
        hard_ratio = avg_metrics['hard_ratio']
        early_exit_rate = 1.0 - hard_ratio
        logger.info(
            f"[Training] Epoch {epoch + 1} - "
            f"Loss: {avg_metrics['loss']:.4f}, "
            f"Dice: {avg_metrics['dice']:.4f}, "
            f"Hard Ratio: {hard_ratio:.2%} (Early Exit: {early_exit_rate:.2%}), "
            f"Light: {avg_loss_components.get('loss_light', 0):.4f}, "
            f"Heavy: {avg_loss_components.get('loss_heavy', 0):.4f}, "
            f"Router: {avg_loss_components.get('loss_router', 0):.4f}, "
            f"Consistency: {avg_loss_components.get('loss_consistency', 0):.4f}"
        )

        # 动态权重更新（可选）
        if self.use_dynamic_weights and len(self.loss_history["light"]) > 10:
            self._update_dynamic_weights()

        return avg_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        验证
        """
        self.model.eval()

        total_loss = 0.0

        total_hard_ratio = 0.0
        num_hard_samples = 0
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
                        loss, _ = self._compute_loss(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss, _ = self._compute_loss(outputs, masks)

                # 计算指标
                metrics = self._compute_metrics(outputs, masks)

                # 累计
                total_loss += loss.item()
                total_dice += metrics["dice"]
                total_iou += metrics["iou"]
                total_acc += metrics["accuracy"]
                num_batches += 1

                # 计算hard_ratio（用于进度条）
                if "is_hard" in outputs and outputs["is_hard"] is not None:
                    is_hard_count = outputs["is_hard"].sum().item()
                    hard_ratio = is_hard_count / images.shape[0]
                else:
                    hard_ratio = 0.0

                # 更新进度条
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "dice": f"{metrics['dice']:.4f}",
                        "hard": f"{hard_ratio:.2f}",
                    }
                )

        # 平均指标
        avg_metrics = {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
            "accuracy": total_acc / num_batches,
            "hard_ratio": total_hard_ratio / num_batches,
        }

        # 路由器统计日志
        logger.info(
            f"[Validation] Epoch {epoch + 1} - "
            f"Loss: {avg_metrics['loss']:.4f}, "
            f"Dice: {avg_metrics['dice']:.4f}, "
            f"Hard Ratio: {avg_metrics['hard_ratio']:.2%}"
        )

        return avg_metrics

    def _update_dynamic_weights(self):
        """
        动态权重更新（参考 DWA 思想）

        根据损失的变化趋势调整权重
        """
        if len(self.loss_history["light"]) < 2:
            return

        # 计算每个损失的近期均值
        recent_losses = {}
        for key in self.loss_history:
            if len(self.loss_history[key]) > 0:
                recent_losses[key] = sum(self.loss_history[key][-10:]) / min(
                    10, len(self.loss_history[key])
                )

        # 简单策略：根据损失大小调整权重（损失大的权重小）
        total_loss = sum(recent_losses.values())
        if total_loss > 0:
            for key in recent_losses:
                if key == "light":
                    self.light_weight = total_loss / (recent_losses[key] + 1e-8)
                elif key == "heavy":
                    self.heavy_weight = total_loss / (recent_losses[key] + 1e-8)


def build_dr_trainer(
    cfg: Any,
    model: nn.Module,
    train_dataloader: Any,
    val_dataloader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
) -> DRTrainer:
    """
    DR²-UNet 训练器工厂函数

    Args:
        cfg: 配置对象
        model: PyTorch 模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器

    Returns:
        DRTrainer 实例
    """
    from utils.checkpoint_handler import CheckpointHandler

    # 创建检查点管理器
    checkpoint_handler = CheckpointHandler(
        base_dir=cfg.trained_models.base_dir,
        model_name=cfg.architecture.name,
    )

    # 创建训练器
    trainer = DRTrainer(
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
