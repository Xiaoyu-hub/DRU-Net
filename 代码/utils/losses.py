"""
损失函数模块
参考 RARE-UNet + LB-UNet 的设计，适用于 DR²-UNet 项目
支持 Dice、BCE、组合损失及路由损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def zero_convention(metric_tensor: torch.Tensor) -> torch.Tensor:
    """
    零约定：当分母为零时返回 1.0，否则返回 0.0
    用于避免除零错误
    """
    return torch.where(
        metric_tensor == 0,
        torch.tensor(1.0, device=metric_tensor.device),
        torch.tensor(0.0, device=metric_tensor.device),
    )


class DiceLoss(nn.Module):
    """
    Dice 损失函数
    适用于医学图像分割，特别是类别不平衡情况
    """

    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 [B, C, H, W] (logits)
            target: 目标值 [B, H, W] (类别索引)
        """
        # 确保 target 是整数类型
        if target.dtype != torch.long:
            target = target.long()

        # 确保 target 值在有效范围内 [0, num_classes-1]
        num_classes = pred.shape[1]
        target = torch.clamp(target, min=0, max=num_classes - 1)

        # 将预测值通过 softmax 转换为概率
        pred = torch.softmax(pred, dim=1)

        # 将目标转换为 one-hot 编码
        target_one_hot = (
            F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        )

        # 计算 Dice
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == "mean":
            return 1 - dice.mean()
        elif self.reduction == "none":
            return 1 - dice
        else:
            return 1 - dice.sum()


class BCEWithLogitsLoss(nn.Module):
    """
    二元交叉熵损失函数
    适用于二分类分割任务
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 [B, 1, H, W] 或 [B, H, W] (logits)
            target: 目标值 [B, 1, H, W] 或 [B, H, W] (0 或 1)
        """
        # 确保维度一致
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)

        return F.binary_cross_entropy_with_logits(
            pred, target.float(), pos_weight=self.pos_weight
        )


class DiceBCELoss(nn.Module):
    """
    Dice + BCE 组合损失函数
    结合两种损失函数的优点
    """

    def __init__(
        self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 [B, C, H, W] (logits)
            target: 目标值 [B, H, W] (类别索引)
        """
        dice = self.dice_loss(pred, target)

        # 处理不同通道数的预测
        # 二分类 (C=1): 使用 sigmoid + BCE
        # 多分类 (C>1): 使用第 2 通道开始的预测
        if pred.shape[1] == 1:
            # 二分类：直接 squeeze 并使用 BCE
            bce = self.bce_loss(pred.squeeze(1), (target > 0).float())
        else:
            # 多分类：取第 2 通道（前景类）
            bce = self.bce_loss(pred[:, 1:, :, :], (target > 0).float())

        return self.dice_weight * dice + self.bce_weight * bce


class CombinedLoss(nn.Module):
    """
    组合损失函数：CrossEntropy + Dice
    参考 RARE-UNet 的实现
    """

    def __init__(self, alpha: float = 0.5, ignore_index: Optional[int] = None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def soft_dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算软 Dice 损失"""
        # 确保 target 是整数类型
        if target.dtype != torch.long:
            target = target.long()

        # 确保 target 值在有效范围内
        num_classes = pred.shape[1]
        target = torch.clamp(target, min=0, max=num_classes - 1)

        # 转换为 one-hot
        target_one_hot = (
            F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        )

        # Softmax
        pred = torch.softmax(pred, dim=1)

        # 计算每个类别的 Dice
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)

        return 1 - dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 [B, C, H, W] (logits)
            target: 目标值 [B, H, W] (类别索引)
        """
        ce_loss = self.ce(pred, target)
        dice_loss = self.soft_dice_loss(pred, target)

        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


class RouterLoss(nn.Module):
    """
    路由器损失函数
    用于训练 DR²-UNet 中的难度路由器
    判断样本应该走轻分支还是重分支
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, router_logits: torch.Tensor, is_hard_sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            router_logits: 路由器输出 [B, 1] (logits)
            is_hard_sample: 目标标签 [B, 1] (1=难样本, 0=易样本)
        """
        return self.bce(router_logits, is_hard_sample.float())


class ConsistencyLoss(nn.Module):
    """
    一致性损失函数
    用于 DR²-UNet 中轻分支和重分支输出的对齐
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self, light_output: torch.Tensor, heavy_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            light_output: 轻分支输出 [B, C, H, W]
            heavy_output: 重分支输出 [B, C, H, W]
        """
        light_prob = torch.softmax(light_output, dim=1)
        heavy_prob = torch.softmax(heavy_output, dim=1)

        if self.loss_type == "mse":
            return F.mse_loss(light_prob, heavy_prob)
        elif self.loss_type == "kl":
            return F.kl_div(light_prob.log(), heavy_prob, reduction="batchmean")
        else:
            return F.mse_loss(light_prob, heavy_prob)


class AuxiliaryLoss(nn.Module):
    """
    辅助损失函数
    用于 DR²-UNet 中多尺度输出的监督
    """

    def __init__(self, loss_fn: nn.Module, weight: float = 0.3):
        super().__init__()
        self.loss_fn = loss_fn
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 辅助分支输出 [B, C, H, W]
            target: 目标值 [B, H, W]
        """
        loss = self.loss_fn(pred, target)
        return self.weight * loss


def build_loss(loss_config) -> nn.Module:
    """
    损失函数工厂函数

    Args:
        loss_config: 损失函数配置，包含:
            - name: 损失函数名称
            - params: 损失函数参数

    Returns:
        损失函数实例
    """
    loss_name = loss_config.name.lower().replace(" ", "").replace("-", "_")
    params = loss_config.params if hasattr(loss_config, "params") else {}

    if loss_name == "dice":
        return DiceLoss(**params)
    elif loss_name == "bce" or loss_name == "bcewithlogits":
        return BCEWithLogitsLoss(**params)
    elif loss_name in ["dicebce", "dice_bce", "dicebceloss"]:
        return DiceBCELoss(**params)
    elif loss_name in ["combined", "combinedloss", "ce_dice"]:
        return CombinedLoss(**params)
    elif loss_name == "router":
        return RouterLoss()
    elif loss_name == "consistency":
        return ConsistencyLoss(**params)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
