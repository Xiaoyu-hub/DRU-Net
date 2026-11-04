"""
评估指标模块
参考 RARE-UNet 的设计，适用于 DR²-UNet 项目
支持 Dice、IoU、Accuracy、Precision、Recall、F1
"""

import torch
import torch.nn.functional as F
from typing import Optional, List


def zero_convention(metric_tensor: torch.Tensor) -> torch.Tensor:
    """
    零约定：当分母为零时返回 1.0，否则返回 0.0
    """
    return torch.where(
        metric_tensor == 0,
        torch.tensor(1.0, device=metric_tensor.device),
        torch.tensor(0.0, device=metric_tensor.device),
    )


def compute_dice_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    计算单个类别的 Dice 分数

    Args:
        preds: 预测结果 [B, H, W] (类别索引)
        masks: 真实掩码 [B, H, W] (类别索引)
        c: 类别索引
        smooth: 平滑项

    Returns:
        Dice 分数
    """
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()

    intersection = (pred_c * mask_c).sum()
    union = pred_c.sum() + mask_c.sum()

    if union == 0:
        return zero_convention(intersection)

    return (2.0 * intersection + smooth) / (union + smooth)


def dice_coefficient(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算 Dice 系数（多类别平均）

    Args:
        preds: 预测结果 [B, H, W] (类别索引)
        masks: 真实掩码 [B, H, W] (类别索引)
        num_classes: 类别数量
        smooth: 平滑项
        ignore_index: 忽略的类别索引
        reduction: 归约方式 ('mean', 'none', 'sum')

    Returns:
        平均 Dice 分数
    """
    dice_scores = []

    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        dice = compute_dice_score(preds, masks, c, smooth)
        dice_scores.append(dice)

    if not dice_scores:
        return torch.tensor(0.0, device=preds.device)

    dice_scores = torch.stack(dice_scores)

    if reduction == "mean":
        return dice_scores.mean()
    elif reduction == "sum":
        return dice_scores.sum()
    else:
        return dice_scores


def compute_iou_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    计算单个类别的 IoU (Intersection over Union)

    Args:
        preds: 预测结果 [B, H, W]
        masks: 真实掩码 [B, H, W]
        c: 类别索引
        smooth: 平滑项

    Returns:
        IoU 分数
    """
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()

    intersection = (pred_c * mask_c).sum()
    union = pred_c.sum() + mask_c.sum() - intersection

    if union == 0:
        return zero_convention(intersection)

    return (intersection + smooth) / (union + smooth)


def iou_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算 IoU 分数（多类别平均）

    Args:
        preds: 预测结果 [B, H, W]
        masks: 真实掩码 [B, H, W]
        num_classes: 类别数量
        smooth: 平滑项
        ignore_index: 忽略的类别索引
        reduction: 归约方式

    Returns:
        平均 IoU 分数
    """
    iou_scores = []

    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        iou = compute_iou_score(preds, masks, c, smooth)
        iou_scores.append(iou)

    if not iou_scores:
        return torch.tensor(0.0, device=preds.device)

    iou_scores = torch.stack(iou_scores)

    if reduction == "mean":
        return iou_scores.mean()
    elif reduction == "sum":
        return iou_scores.sum()
    else:
        return iou_scores


def pixel_accuracy(
    preds: torch.Tensor,
    masks: torch.Tensor,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    计算像素准确率

    Args:
        preds: 预测结果 [B, H, W]
        masks: 真实掩码 [B, H, W]
        ignore_index: 忽略的类别索引

    Returns:
        像素准确率
    """
    if ignore_index is not None:
        valid_mask = masks != ignore_index
        preds = preds[valid_mask]
        masks = masks[valid_mask]

    correct = (preds == masks).sum()
    total = preds.numel()

    return correct.float() / total


def compute_precision_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    计算单个类别的精确率 (Precision)

    Args:
        preds: 预测结果 [B, H, W]
        masks: 真实掩码 [B, H, W]
        c: 类别索引
        smooth: 平滑项

    Returns:
        精确率
    """
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()

    tp = (pred_c * mask_c).sum()  # True Positive
    fp = (pred_c * (1 - mask_c)).sum()  # False Positive

    if tp + fp == 0:
        return zero_convention(tp)

    return (tp + smooth) / (tp + fp + smooth)


def precision_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算精确率（多类别平均）
    """
    precision_scores = []

    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        precision = compute_precision_score(preds, masks, c, smooth)
        precision_scores.append(precision)

    if not precision_scores:
        return torch.tensor(0.0, device=preds.device)

    precision_scores = torch.stack(precision_scores)

    if reduction == "mean":
        return precision_scores.mean()
    else:
        return precision_scores


def compute_recall_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    计算单个类别的召回率 (Recall)

    Args:
        preds: 预测结果 [B, H, W]
        masks: 真实掩码 [B, H, W]
        c: 类别索引
        smooth: 平滑项

    Returns:
        召回率
    """
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()

    tp = (pred_c * mask_c).sum()  # True Positive
    fn = ((1 - pred_c) * mask_c).sum()  # False Negative

    if tp + fn == 0:
        return zero_convention(tp)

    return (tp + smooth) / (tp + fn + smooth)


def recall_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算召回率（多类别平均）
    """
    recall_scores = []

    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        recall = compute_recall_score(preds, masks, c, smooth)
        recall_scores.append(recall)

    if not recall_scores:
        return torch.tensor(0.0, device=preds.device)

    recall_scores = torch.stack(recall_scores)

    if reduction == "mean":
        return recall_scores.mean()
    else:
        return recall_scores


def f1_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算 F1 分数（多类别平均）
    """
    precision = precision_score(
        preds, masks, num_classes, smooth, ignore_index, reduction="none"
    )
    recall = recall_score(
        preds, masks, num_classes, smooth, ignore_index, reduction="none"
    )

    f1 = 2 * precision * recall / (precision + recall + smooth)

    if reduction == "mean":
        return f1.mean()
    else:
        return f1


class MetricsCalculator:
    """
    评估指标计算器
    方便在验证过程中批量计算多个指标
    """

    def __init__(
        self, num_classes: int, ignore_index: Optional[int] = None, device: str = "cpu"
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device

        self.reset()

    def reset(self):
        """重置累计的指标"""
        self.total_dice = 0.0
        self.total_iou = 0.0
        self.total_acc = 0.0
        self.total_precision = 0.0
        self.total_recall = 0.0
        self.total_f1 = 0.0
        self.num_samples = 0

    def update(self, preds: torch.Tensor, masks: torch.Tensor):
        """
        更新指标累计值

        Args:
            preds: 预测结果 [B, H, W] (logits 或 类别索引)
            masks: 真实掩码 [B, H, W] (类别索引)
        """
        # 如果 preds 是 logits，转换为类别索引
        if preds.dim() == 4 and preds.shape[1] > 1:
            preds = preds.argmax(dim=1)

        # 确保在同一设备上
        preds = preds.to(self.device)
        masks = masks.to(self.device)

        batch_size = preds.shape[0]

        # 计算各项指标
        dice = dice_coefficient(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        iou = iou_score(preds, masks, self.num_classes, ignore_index=self.ignore_index)
        acc = pixel_accuracy(preds, masks, ignore_index=self.ignore_index)

        # 累计
        self.total_dice += dice.item() * batch_size
        self.total_iou += iou.item() * batch_size
        self.total_acc += acc.item() * batch_size
        self.num_samples += batch_size

    def compute(self) -> dict:
        """
        计算平均指标

        Returns:
            包含各项平均指标的字典
        """
        if self.num_samples == 0:
            return {
                "dice": 0.0,
                "iou": 0.0,
                "accuracy": 0.0,
            }

        return {
            "dice": self.total_dice / self.num_samples,
            "iou": self.total_iou / self.num_samples,
            "accuracy": self.total_acc / self.num_samples,
        }
