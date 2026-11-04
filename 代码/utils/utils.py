"""
核心工具函数模块
参考 RARE-UNet + LB-UNet 的设计，适用于 DR²-UNet 项目
"""

import os
import random
import logging
import logging.handlers
import functools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np


def set_seed(seed: int):
    """
    设置所有随机种子，确保实验可复现

    Args:
        seed: 随机种子整数
    """
    if seed is None:
        return

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python and numpy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU/GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn 确定性模式
    cudnn.benchmark = False
    cudnn.deterministic = True


def setup_environment(cfg):
    """
    设置训练环境：随机种子 + cudnn 模式

    Args:
        cfg: Hydra 配置对象，需要包含:
            - cfg.seed: 随机种子
            - cfg.optimization.mode: 优化模式 ("deterministic" / "benchmark")
    """
    # 设置随机种子
    seed = getattr(cfg, "seed", None)
    set_seed(seed)

    # 根据配置设置 cudnn 模式
    mode = (
        getattr(cfg.optimization, "mode", "benchmark")
        if hasattr(cfg, "optimization")
        else "benchmark"
    )

    if mode == "deterministic":
        # 确定性模式，确保可复现（可能有性能损失）
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif mode == "benchmark":
        # Benchmark 模式，自动选择最优算法（提高性能）
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_logger(name: str, log_dir: str, level: int = logging.INFO):
    """
    创建日志记录器，同时输出到文件和控制台

    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        level: 日志级别 (默认 INFO)

    Returns:
        logger: 配置好的日志记录器
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复输出

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 文件 handler（按天轮转）
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when="D", encoding="utf-8"
    )
    file_handler.setLevel(level)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_config_info(cfg, logger):
    """
    将配置信息记录到日志

    Args:
        cfg: Hydra 配置对象
        logger: 日志记录器
    """
    logger.info("#" * 50)
    logger.info("# Config Information")
    logger.info("#" * 50)

    # 递归获取配置信息
    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # 处理 OmegaConf DictConfig
    if hasattr(cfg, "to_dict"):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = cfg

    if isinstance(cfg_dict, dict):
        for key, value in flatten_dict(cfg_dict).items():
            logger.info(f"{key}: {value}")
    else:
        # 简单对象
        for key in dir(cfg):
            if not key.startswith("_"):
                try:
                    value = getattr(cfg, key)
                    if not callable(value):
                        logger.info(f"{key}: {value}")
                except:
                    pass

    logger.info("#" * 50)


def get_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """
    优化器工厂函数

    Args:
        model: PyTorch 模型
        cfg: 训练配置对象，需要包含:
            - cfg.optimizer.name: 优化器名称
            - cfg.learning_rate: 学习率
            - cfg.weight_decay: 权重衰减

    Returns:
        optimizer: PyTorch 优化器实例
    """
    optimizer_name = cfg.optimizer.name
    lr = cfg.learning_rate
    weight_decay = cfg.weight_decay

    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, eps=1e-8
        )
    elif optimizer_name == "Adadelta":
        return torch.optim.Adadelta(
            model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
        )
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(
            model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
        )
    else:
        # 默认使用 AdamW
        logger = logging.getLogger(__name__)
        logger.warning(f"Unknown optimizer {optimizer_name}, using AdamW instead")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer: torch.optim.Optimizer, cfg):
    """
    学习率调度器工厂函数

    Args:
        optimizer: PyTorch 优化器
        cfg: 训练配置对象，需要包含:
            - cfg.scheduler.name: 调度器名称
            - cfg.scheduler.params: 调度器参数

    Returns:
        scheduler: 学习率调度器实例
    """
    scheduler_name = cfg.scheduler.name
    params = cfg.scheduler.params

    if scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.T_max,
            eta_min=params.eta_min,
            last_epoch=params.last_epoch,
        )
    elif scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params.mode,
            factor=params.factor,
            patience=params.patience,
            threshold=params.threshold,
            cooldown=params.cooldown,
            min_lr=params.min_lr,
            eps=params.eps,
        )
    elif scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.step_size,
            gamma=params.gamma,
            last_epoch=params.last_epoch,
        )
    elif scheduler_name == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=params.milestones,
            gamma=params.gamma,
            last_epoch=params.last_epoch,
        )
    elif scheduler_name == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=params.gamma, last_epoch=params.last_epoch
        )
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.T_0,
            T_mult=params.T_mult,
            eta_min=params.eta_min,
            last_epoch=params.last_epoch,
        )
    else:
        # 默认不使用调度器
        logger = logging.getLogger(__name__)
        logger.warning(f"Unknown scheduler {scheduler_name}, returning None")
        return None


def resize_masks_to(maps: torch.Tensor, gt_maps: torch.Tensor) -> torch.Tensor:
    """
    将 GT 掩码 resize 到与特征图相同的空间尺寸

    Args:
        maps: 特征图 [B, C, H, W] 或 [B, C, D, H, W]
        gt_maps: GT 掩码 [B, H, W] 或 [B, D, H, W] 或带批次维度的形式

    Returns:
        resized: Resize 后的掩码
    """
    # 处理维度
    if maps.ndim == 4:  # 2D: [B, C, H, W]
        if gt_maps.ndim == 2:  # [B, H, W]
            gt_maps = gt_maps.unsqueeze(1)  # [B, 1, H, W]
        elif gt_maps.ndim == 3 and gt_maps.shape[1] not in [
            1,
            maps.shape[1],
        ]:  # [B, H, W] but not channel
            gt_maps = gt_maps.unsqueeze(1)

        resized = F.interpolate(gt_maps.float(), size=maps.shape[2:], mode="nearest")
        return resized.squeeze(1).long()

    elif maps.ndim == 5:  # 3D: [B, C, D, H, W]
        if gt_maps.ndim == 4:  # [B, D, H, W]
            gt_maps = gt_maps.unsqueeze(1)  # [B, 1, D, H, W]

        resized = F.interpolate(gt_maps.float(), size=maps.shape[2:], mode="nearest")
        return resized.squeeze(1).long()

    else:
        raise ValueError(f"Expected maps to have 4 or 5 dimensions, got {maps.ndim}")


def count_parameters(model: nn.Module) -> int:
    """
    计算模型的可训练参数数量

    Args:
        model: PyTorch 模型

    Returns:
        count: 可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(gpu_devices: list = None) -> torch.device:
    """
    获取计算设备

    Args:
        gpu_devices: GPU 设备列表，默认使用第一个可用的 GPU

    Returns:
        device: torch.device
    """
    if gpu_devices is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_devices[0]}")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def ema_decorator(ema_decay: float = 0.999):
    """
    EMA (Exponential Moving Average) 装饰器

    Args:
        ema_decay: EMA 衰减率

    Returns:
        装饰器函数
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            # 如果模型有 ema 参数
            if hasattr(self, "ema_model") and self.ema_model is not None:
                with torch.no_grad():
                    for ema_param, param in zip(
                        self.ema_model.parameters(), self.parameters()
                    ):
                        ema_param.data.mul_(ema_decay).add_(
                            param.data, alpha=1 - ema_decay
                        )

            return result

        return wrapper

    return decorator


# 尝试导入 omegaconf（用于配置处理）
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None
