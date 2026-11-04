"""
DR²-UNet 训练入口
参考 RARE-UNet 的设计
"""

import os
import sys
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.utils import setup_environment, get_logger
from utils.losses import build_loss
from utils.checkpoint_handler import CheckpointHandler
from data.datasets import build_dataset
from models.dr_unet import build_dr_unet
from models.unet import build_unet
from trainers.dr_trainer import build_dr_trainer
from trainers.trainer import build_trainer


def build_dataloaders(cfg: DictConfig):
    """
    构建训练和验证数据加载器

    Args:
        cfg: Hydra 配置对象

    Returns:
        train_loader, val_loader
    """
    # 构建训练数据集
    train_dataset = build_dataset(cfg, phase="train")
    val_dataset = build_dataset(cfg, phase="val")

    # 获取训练参数
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers
    pin_memory = cfg.training.get("pin_memory", True)
    persistent_workers = (
        cfg.training.get("persistent_workers", True) if num_workers > 0 else False
    )
    prefetch_factor = (
        cfg.training.get("prefetch_factor", 2) if num_workers > 0 else None
    )

    # 构建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )

    return train_loader, val_loader


def setup_logging(cfg: DictConfig, log_dir: Path):
    """
    设置日志

    Args:
        cfg: 配置对象
        log_dir: 日志目录
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger("dr_unet")
    logger.setLevel(logging.INFO)

    # 文件 handler
    log_file = log_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    DR²-UNet 训练入口
    """
    # ===== 1. 环境设置 =====
    setup_environment(cfg)

    # ===== 2. 创建实验目录 =====
    exp_name = f"{cfg.architecture.name}_{cfg.dataset.name}_{OmegaConf.create({'now': '%Y%m%d_%H%M%S'})}"
    exp_dir = Path(cfg.trained_models.base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ===== 3. 日志设置 =====
    log_dir = exp_dir / "logs"
    logger = setup_logging(cfg, log_dir)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ===== 3.5 WandB 设置 =====
    if cfg.wandb.get("log", False):
        import wandb
        import json

        wandb_config = {
            "architecture": cfg.architecture.name,
            "dataset": cfg.dataset.name,
            "num_epochs": cfg.training.num_epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "optimizer": cfg.training.optimizer.name,
        }
        wandb.init(
            project=cfg.wandb.wandb_project,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            tags=cfg.wandb.get("tags", []),
            config=wandb_config,
        )
        logger.info(f"WandB initialized: {cfg.wandb.name}")

    # ===== 4. GPU 设置 =====
    gpu_cfg = cfg.gpu
    if gpu_cfg.mode == "single":
        device_id = gpu_cfg.devices[0]
        device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using single GPU: {device}")
    else:
        # 多GPU模式（暂时使用单GPU）
        device_id = gpu_cfg.devices[0]
        device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )
        logger.warning("Multi-GPU mode not fully implemented, using single GPU")

    # ===== 5. 数据加载 =====
    logger.info("Building datasets...")
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )
    logger.info(
        f"Batch size: {cfg.training.batch_size}, Epochs: {cfg.training.num_epochs}"
    )

    # ===== 6. 模型 =====
    logger.info("Building model...")
    arch_name = cfg.architecture.name
    if arch_name == "dr_unet":
        model = build_dr_unet(cfg)
        logger.info("Using DR-UNet (dual-routing) as model.")
    else:
        model = build_unet(cfg)
        logger.info("Using baseline UNet2D as model.")
    model = model.to(device)
    logger.info(f"Model: {arch_name}")

    # 打印模型参数数量
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params:,}")

    # ===== 7. 损失函数 =====
    logger.info("Building loss function...")
    criterion = build_loss(cfg.training.loss)
    logger.info(f"Loss: {cfg.training.loss.name}")

    # ===== 8. 优化器 =====
    logger.info("Building optimizer...")
    from utils.utils import get_optimizer

    optimizer = get_optimizer(model, cfg.training)
    logger.info(
        f"Optimizer: {cfg.training.optimizer.name}, LR: {cfg.training.learning_rate}"
    )

    # ===== 9. 学习率调度器 =====
    logger.info("Building scheduler...")
    from utils.utils import get_scheduler

    scheduler = get_scheduler(optimizer, cfg.training)
    if scheduler is not None:
        logger.info(f"Scheduler: {cfg.training.scheduler.name}")
    else:
        logger.info("No scheduler")

    # ===== 10. 训练器 =====
    logger.info("Building trainer...")
    if arch_name == "dr_unet":
        trainer = build_dr_trainer(
            cfg=cfg,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    else:
        trainer = build_trainer(
            cfg=cfg,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    # ===== 11. 开始训练 =====
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)

    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1

    logger.info("=" * 50)
    logger.info(f"Training finished. Results saved to: {exp_dir}")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
