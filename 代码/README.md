# DR²-UNet: Dual-Routing U-Net for Resource-Constrained Lesion Segmentation

项目结构模仿 RARE-UNet，便于实现难度路由、多出口与动态多损失。

## 目录结构

```
代码/
├── config/                     # 配置
│   ├── architecture/          # 模型结构 (unet.yaml, dr_unet.yaml)
│   ├── dataset/                # 数据集配置
│   ├── training/               # 训练配置
│   └── base.yaml
├── data/                       # 数据与预处理
│   ├── data_manager.py
│   ├── datasets.py
│   └── preprocess_data.py
├── models/                     # 模型
│   ├── unet.py                 # 基线 U-Net
│   └── dr_unet.py             # DR²-UNet
├── trainers/                   # 训练逻辑
│   ├── trainer.py
│   ├── dr_trainer.py
│   └── early_stopping.py
├── utils/                      # 工具
│   ├── checkpoint_handler.py
│   ├── logging.py
│   ├── losses.py
│   ├── metric_collecter.py
│   ├── metrics.py
│   ├── table.py
│   ├── utils.py
│   ├── wandb_logger.py
│   └── weight_strategies.py
├── example.py
├── inference.py
├── train.py
├── README.md
└── setup.py
```

## 使用说明

安装、数据准备、训练、评估与推理的命令见仓库根目录的 [README.md](../README.md)。
