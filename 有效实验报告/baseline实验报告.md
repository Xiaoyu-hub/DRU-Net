## 实验报告：Baseline U-Net（ISIC2018 皮肤病灶二分类分割）

### 1. 实验概述
- **实验目的**：训练并评估基线分割模型（2D U-Net）在 ISIC2018 皮肤病灶分割任务上的性能，为后续 DR²-UNet（双路由+轻重分支）对比实验提供参考。
- **任务定义**：二分类语义分割（背景 vs 病灶），输出通道 `out_channels=1`。
- **实验有效性结论**：**有效实验**。训练流程完整执行至 `300/300` epoch，验证指标随训练变化并在中期达到峰值，且持续保存 `baseline_last.pth` 与 `baseline_best.pth`，日志无崩溃/异常终止记录。

### 2. 运行环境与复现信息
- **运行时间**：从 `2026-03-09 17:37` 至 `2026-03-10 07:41`（日志记录训练总耗时 **844.75 min**）。
- **硬件**：单卡 GPU，日志显示 `cuda:2`。
- **随机性控制**：`seed=42`，`optimization.mode=deterministic`。
- **配置快照**：见本目录 `.hydra/config.yaml`（Hydra 自动保存）。

### 3. 数据集与预处理
- **数据集**：ISIC2018（皮肤病灶分割）。
- **样本数**：训练集 **2075**，验证集 **259**（来自 `train.log`）。
- **输入尺寸**：`256×256`（resize）。
- **输入通道**：RGB，`in_channels=3`。
- **标签**：二值掩码（0/1）。
- **数据增强**：启用（随机水平翻转、垂直翻转、旋转、ColorJitter）。
- **多尺度训练**：本次 baseline 运行中为 `multiscale=false`（见 `.hydra/config.yaml`）。

### 4. 模型与训练设置
- **模型**：Baseline `UNet2D`（depth=5，n_filters=32，InstanceNorm，LeakyReLU，dropout=0.2）。
- **参数量**：**17,472,065**（来自 `train.log`）。
- **损失函数**：`DiceBCE loss`（Dice + BCE）。
- **优化器**：AdamW，学习率 `1e-3`，权重衰减 `1e-4`。
- **学习率调度**：CosineAnnealingLR（`T_max=50, eta_min=1e-6`）。
- **训练轮次**：300 epochs。
- **Batch size**：8；`num_workers=4`；`pin_memory=true`；AMP：`use_amp=true`。
- **Checkpoint**：
  - 最新：`trained_models/baseline_last.pth`
  - 最优：`trained_models/baseline_best.pth`

### 5. 结果（主指标）
本项目验证阶段日志提供 Dice / IoU / Loss。下表汇总关键节点（从 `train.log` 解析得到）：

| 指标 | 最优验证（epoch=111） | 并列最优（epoch=201） | 最终（epoch=300） |
|---|---:|---:|---:|
| Val Loss | 0.1027 | 0.1113 | 0.1441 |
| Val Dice | **0.9604** | **0.9604** | 0.9547 |
| Val IoU | 0.9335 | 0.9329 | 0.9264 |
| 训练耗时/epoch（Train Time, s） | 150.00 | 149.37 | 153.47 |
| 验证耗时/epoch（Val Time, s） | 18.05 | 17.82 | 18.16 |

对应训练集指标（同一 epoch）：
- **epoch=111**：Train Loss 0.0901，Train Dice 0.9770，Train IoU 0.9554  
- **epoch=201**：Train Loss 0.0746，Train Dice 0.9804，Train IoU 0.9618  
- **epoch=300**：Train Loss 0.0619，Train Dice 0.9832，Train IoU 0.9671  

### 6. 讨论与结论
- **收敛性**：验证 Dice 在训练中期达到峰值（epoch=111/201），后期（epoch=300）略有回落，表现为轻微过拟合或学习率调度带来的波动。
- **基线强度**：在该数据划分与输入尺寸设置下，baseline U-Net 已达到较高的验证 Dice（0.9604），可作为后续 DR²-UNet 的对照基线。
- **训练策略提示**：
  - 最佳验证性能出现在中期，后续实验可考虑引入更严格的早停/模型选择策略（以验证 Dice 为准）。
  - 本次使用 CosineAnnealingLR 且 `T_max=50`、训练 300 epoch，学习率会经历多段余弦周期（非 warm restart 但会产生周期性变化）。若希望单调退火，可将 `T_max` 设为总 epoch 或使用 WarmRestarts 并明确周期设计。

### 7. 产出文件与校验信息
- **日志**：`train.log`
- **Hydra 配置快照**：`.hydra/config.yaml`, `.hydra/hydra.yaml`, `.hydra/overrides.yaml`
- **模型权重**：
  - `trained_models/baseline_best.pth`（MD5：`1c49ce563e565094510becbb9ffd60b0`，209,763,786 bytes）
  - `trained_models/baseline_last.pth`（MD5：`7fb97c3c903029a97c6b0bc4a95f5b6e`，209,763,786 bytes）

