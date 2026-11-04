# Notes: DR-UNet 项目详细报告

## 研究资料收集

### 已收集的文档
1. 论文初稿 (论文初稿.md) - 包含 Abstract, Introduction, Methods, Experiments, Discussion, Conclusion
2. 主实验报告 (drunet-dyandli-unet实验报告.md) - 包含四种推理模式对比
3. 消融实验报告 (消融实验.md) - 包含消融实验A1结果
4. 创新点总结 (DR2-unet创新点总结.md) - 创新点梳理

### 代码文件分析

#### dr_unet.py (525行)
主要类：
- `DepthwiseSeparableConv` (18-41行): 深度可分卷积
- `LightBranch` (44-189行): 轻分支
- `HeavyBranch` (192-203行): 重分支
- `DifficultyRouter` (206-302行): 难度路由器
- `DRUNet` (305-512行): 主模型

#### 关键配置参数
- Light Branch channels: [8, 16, 24, 32]
- Heavy Branch channels: [32, 64, 128, 256] (标准U-Net)
- Router hidden_dim: 64
- Router threshold: 0.3

### 实验数据汇总

#### 主实验结果 (ISIC2018测试集, 260张)
| 模式 | Dice | IoU | 计算量占比 |
|------|------|-----|-----------|
| Heavy-only | 0.8921 | 0.8258 | 100% |
| Dynamic | 0.8711 | 0.7989 | ~25% |
| Light-only | 0.8694 | 0.7972 | 20% |
| Baseline U-Net | 0.8844 | 0.8144 | 100% |

#### 动态路由分析
- 早退率: 96.92%
- 困难样本比例: 3.08%

#### 消融实验A1结果
- 早退率: 94.62%
- Dynamic Dice: 0.8742
- Heavy Dice: 0.8937

### 参考论文
已下载8篇PDF论文在 参考 文件夹中：
- U-Net (原始论文)
- Attention U-Net
- MobileNetV2
- Xception
- R2U-Net
- MultiResUNet
- UNet++
- ScaleFusionNet

## 报告撰写进度

### Phase 1: 项目结构梳理 ✅
- [x] 探索项目文件夹结构
- [x] 收集所有关键文档
- [x] 分析代码文件

### Phase 2: 模型架构设计思路
- 待撰写

### Phase 3: 核心代码分析
- 待撰写

### Phase 4: 实验结果分析
- 待撰写

### Phase 5: 消融实验分析
- 待撰写

### Phase 6: 对比分析
- 待撰写

### Phase 7: 报告整合
- 待撰写
