# DR²-UNet：面向资源受限场景的双路由皮肤病灶分割

在标准 U-Net 上加了「难度路由」：浅层轻分支先出结果，判断为难样本再进重分支精修。目标是在精度基本不降的前提下，让大部分容易样本只走 20% 的计算路径。

在 ISIC2018（2075 训练 / 259 验证 / 260 测试，256×256）上，动态路由模式下 96.92% 的样本在轻分支早退，平均计算量约为全路径的 25%。

## 方法

**轻/重双分支，共享编码器。** 编码器只算一遍，特征同时喂给两个解码器：

| 分支 | 通道配置 | 作用 |
|---|---|---|
| Light | [8, 16, 24, 32] | 快速出口，处理大部分简单病灶 |
| Heavy | [32, 64, 128, 256] | 标准 U-Net 宽度，精修难样本 |

**难度 Router。** 输入是轻分支编码器的全局池化特征、预测熵、全局置信度三者的组合（可配置开关），输出走哪条路径。阈值 0.3，隐藏层 64 维。仓库里带了只保留 GAP 特征的消融配置，见 `有效实验报告/消融实验.md`。

**多出口 + 动态多损失。** 训练时同时监督轻分支输出、最终输出、路由决策和一致性项，权重按训练阶段调整。三种推理模式对应三种部署取舍：

- `light`：只走轻分支，约 20% 计算量
- `dynamic`：按 Router 判断决定，约 25%
- `heavy`：全路径，精度上限

## 结果

测试集 260 张的实测结果：

| 方法 | Dice | IoU | 计算量 |
|---|---|---|---|
| Baseline U-Net | 0.8844 | 0.8144 | 100% |
| Heavy 全路径 | 0.8921 | 0.8258 | 100% |
| Dynamic 动态路由 | 0.8711 | 0.7989 | ~25% |
| Light 仅轻分支 | 0.8694 | 0.7972 | 20% |

验证集上最佳 Dice 0.9588 / IoU 0.9323，早退率 97.73%，出现在第 140 轮左右；训练跑满 300 轮，单卡 batch 8 + AMP，141 s/epoch，合计约 13.7 小时。

说得直白些：**动态路由在测试集上比 baseline 低 1.3 个点，换来约 4 倍计算量下降。** 这个交换在资源受限场景是否划算，取决于你能容忍多少精度损失，仓库里三种模式都可以直接跑出来对比。

模型总参数量 17.5M（双分支共享编码器）。

## 目录结构

```
代码/
├── config/              # Hydra 配置：architecture / dataset / training 可用命令行覆盖
├── data/                # 数据集与预处理（preprocess_data.py 做 resize 与二值化）
├── models/              # unet.py 基线，dr_unet.py 双路由实现
├── trainers/            # 训练循环、早停
├── utils/               # 损失、指标、checkpoint、后处理
├── train.py             # 训练入口
├── test.py              # 测试集评估（轻重动态三模式对比）
└── inference.py         # 单图推理，支持 TTA 与形态学后处理
有效实验报告/            # 主实验、基线对比、消融实验的实测记录
项目报告/                # 技术报告与任务规划
figures/                 # 架构图（绘图脚本一并保留）
```

## 环境

Python 3.9+，PyTorch 1.12+（AMP 需要 CUDA），另外用到 hydra-core、omegaconf、opencv-python、numpy、pillow、tqdm。

```bash
pip install torch torchvision
pip install hydra-core omegaconf opencv-python numpy pillow tqdm
```

## 数据准备

ISIC2018 需要在 [ISIC Archive](https://www.isic-archive.com/) 自行申请下载，仓库不分发数据。放成这样即可：

```
数据集/isic2018/
├── images/    # 输入图像
└── masks/     # 对应掩膜，0 背景 1 病灶
```

路径在 `代码/config/dataset/isic2018.yaml` 的 `base_path` 里改，或者运行时覆盖。

## 训练

```bash
cd 代码
python train.py                                   # 默认 DR²-UNet
python train.py architecture=unet                 # 换成基线 U-Net
python train.py gpu.devices=[1] training.batch_size=4   # 换卡、改 batch
python data/preprocess_data.py                    # 先把原始数据 resize 到 256×256
```

结果落在 `trained_models/<实验名>/`，日志里有每轮 Dice、IoU、硬样本比例、早退率。checkpoint 按验证 Dice 存最优，早停 patience 50。

## 评估与推理

```bash
cd 代码
python test.py --checkpoint trained_models/xxx/dr_unet_best.pth --model dr_unet --mode dynamic
python inference.py --checkpoint trained_models/xxx/dr_unet_best.pth --image path/to/img.jpg --mode dynamic --output mask.png
```

`test.py` 会把三种模式的 Dice / IoU / 早退率一起跑出来。

## 说明与局限

- 消融只完成了「仅 GAP 特征」一组（A1），加上熵特征和置信度两组还在排队，`有效实验报告/消融实验.md` 里标了状态。现在还不能说风险特征一定有用，只能说 A1 的表现和完整 Router 相当。
- 单卡训练，没有做多卡并行；`trainers/` 里留了多 GPU 的分支但未实现。
- 后处理模块（形态学清理、去小连通域）接口在 `utils/postprocess.py`，评估时没有默认开启。
- 训练日志与中间产物没有提交，`有效实验报告/` 里的数字来自当时的运行记录。

## 致谢

项目结构参考 RARE-UNet，训练设置（batch 8、AdamW、CosineAnnealing、300 轮）沿用 LB-UNet 在 11GB 显存下的配置。U-Net 及各类变体的原始实现归各自作者所有。

## License

MIT，见 [LICENSE](LICENSE)。
