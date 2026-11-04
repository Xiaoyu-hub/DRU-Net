"""
DR²-UNet 模型实现
难度路由 + 轻重双分支
参考 LB-UNet + BranchyNet（早退思想）+ 医学分割不确定性路由
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from models.unet import UNet2D, _get_norm_layer, _get_activation_layer

logger = logging.getLogger(__name__)


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分卷积（轻量级卷积）
    参考 LB-UNet 的设计
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        # 点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightBranch(nn.Module):
    """
    轻分支：轻量级 U-Net
    使用深度可分卷积减少参数量
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        arch_cfg = cfg.architecture

        # 获取轻分支通道配置
        light_channels = (
            arch_cfg.light_branch.channels
            if hasattr(arch_cfg, "light_branch")
            else [8, 16, 24, 32]
        )

        self.use_depthwise = (
            arch_cfg.light_branch.use_depthwise_conv
            if hasattr(arch_cfg, "light_branch")
            else True
        )

        in_channels = arch_cfg.in_channels
        norm_type = arch_cfg.norm_type
        activation_type = arch_cfg.activation_type

        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i, out_ch in enumerate(light_channels):
            conv = (
                DepthwiseSeparableConv(in_channels, out_ch)
                if self.use_depthwise
                else nn.Conv2d(in_channels, out_ch, 3, padding=1)
            )
            self.encoders.append(
                nn.Sequential(
                    conv,
                    _get_norm_layer(norm_type, out_ch),
                    _get_activation_layer(activation_type),
                )
            )
            self.pools.append(nn.MaxPool2d(2))
            in_channels = out_ch

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(in_channels, light_channels[-1])
            if self.use_depthwise
            else nn.Conv2d(in_channels, light_channels[-1], 3, padding=1),
            _get_norm_layer(norm_type, light_channels[-1]),
            _get_activation_layer(activation_type),
        )

        # 解码器
        # 正确逻辑：
        # - encoder_features: [enc0(8), enc1(16), enc2(24), enc3(32)]
        # - 编码器 4 层，4 次下采样：256→128→64→32→16
        # - 解码器 4 层，4 次上采样：16→32→64→128→256
        # - skip 连接: [enc3(32), enc2(24), enc1(16), enc0(8)]

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # 第一个 decoder 的输入是 bottleneck 输出
        prev_out_ch = light_channels[-1]  # 32

        # 编码器通道列表（从后往前）：[32, 24, 16, 8]
        encoder_channel_list = list(reversed(light_channels))  # [32, 24, 16, 8]

        for i in range(len(encoder_channel_list)):
            skip_ch = encoder_channel_list[i]

            # 输出通道：第一层是 light_channels[-2]=24，最后一层是 light_channels[0]=8
            if i < len(encoder_channel_list) - 1:
                out_ch = light_channels[-(i + 2)]  # i=0:24, i=1:16, i=2:8
            else:
                out_ch = light_channels[0]  # i=3:8

            self.up_convs.append(nn.ConvTranspose2d(prev_out_ch, out_ch, 2, stride=2))

            # cat 后的通道数 = 上采样输出 + skip 连接
            in_ch_cat = out_ch + skip_ch
            conv = (
                DepthwiseSeparableConv(in_ch_cat, out_ch)
                if self.use_depthwise
                else nn.Conv2d(in_ch_cat, out_ch, 3, padding=1)
            )
            self.decoders.append(
                nn.Sequential(
                    conv,
                    _get_norm_layer(norm_type, out_ch),
                    _get_activation_layer(activation_type),
                )
            )

            prev_out_ch = out_ch

        # 输出层
        self.final_conv = nn.Conv2d(light_channels[0], arch_cfg.out_channels, 1)

        # 保存用于 Router 的特征
        self.encoder_features = []
        self.gap_features = None

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: 分割输出 [B, C, H, W]
            gap_features: 全局平均池化特征，用于 Router
        """
        self.encoder_features = []

        # 编码器
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            self.encoder_features.append(x)
            x = pool(x)

        # 瓶颈
        x = self.bottleneck(x)

        # 全局平均池化特征（用于 Router）
        self.gap_features = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # 解码器 - 使用对应的 skip connection
        # encoder_features 存储顺序: [enc0(8), enc1(16), enc2(24), enc3(32)]
        # 编码器 4 层，4 次下采样：256→128→64→32→16
        # 解码器 4 层，4 次上采样：16→32→64→128→256
        # Decoder0: 16→32, 需要 enc3(32)=index 3
        # Decoder1: 32→64, 需要 enc2(24)=index 2
        # Decoder2: 64→128, 需要 enc1(16)=index 1
        # Decoder3: 128→256, 需要 enc0(8)=index 0
        skip_indices = [3, 2, 1, 0]

        for i, (up_conv, dec) in enumerate(zip(self.up_convs, self.decoders)):
            x = up_conv(x)
            skip = self.encoder_features[skip_indices[i]]
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        output = self.final_conv(x)
        return output, self.gap_features


class HeavyBranch(nn.Module):
    """
    重分支：标准 U-Net（重计算路径）
    用于处理难样本的精修
    """

    def __init__(self, cfg):
        super().__init__()
        self.net = UNet2D(cfg)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class DifficultyRouter(nn.Module):
    """
    难度路由器
    根据轻分支的特征和预测结果判断样本难度
    """

    def __init__(self, cfg):
        super().__init__()
        arch_cfg = cfg.architecture
        router_cfg = arch_cfg.router if hasattr(arch_cfg, "router") else {}

        self.use_entropy = router_cfg.get("use_entropy", True)
        self.use_feature = router_cfg.get("use_feature", True)
        self.use_confidence = router_cfg.get("use_confidence", True)

        # 获取轻分支输出通道数
        light_channels = (
            arch_cfg.light_branch.channels[-1]
            if hasattr(arch_cfg, "light_branch")
            else 32
        )
        hidden_dim = router_cfg.get("hidden_dim", 64)

        # 特征输入维度
        feature_dim = 0
        if self.use_feature:
            feature_dim += light_channels  # 32
        if self.use_entropy:
            feature_dim += 1  # entropy
        if self.use_confidence:
            feature_dim += 1  # confidence

        print(
            f"[Router Config] use_feature={self.use_feature}, use_entropy={self.use_entropy}, use_confidence={self.use_confidence}"
        )
        print(
            f"[Router Config] light_channels={light_channels}, feature_dim={feature_dim}, hidden_dim={hidden_dim}"
        )

        # Router 网络
        self.router = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, gap_features: torch.Tensor, light_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gap_features: 轻分支全局池化特征 [B, C]
            light_output: 轻分支输出 [B, C, H, W]

        Returns:
            router_logits: 路由器输出 [B, 1]（正数=难样本，负数=易样本）
            is_hard: 难度标签 [B, 1]
        """
        router_inputs = []

        # 特征输入
        if self.use_feature:
            router_inputs.append(gap_features)

        # 熵和置信度
        if self.use_entropy or self.use_confidence:
            # light_output: [B, C, H, W] -> 使用全局池化得到标量
            probs = torch.softmax(light_output, dim=1)  # [B, C, H, W]

            # 对每个通道取平均，然后计算全局统计
            # 置信度：最大概率的均值
            max_prob = probs.max(dim=1)[0]  # [B, H, W]
            confidence = max_prob.mean(dim=[1, 2]).unsqueeze(1)  # [B, 1]

            # 熵：-sum(p * log(p)) 的均值
            entropy_per_pixel = -(probs * torch.log(probs + 1e-8)).sum(
                dim=1
            )  # [B, H, W]
            entropy = entropy_per_pixel.mean(dim=[1, 2]).unsqueeze(1)  # [B, 1]

            if self.use_confidence:
                router_inputs.append(confidence)
            if self.use_entropy:
                router_inputs.append(entropy)

        # 拼接所有输入
        router_input = torch.cat(router_inputs, dim=1)

        # Router 决策
        router_logits = self.router(router_input)

        # 难度标签（使用 sigmoid 和阈值）
        threshold = 0.5
        is_hard = (torch.sigmoid(router_logits) > threshold).float()

        return router_logits, is_hard


class DRUNet(nn.Module):
    """
    DR-UNet: Dual-Routing U-Net

    难度路由：
    根据 Router 判断样本难度，决定是否使用重分支
    """

    def __init__(self, cfg, mode: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        arch_cfg = cfg.architecture

        # 基础参数
        self.depth = arch_cfg.depth
        self.in_channels = arch_cfg.in_channels
        self.num_classes = arch_cfg.out_channels
        self.n_filters = arch_cfg.n_filters
        self.norm_type = arch_cfg.norm_type
        self.activation_type = arch_cfg.activation_type

        # 配置参数
        self.use_light_branch = arch_cfg.get("use_light_branch", True)
        self.use_heavy_branch = arch_cfg.get("use_heavy_branch", True)
        self.use_router = arch_cfg.get("use_router", True)

        # 推理模式
        self.inference_mode = (
            arch_cfg.inference.mode if hasattr(arch_cfg, "inference") else "dynamic"
        )

        # 初始化各分支
        if self.use_light_branch:
            self.light_branch = LightBranch(cfg)

        if self.use_heavy_branch:
            self.heavy_branch = HeavyBranch(cfg)

        if self.use_router:
            self.router = DifficultyRouter(cfg)

        logger.info(
            f"DRUNet initialized: mode={mode}, "
            f"light={self.use_light_branch}, "
            f"heavy={self.use_heavy_branch}, "
            f"router={self.use_router}"
        )

    def _sync_heavy_branch_mode(self):
        """保留接口，当前重分支不需要 mode 同步"""
        return

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            包含以下键的字典：
            - light_output: 轻分支输出
            - heavy_output: 重分支输出（可选）
            - router_logits: 路由器输出
            - is_hard: 难度标签
            - final_output: 最终输出
        """
        if self.mode == "inference" or self.mode == "test":
            return self.forward_inference(x)
        else:
            return self.forward_train(x)

    def forward_train(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        训练模式前向传播
        轻分支固定 256×256；重分支同样在 256×256 上训练（便于对齐与融合）
        """
        outputs = {}
        self._sync_heavy_branch_mode()

        # 训练时统一到 256×256（对齐 RARE 设计：输入始终为 full-res）
        h, w = x.shape[2], x.shape[3]
        if (h, w) != (256, 256):
            x_processed = F.interpolate(
                x, size=(256, 256), mode="bilinear", align_corners=False
            )
        else:
            x_processed = x

        # 1. 轻分支
        if self.use_light_branch:
            light_output, gap_features = self.light_branch(x_processed)
            outputs["light_output"] = light_output
            outputs["gap_features"] = gap_features
        else:
            light_output = None
            gap_features = None

        # 2. Router
        if self.use_router and self.use_light_branch:
            router_logits, is_hard = self.router(gap_features, light_output)
            outputs["router_logits"] = router_logits
            outputs["is_hard"] = is_hard
        else:
            router_logits = None
            is_hard = None

        # 3. 重分支（标准 U-Net）
        if self.use_heavy_branch:
            heavy_output = self.heavy_branch(x_processed)
            outputs["heavy_output"] = heavy_output
        else:
            heavy_output = None

        # 4. 最终输出（训练时使用融合输出，让两个分支都有训练信号）
        if self.use_heavy_branch and light_output is not None:
            # 训练时始终有重分支，融合轻分支和重分支
            # 权重 0.3 让重分支主导训练
            outputs["final_output"] = light_output + 0.3 * heavy_output
        elif light_output is not None:
            # 只有轻分支
            outputs["final_output"] = light_output
        elif self.use_heavy_branch:
            outputs["final_output"] = heavy_output
        else:
            raise ValueError("At least one branch should be enabled")

        return outputs

    def forward_inference(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        推理模式前向传播
        重分支为标准 U-Net；动态路由仅由“难度路由器”决定是否触发重分支
        """
        outputs = {}
        self._sync_heavy_branch_mode()

        # 轻分支固定 256×256；重分支也使用 256×256，便于融合
        h, w = x.shape[2], x.shape[3]
        x_light = (
            F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            if (h, w) != (256, 256)
            else x
        )
        x_heavy = x_light

        if self.inference_mode == "light":
            light_output, _ = self.light_branch(x_light)
            outputs["light_output"] = light_output
            outputs["final_output"] = light_output
            outputs["used_branch"] = "light"

        elif self.inference_mode == "heavy":
            heavy_output = self.heavy_branch(x_heavy)
            outputs["heavy_output"] = heavy_output
            outputs["final_output"] = heavy_output
            outputs["used_branch"] = "heavy"

        elif self.inference_mode == "dynamic":
            if self.use_light_branch:
                light_output, gap_features = self.light_branch(x_light)
                outputs["light_output"] = light_output

                if self.use_router:
                    router_logits, is_hard = self.router(gap_features, light_output)
                    outputs["router_logits"] = router_logits
                    outputs["is_hard"] = is_hard

                    # 逐样本判断：只对难样本走重分支
                    batch_size = is_hard.size(0)
                    final_output = light_output.clone()
                    used_branch_list = []

                    # 找到难样本的索引
                    hard_indices = torch.where(is_hard.squeeze() > 0)[0]

                    if len(hard_indices) > 0 and self.use_heavy_branch:
                        # 只对难样本计算重分支
                        hard_x = x_heavy[hard_indices]
                        heavy_output = self.heavy_branch(hard_x)
                        outputs["heavy_output"] = heavy_output

                        # 只融合难样本
                        for i, idx in enumerate(hard_indices):
                            final_output[idx] = (
                                light_output[idx] + 0.3 * heavy_output[i]
                            )

                    # 记录每个样本使用的分支
                    for i in range(batch_size):
                        if i in hard_indices:
                            used_branch_list.append("heavy")
                        else:
                            used_branch_list.append("light")

                    outputs["used_branch"] = used_branch_list
                else:
                    final_output = light_output
                    outputs["used_branch"] = ["light"] * light_output.size(0)
            else:
                final_output = self.heavy_branch(x_heavy)
                outputs["heavy_output"] = final_output
                outputs["used_branch"] = "heavy"

            outputs["final_output"] = final_output
        else:
            raise ValueError(f"Unknown inference mode: {self.inference_mode}")

        return outputs


def build_dr_unet(cfg) -> DRUNet:
    """
    DR-UNet 模型构建工厂函数

    Args:
        cfg: Hydra 配置对象

    Returns:
        DRUNet 模型实例
    """
    return DRUNet(cfg)
