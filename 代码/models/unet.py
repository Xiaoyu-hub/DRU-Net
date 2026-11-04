"""
Baseline U-Net 模型 (2D)
参考 RARE-UNet 的模块化设计，适用于皮肤病灶分割
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_norm_layer(norm_type: str, num_features: int):
    """归一化层工厂函数"""
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    elif norm_type == "group":
        # GroupNorm: 自动计算 groups 数
        num_groups = min(32, num_features // 4)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    elif norm_type == "layer":
        return nn.LayerNorm(num_features)
    elif norm_type == "none" or norm_type == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization type: {norm_type}")


def _get_activation_layer(activation_type: str):
    """激活层工厂函数"""
    activation_type = activation_type.lower()
    if activation_type == "relu":
        return nn.ReLU(inplace=True)
    elif activation_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "silu" or activation_type == "swish":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation type: {activation_type}")


class ConvBlock(nn.Module):
    """单卷积块: Conv -> Norm -> Activation"""
    def __init__(self, in_channels, out_channels, norm_type, activation_type, kernel_size=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            _get_norm_layer(norm_type, num_features=out_channels),
            _get_activation_layer(activation_type),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """编码器块: 双卷积 (Conv -> Norm -> Act -> Conv -> Norm -> Act)"""
    def __init__(self, in_channels, out_channels, norm_type, activation_type, kernel_size=3):
        super(EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            _get_norm_layer(norm_type, num_features=out_channels),
            _get_activation_layer(activation_type),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            _get_norm_layer(norm_type, num_features=out_channels),
            _get_activation_layer(activation_type),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """解码器块: 单卷积 (Conv -> Norm -> Act)"""
    def __init__(self, in_channels, out_channels, norm_type, activation_type, kernel_size=3):
        super(DecoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            _get_norm_layer(norm_type, num_features=out_channels),
            _get_activation_layer(activation_type),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """
    Baseline U-Net 2D 模型
    
    适用于皮肤病灶分割任务，支持配置化参数
    """
    def __init__(self, cfg):
        super(UNet2D, self).__init__()
        
        # 从配置读取参数
        self.depth = cfg.architecture.depth
        self.in_channels = cfg.dataset.in_channels
        self.num_classes = cfg.dataset.num_classes - 1  # 背景不算作输出通道
        self.n_filters = cfg.architecture.n_filters
        self.target_shape = tuple(cfg.dataset.target_shape)  # 用于 RARE 子类推理
        self.dropout = cfg.architecture.dropout
        self.norm_type = cfg.architecture.norm_type
        self.activation_type = cfg.architecture.activation_type
        
        # 日志记录参数
        self.logger = logging.getLogger(__name__)
        for name, val in vars(self).items():
            skip_log = name == "logger" or name.startswith('_') or isinstance(val, nn.Module)
            if skip_log:
                continue
            self.logger.info(f"{name} = {val}")

        # ========== 构建编码器 ==========
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.enc_dropouts = nn.ModuleList()
        
        for d in range(self.depth):
            # 计算输入通道数
            in_ch = self.in_channels if d == 0 else self.n_filters * (2 ** (d - 1))
            # 计算输出通道数
            out_ch = self.n_filters * (2 ** d)
            
            # 限制最大通道数
            in_ch = min(in_ch, 512)
            out_ch = min(out_ch, 512)
            
            self.encoders.append(
                EncoderBlock(in_ch, out_ch, self.norm_type, self.activation_type)
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.enc_dropouts.append(nn.Dropout2d(self.dropout))

        # ========== 瓶颈层 (Bottleneck) ==========
        bottleneck_in = min(self.n_filters * (2 ** (self.depth - 1)), 512)
        bottleneck_out = min(self.n_filters * (2 ** self.depth), 512)
        self.bottleneck = EncoderBlock(
            bottleneck_in,
            bottleneck_out,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
        )

        # ========== 构建解码器 ==========
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dec_dropouts = nn.ModuleList()
        
        for d in range(self.depth - 1, -1, -1):
            # 上采样层
            up_in = self.n_filters * (2 ** (d + 1))
            up_out = self.n_filters * (2 ** d)
            up_in = min(up_in, 512)
            up_out = min(up_out, 512)
            
            self.up_convs.append(
                nn.ConvTranspose2d(
                    up_in, up_out, kernel_size=2, stride=2
                )
            )
            
            # 解码器块（输入 = 上采样输出 + 跳跃连接）
            dec_in = 2 * up_out
            dec_out = up_out
            self.decoders.append(
                DecoderBlock(dec_in, dec_out, self.norm_type, self.activation_type)
            )
            self.dec_dropouts.append(nn.Dropout2d(self.dropout))

        # ========== 输出层 ==========
        self.final_conv = nn.Conv2d(self.n_filters, self.num_classes, kernel_size=1)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            输出分割掩码 [B, num_classes, H, W]
        """
        # 检查输入尺寸
        min_size = 2 ** self.depth
        assert all(dim >= min_size for dim in x.shape[2:]), \
            f"Input spatial dimensions must be at least {min_size}, but got {x.shape[2:]}"

        # ========== 编码器路径 ==========
        enc_feats = []
        out = x
        
        for enc, pool, drop in zip(self.encoders, self.pools, self.enc_dropouts):
            out = enc(out)
            enc_feats.append(out)
            out = drop(pool(out))

        # ========== 瓶颈层 ==========
        center_out = self.bottleneck(out)

        # ========== 解码器路径 ==========
        out = center_out
        for up_conv, dec, drop, skip in zip(
            self.up_convs, self.decoders, self.dec_dropouts, reversed(enc_feats)
        ):
            out = up_conv(out)
            # 跳跃连接: 上采样特征 + 编码器特征
            out = torch.cat([out, skip], dim=1)
            out = dec(out)
            out = drop(out)

        # ========== 输出 ==========
        final = self.final_conv(out)
        
        return final


class UNet2DWithAuxHeads(nn.Module):
    """
    带辅助输出头的 U-Net (支持多尺度监督)
    适用于 DR²-UNet 轻分支
    """
    def __init__(self, cfg, aux_heads=3):
        super(UNet2DWithAuxHeads, self).__init__()
        
        self.unet = UNet2D(cfg)
        self.aux_heads = aux_heads
        
        # 添加辅助输出头（从不同层级提取特征）
        # 这里暂时预留接口，具体实现可在 DR²-UNet 中扩展

    def forward(self, x):
        return self.unet(x)


def build_unet(cfg):
    """
    U-Net 模型构建工厂函数
    
    Args:
        cfg: Hydra 配置对象
        
    Returns:
        UNet2D 模型实例
    """
    return UNet2D(cfg)
