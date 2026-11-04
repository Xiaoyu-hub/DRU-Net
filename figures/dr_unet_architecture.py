#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DR-UNet 模型架构图
配色方案：参考百元大钞 - 以墨绿色为主色调
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 百元大钞配色方案
COLORS = {
    # 主色调 - 墨绿色系
    "primary_green": "#1D6F42",  # 墨绿 - 主色
    "light_green": "#2E8B57",  # 海绿色
    "pale_green": "#98FB98",  # 淡绿色
    "dark_green": "#0D4F2F",  # 深墨绿
    # 辅助色 - 红橙色系（点缀）
    "accent_red": "#C41E3A",  # 中国红
    "orange": "#E86832",  # 橙色
    # 中性色
    "white": "#FFFFFF",
    "light_gray": "#F5F5F5",
    "gray": "#808080",
    "dark_gray": "#404040",
    "black": "#1A1A1A",
    # 渐变色
    "gradient_start": "#2E8B57",
    "gradient_end": "#1D6F42",
}


def draw_block(ax, x, y, width, height, color, label, text_color="white", fontsize=9):
    """绘制一个方块"""
    rect = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor=COLORS["dark_green"],
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
    )
    return rect


def draw_arrow(ax, start, end, color=COLORS["dark_green"], style="simple"):
    """绘制箭头连接"""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->", color=color, lw=2, connectionstyle="arc3,rad=0"
        ),
    )


def draw_dashed_arrow(ax, start, end, color=COLORS["gray"]):
    """绘制虚线箭头"""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.5,
            linestyle="--",
            connectionstyle="arc3,rad=0",
        ),
    )


def create_architecture_diagram():
    """创建 DR-UNet 架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # 标题
    ax.text(
        8,
        11.5,
        "DR-UNet: Dual-Routing U-Net for Skin Lesion Segmentation",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=COLORS["dark_green"],
    )
    ax.text(
        8,
        11,
        "(Difficulty-Aware Routing with Light/Heavy Dual-Branch Architecture)",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["gray"],
    )

    # ============== 输入层 ==============
    draw_block(ax, 0.5, 9, 1.5, 1.2, COLORS["dark_green"], "Input\n256×256", fontsize=8)

    # ============== 主干网络 ==============
    # 编码器部分 (4层)
    encoder_x = 2.5
    encoder_channels = [8, 16, 24, 32]
    encoder_blocks = []

    for i, ch in enumerate(encoder_channels):
        y_pos = 9 - i * 0.8
        color = COLORS["primary_green"]
        draw_block(
            ax,
            encoder_x + i * 1.2,
            y_pos,
            1,
            0.7,
            color,
            f"Enc-{i + 1}\n{ch}ch",
            fontsize=7,
        )

    # 向下箭头
    for i in range(3):
        y_pos = 8.6 - i * 0.8
        ax.annotate(
            "",
            xy=(2.5 + i * 1.2 + 0.5, 8.3 - i * 0.8),
            xytext=(2.5 + i * 1.2 + 0.5, y_pos),
            arrowprops=dict(arrowstyle="->", color=COLORS["dark_green"], lw=1.5),
        )

    # ============== 分支结构 ==============
    # Light Branch (左侧 - 浅绿色)
    ax.text(
        6,
        9.5,
        "Light Branch",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["light_green"],
    )

    # 轻分支解码器
    for i in range(4):
        y_pos = 6.5 - i * 0.7
        color = COLORS["pale_green"]
        draw_block(
            ax,
            5.5,
            y_pos,
            1,
            0.5,
            color,
            f"Dec-L{i + 1}",
            text_color=COLORS["dark_gray"],
            fontsize=7,
        )

    # 轻分支输出
    draw_block(
        ax, 5.5, 1.5, 1.2, 0.8, COLORS["light_green"], "Light\nOutput", fontsize=8
    )

    # Heavy Branch (右侧 - 深绿色)
    ax.text(
        10.5,
        9.5,
        "Heavy Branch",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["primary_green"],
    )

    # 重分支解码器
    for i in range(5):
        y_pos = 8.5 - i * 0.9
        color = COLORS["primary_green"]
        draw_block(ax, 9.8, y_pos, 1.3, 0.7, color, f"Dec-H{i + 1}", fontsize=7)

    # 重分支输出
    draw_block(
        ax, 9.8, 1.5, 1.3, 0.8, COLORS["dark_green"], "Heavy\nOutput", fontsize=8
    )

    # ============== Router ==============
    ax.text(
        13,
        7,
        "Difficulty\nRouter",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["accent_red"],
    )

    # Router 内部结构
    router_box = FancyBboxPatch(
        (12.3, 5.5),
        1.4,
        2.5,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#FFF5F5",
        edgecolor=COLORS["accent_red"],
        linewidth=2,
    )
    ax.add_patch(router_box)
    ax.text(13, 7.5, "MLP", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(
        13,
        6.8,
        "GAP Features",
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["gray"],
    )
    ax.text(
        13,
        6.3,
        "Confidence",
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["gray"],
    )
    ax.text(
        13, 5.8, "Entropy", ha="center", va="center", fontsize=7, color=COLORS["gray"]
    )

    # ============== 连接线 ==============
    # 输入到编码器
    ax.annotate(
        "",
        xy=(3.2, 9),
        xytext=(2, 9.6),
        arrowprops=dict(arrowstyle="->", color=COLORS["dark_green"], lw=2),
    )

    # 编码器到分支
    ax.annotate(
        "",
        xy=(5.5, 8.5),
        xytext=(4.2, 8.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["light_green"], lw=2),
    )
    ax.annotate(
        "",
        xy=(9.8, 8.5),
        xytext=(4.2, 8.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["primary_green"], lw=2),
    )

    # Router 到分支的决策线
    # Easy 路径 (绿色虚线)
    ax.annotate(
        "",
        xy=(7.2, 2),
        xytext=(12.8, 5),
        arrowprops=dict(
            arrowstyle="->", color=COLORS["light_green"], lw=1.5, linestyle="--"
        ),
    )
    ax.text(
        9,
        3.2,
        "Easy: Exit",
        fontsize=8,
        color=COLORS["light_green"],
        fontweight="bold",
        style="italic",
    )

    # Hard 路径 (红色虚线)
    ax.annotate(
        "",
        xy=(10.5, 2),
        xytext=(12.5, 5),
        arrowprops=dict(
            arrowstyle="->", color=COLORS["accent_red"], lw=1.5, linestyle="--"
        ),
    )
    ax.text(
        10.8,
        3.2,
        "Hard: Refine",
        fontsize=8,
        color=COLORS["accent_red"],
        fontweight="bold",
        style="italic",
    )

    # ============== 融合输出 ==============
    draw_block(
        ax, 7.5, 0.8, 1.5, 0.8, COLORS["dark_green"], "Final\nOutput", fontsize=9
    )

    # 从 light 和 heavy 到最终输出
    ax.annotate(
        "",
        xy=(8.25, 1.6),
        xytext=(6.7, 1.9),
        arrowprops=dict(arrowstyle="->", color=COLORS["dark_green"], lw=1.5),
    )
    ax.annotate(
        "",
        xy=(8.25, 1.6),
        xytext=(11.1, 1.9),
        arrowprops=dict(arrowstyle="->", color=COLORS["dark_green"], lw=1.5),
    )

    # ============== 图例 ==============
    legend_x = 0.5
    legend_y = 0.3

    # 编码器图例
    ax.add_patch(
        FancyBboxPatch(
            (legend_x, legend_y),
            0.3,
            0.25,
            boxstyle="round,pad=0.02",
            facecolor=COLORS["dark_green"],
        )
    )
    ax.text(legend_x + 0.4, legend_y + 0.12, "Encoder", fontsize=8, va="center")

    # 轻分支图例
    ax.add_patch(
        FancyBboxPatch(
            (legend_x + 1.5, legend_y),
            0.3,
            0.25,
            boxstyle="round,pad=0.02",
            facecolor=COLORS["pale_green"],
        )
    )
    ax.text(legend_x + 1.9, legend_y + 0.12, "Light Branch", fontsize=8, va="center")

    # 重分支图例
    ax.add_patch(
        FancyBboxPatch(
            (legend_x + 3.2, legend_y),
            0.3,
            0.25,
            boxstyle="round,pad=0.02",
            facecolor=COLORS["primary_green"],
        )
    )
    ax.text(legend_x + 3.6, legend_y + 0.12, "Heavy Branch", fontsize=8, va="center")

    # Router 图例
    ax.add_patch(
        FancyBboxPatch(
            (legend_x + 5, legend_y),
            0.3,
            0.25,
            boxstyle="round,pad=0.02",
            facecolor="#FFF5F5",
            edgecolor=COLORS["accent_red"],
        )
    )
    ax.text(legend_x + 5.4, legend_y + 0.12, "Router", fontsize=8, va="center")

    # ============== 推理模式说明 ==============
    mode_text = """
    Inference Modes:
    [1] Light: Only light branch (fastest)
    [2] Heavy: Full path (most accurate)  
    [3] Dynamic: Router decides (balanced)
    """
    ax.text(
        0.5,
        -0.3,
        mode_text,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        color=COLORS["dark_gray"],
        bbox=dict(
            boxstyle="round",
            facecolor=COLORS["light_gray"],
            edgecolor=COLORS["gray"],
            alpha=0.5,
        ),
    )

    plt.tight_layout()
    return fig


def main():
    """主函数"""
    fig = create_architecture_diagram()

    # 保存图片
    output_path = r"D:\Code\CV-practive\DRU-Net：面向资源有限环境下皮肤病灶分割\figures\dr_unet_architecture.png"
    fig.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Figure saved to: {output_path}")

    # 同时保存 PDF 版本（用于论文）
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(
        pdf_path, format="pdf", bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"PDF saved to: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()
