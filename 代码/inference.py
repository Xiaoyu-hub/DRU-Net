"""
推理脚本
支持多种推理模式：轻分支 / 重分支 / 动态路由
支持后处理：TTA、形态学清理、小区域移除
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from omegaconf import OmegaConf
from models.dr_unet import DRUNet
from utils.postprocess import PostProcessor


def load_model(cfg, checkpoint_path: str, device: torch.device) -> DRUNet:
    """加载模型"""
    model = DRUNet(cfg, mode="inference")

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(
            f"Warning: No checkpoint found at {checkpoint_path}, using random weights"
        )

    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, target_size=(256, 256)) -> torch.Tensor:
    """预处理图像"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_array = (img_array - mean) / std

    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def postprocess_output(output: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """将模型输出转换为二值掩码"""
    if output.dim() == 4:
        output = output.squeeze(1)
    prob = torch.sigmoid(output)
    mask = (prob > threshold).cpu().numpy().astype(np.uint8)
    return mask


def run_inference(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    inference_mode: str = "dynamic",
    use_postprocess: bool = False,
    postprocessor: Optional[PostProcessor] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """运行推理"""
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        
        # 路由决策统计
        if isinstance(outputs, dict):
            # 检查路由决策
            if "is_hard" in outputs and outputs["is_hard"] is not None:
                is_hard = outputs["is_hard"].cpu().numpy()
                hard_count = is_hard.sum()
                total = is_hard.size
                hard_ratio = hard_count / total
                
                if verbose:
                    print(f"[Router Decision] Hard Ratio: {hard_ratio:.2%} ({hard_count}/{total})")
            
            # 确定最终输出
            if "final_output" in outputs:
                final_output = outputs["final_output"]
            elif inference_mode == "heavy":
                final_output = outputs.get("heavy_output", outputs.get("light_output"))
            elif inference_mode == "light":
                final_output = outputs.get("light_output", outputs.get("heavy_output"))
            else:  # dynamic
                # 计算used_branch列表
                is_hard = outputs.get("is_hard")
                if is_hard is not None and self.use_router:
                    is_hard_tensor = is_hard.cpu()
                    batch_size = is_hard_tensor.size(0)
                    used_branch = []
                    for i in range(batch_size):
                        if is_hard_tensor[i] > 0:
                            used_branch.append("heavy")
                        else:
                            used_branch.append("light")
                    outputs["used_branch"] = used_branch
                    final_output = outputs.get("final_output", outputs.get("light_output"))
                else:
                    final_output = outputs.get("final_output", outputs.get("light_output"))
        else:
            final_output = outputs

        if use_postprocess and postprocessor is not None:
            final_output = postprocessor(
                final_output, original_image=image, model=model, device=device
            )

    return final_output


def main():
    parser = argparse.ArgumentParser(description="DR2-UNet Inference")
    parser.add_argument(
        "--config", type=str, default="config/base.yaml", help="Config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--output", type=str, default="output.png", help="Output mask path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=["light", "heavy", "dynamic"],
        help="Inference mode",
    )
    parser.add_argument("--use_tta", action="store_true", help="Use TTA")
    parser.add_argument(
        "--use_morphological", action="store_true", help="Use morphological cleanup"
    )
    parser.add_argument(
        "--use_remove_small", action="store_true", help="Remove small regions"
    )
    parser.add_argument(
        "--min_region_size", type=int, default=50, help="Minimum region size"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)

    model = load_model(cfg, args.checkpoint, device)
    model.inference_mode = args.mode

    postprocessor = None
    if args.use_tta or args.use_morphological or args.use_remove_small:
        postprocessor = PostProcessor(
            use_tta=args.use_tta,
            use_morphological=args.use_morphological,
            use_remove_small=args.use_remove_small,
            min_region_size=args.min_region_size,
        )

    image_tensor = preprocess_image(args.image)

    output = run_inference(
        model,
        image_tensor,
        device,
        inference_mode=args.mode,
        use_postprocess=postprocessor is not None,
        postprocessor=postprocessor,
        verbose=args.verbose,
    )

    mask = postprocess_output(output)

    mask_img = Image.fromarray(mask.squeeze() * 255)
    mask_img.save(args.output)
    
    # 推理结果摘要
    total_pixels = mask.size
    foreground_pixels = mask.sum()
    foreground_ratio = foreground_pixels / total_pixels
    
    print(f"
[Inference Summary]")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Output Path: {args.output}")
    print(f"  Foreground Ratio: {foreground_ratio:.2%} ({foreground_pixels}/{total_pixels})")
    
    # 分支使用统计
    if "router" in args.mode or "dynamic" in args.mode:
        if output.get("used_branch"):
            if isinstance(output["used_branch"], list):
                light_count = sum(1 for b in output["used_branch"] if b == "light")
                heavy_count = len(output["used_branch"]) - light_count
            else:
                light_count = output["used_branch"].count("light")
                heavy_count = output["used_branch"].count("heavy")
            
            print(f"  Router: Enabled")
            print(f"  Branch Usage: Light={light_count}, Heavy={heavy_count}")
            if light_count > 0:
                print(f"  Early Exit Rate: {light_count / len(output['used_branch']) * 100:.1f}%")
        else:
            print(f"  Router: Enabled (use --verbose for details)")

    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()
