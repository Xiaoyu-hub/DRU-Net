"""
后处理模块
用于推理阶段的后处理操作
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from typing import Optional, Tuple


def apply_crf(post_processed_img: torch.Tensor, output: torch.Tensor, 
              num_iterations: int = 5, spatial_weight: float = 3,
              bilateral_weight: float = 5) -> torch.Tensor:
    """
    使用条件随机场(CRF)进行后处理
    注意：需要安装 pydensecrf 库

    Args:
        post_processed_img: 原始图像 [B, C, H, W]
        output: 模型输出 [B, 1, H, W] (sigmoid probability)
        num_iterations: CRF 迭代次数
        spatial_weight: 空间核权重
        bilateral_weight: 双边核权重

    Returns:
        refined_output: 后处理后的输出
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        return output

    batch_size = output.shape[0]
    refined_outputs = []

    for i in range(batch_size):
        img = post_processed_img[i].cpu().numpy()
        probs = output[i, 0].cpu().numpy()

        c, h, w = probs.shape
        U = unary_from_softmax(probs)
        d = dcrf.DenseCRF2D(w, h, c)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=spatial_weight, compat=spatial_weight)
        d.addPairwiseBilateral(
            sxy=bilateral_weight, 
            srgb=50, 
            rgbim=img.transpose(1, 2, 0),
            compat=bilateral_weight
        )

        Q = d.inference(num_iterations)
        Q = np.array(Q).reshape((c, h, w))
        refined_outputs.append(torch.from_numpy(Q))

    return torch.stack(refined_outputs, dim=0).to(output.device)


def morphological_cleanup(mask: torch.Tensor, 
                         operation: str = "close",
                         kernel_size: int = 5) -> torch.Tensor:
    """
    使用形态学操作清理分割结果

    Args:
        mask: 二值掩码 [B, 1, H, W]
        operation: 操作类型 ("open", "close", "erode", "dilate")
        kernel_size: 卷积核大小

    Returns:
        cleaned_mask: 清理后的掩码
    """
    if not isinstance(mask, np.ndarray):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    batch_size = mask_np.shape[0]
    cleaned_masks = []

    struct = ndimage.generate_binary_structure(2, 1)

    for i in range(batch_size):
        m = mask_np[i, 0]

        if operation == "open":
            m = ndimage.binary_opening(m, structure=struct)
        elif operation == "close":
            m = ndimage.binary_closing(m, structure=struct)
        elif operation == "erode":
            m = ndimage.binary_erosion(m, structure=struct)
        elif operation == "dilate":
            m = ndimage.binary_dilation(m, structure=struct)

        cleaned_masks.append(m.astype(np.float32))

    return torch.from_numpy(np.stack(cleaned_masks, axis=0)).unsqueeze(1).to(mask.device)


def remove_small_regions(mask: torch.Tensor, 
                         min_size: int = 50) -> torch.Tensor:
    """
    移除小的连通区域

    Args:
        mask: 二值掩码 [B, 1, H, W]
        min_size: 最小区域大小（像素数）

    Returns:
        cleaned_mask: 移除小区域后的掩码
    """
    if not isinstance(mask, np.ndarray):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    batch_size = mask_np.shape[0]
    cleaned_masks = []

    for i in range(batch_size):
        m = mask_np[i, 0]

        labeled_array, num_features = ndimage.label(m)
        if num_features > 0:
            sizes = ndimage.sum(m, labeled_array, range(1, num_features + 1))
            mask_sizes = sizes < min_size
            remove_pixel = mask_sizes[labeled_array - 1]
            m[remove_pixel] = 0

        cleaned_masks.append(m.astype(np.float32))

    return torch.from_numpy(np.stack(cleaned_masks, axis=0)).unsqueeze(1).to(mask.device)


def test_time_augmentation(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    tta_transforms: Optional[list] = None
) -> torch.Tensor:
    """
    测试时增强 (TTA)

    Args:
        model: PyTorch 模型
        images: 输入图像 [B, C, H, W]
        device: 计算设备
        tta_transforms: TTA 变换列表

    Returns:
        averaged_output: 增强后的平均输出
    """
    if tta_transforms is None:
        tta_transforms = ["original", "hflip", "vflip", "rotate90"]

    outputs = []

    model.eval()
    with torch.no_grad():
        for transform in tta_transforms:
            aug_img = images.clone()

            if transform == "hflip":
                aug_img = torch.flip(aug_img, dims=[3])
            elif transform == "vflip":
                aug_img = torch.flip(aug_img, dims=[2])
            elif transform == "rotate90":
                aug_img = torch.rot90(aug_img, 1, [2, 3])

            output = model(aug_img.to(device))

            if isinstance(output, dict):
                output = output.get("final_output", output.get("light_output", output))

            if transform == "hflip":
                output = torch.flip(output, dims=[3])
            elif transform == "vflip":
                output = torch.flip(output, dims=[2])
            elif transform == "rotate90":
                output = torch.rot90(output, -1, [2, 3])

            outputs.append(output)

    averaged_output = torch.stack(outputs).mean(dim=0)
    return averaged_output


def apply_postprocessing(
    output: torch.Tensor,
    original_image: Optional[torch.Tensor] = None,
    use_tta: bool = False,
    use_morphological: bool = False,
    use_remove_small: bool = False,
    min_region_size: int = 50,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    应用完整的后处理流程

    Args:
        output: 模型输出 [B, 1, H, W] (sigmoid probability)
        original_image: 原始图像 [B, C, H, W] (用于 CRF)
        use_tta: 是否使用 TTA
        use_morphological: 是否使用形态学操作
        use_remove_small: 是否移除小区域
        min_region_size: 最小区域大小
        model: 模型 (用于 TTA)
        device: 设备

    Returns:
        processed_output: 后处理后的输出
    """
    processed = output.clone()

    if use_tta and model is not None:
        if device is None:
            device = output.device
        processed = test_time_augmentation(model, original_image, device)

    if use_remove_small:
        processed = (processed > 0.5).float()
        processed = remove_small_regions(processed, min_size=min_region_size)
        processed = processed.float()

    if use_morphological:
        processed = (processed > 0.5).float()
        processed = morphological_cleanup(processed, operation="close")
        processed = processed.float()

    return processed


class PostProcessor:
    """
    后处理器类
    封装完整的后处理流程
    """

    def __init__(
        self,
        use_tta: bool = False,
        use_morphological: bool = False,
        use_remove_small: bool = False,
        min_region_size: int = 50,
        tta_transforms: Optional[list] = None,
    ):
        self.use_tta = use_tta
        self.use_morphological = use_morphological
        self.use_remove_small = use_remove_small
        self.min_region_size = min_region_size
        self.tta_transforms = tta_transforms or ["original", "hflip", "vflip"]

    def __call__(
        self,
        output: torch.Tensor,
        original_image: Optional[torch.Tensor] = None,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return apply_postprocessing(
            output=output,
            original_image=original_image,
            use_tta=self.use_tta,
            use_morphological=self.use_morphological,
            use_remove_small=self.use_remove_small,
            min_region_size=self.min_region_size,
            model=model,
            device=device,
        )
