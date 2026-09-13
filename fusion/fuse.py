"""
Stage 4: FUSION
Combines the preprocessed optical (HxWx3) and SAR (HxW) images into a
single composite RGB-shaped image that a standard vision-language model
can consume - VLMs are trained on natural photos, not raw SAR, so we
never feed SAR in directly. Instead we encode SAR structure into a
channel of an RGB-shaped image.

Three fusion strategies are provided:
  - channel_replace_fusion : simplest, most literal (good default)
  - alpha_blend_fusion     : SAR overlaid as a soft highlight
  - edge_overlay_fusion    : SAR edges drawn onto optical (highlights
                              structures like buildings that SAR is
                              especially good at detecting)
"""

import cv2
import numpy as np


def channel_replace_fusion(optical_norm: np.ndarray, sar_norm: np.ndarray) -> np.ndarray:
    """Replaces the optical image's red channel with the SAR intensity.
    SAR is especially sensitive to man-made structures (buildings, roads,
    ships), so boosting it into a visible channel surfaces that
    information directly to the VLM.

    Inputs: optical_norm HxWx3 float32 [0,1], sar_norm HxW float32 [0,1]
    Returns: HxWx3 float32 [0,1] composite
    """
    composite = optical_norm.copy()
    composite[:, :, 0] = sar_norm  # red channel <- SAR
    return composite


def alpha_blend_fusion(optical_norm: np.ndarray, sar_norm: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Soft overlay: blends SAR (shown as a grayscale tint) on top of the
    optical image. Keeps more of the optical image's natural appearance
    than channel replacement, useful when you want the VLM to still
    clearly recognize normal visual features.
    """
    sar_rgb = np.stack([sar_norm] * 3, axis=-1)
    return (1 - alpha) * optical_norm + alpha * sar_rgb


def edge_overlay_fusion(optical_norm: np.ndarray, sar_norm: np.ndarray) -> np.ndarray:
    """Extracts edges from the SAR image (where structures like building
    outlines are often sharper in SAR than in optical) and draws them in
    a bright color on top of the optical image. This highlights exactly
    the kind of structural detail SAR contributes that optical alone
    would miss.
    """
    sar_uint8 = (sar_norm * 255).astype(np.uint8)
    edges = cv2.Canny(sar_uint8, threshold1=50, threshold2=150)

    composite = (optical_norm * 255).astype(np.uint8).copy()
    composite[edges > 0] = [255, 255, 0]  # highlight SAR-derived edges in yellow
    return composite.astype(np.float32) / 255.0


def fuse(optical_norm: np.ndarray, sar_norm: np.ndarray, method: str = "channel_replace") -> np.ndarray:
    """Single entry point Stage 4 exposes to the pipeline.

    method: "channel_replace" | "alpha_blend" | "edge_overlay"
    """
    if method == "channel_replace":
        return channel_replace_fusion(optical_norm, sar_norm)
    elif method == "alpha_blend":
        return alpha_blend_fusion(optical_norm, sar_norm)
    elif method == "edge_overlay":
        return edge_overlay_fusion(optical_norm, sar_norm)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import make_synthetic_pair
    from preprocessing.prep import preprocess_pair

    optical, sar = make_synthetic_pair()
    optical_p, sar_p = preprocess_pair(optical, sar)

    for method in ["channel_replace", "alpha_blend", "edge_overlay"]:
        composite = fuse(optical_p, sar_p, method=method)
        print(f"{method}: shape={composite.shape}, range=({composite.min():.2f}, {composite.max():.2f})")
