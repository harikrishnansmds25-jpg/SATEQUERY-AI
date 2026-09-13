"""
Stage 3: MODALITY-SPECIFIC PREPROCESSING
Optical and SAR need different treatment before fusion:
  - SAR: despeckle (median filter) then normalize
  - Optical: standard resize + normalize
Both are brought to the same spatial size so fusion can align them
pixel-for-pixel.
"""

import cv2
import numpy as np


def preprocess_optical(optical_img: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize and normalize the optical image to [0, 1] float."""
    resized = cv2.resize(optical_img, target_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def despeckle_sar(sar_img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median filtering is a simple, standard first-pass despeckling
    method for SAR - real pipelines often use Lee or Frost filters, but
    median filtering is fast, dependency-free, and good enough to clean
    up visible speckle for a demo.
    """
    return cv2.medianBlur(sar_img, kernel_size)


def preprocess_sar(sar_img: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Despeckle, resize, and normalize the SAR image to [0, 1] float."""
    despeckled = despeckle_sar(sar_img)
    resized = cv2.resize(despeckled, target_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def preprocess_pair(optical_img: np.ndarray, sar_img: np.ndarray,
                     target_size: tuple[int, int] = (256, 256)) -> tuple[np.ndarray, np.ndarray]:
    """Full Stage 3 entry point. Returns (optical_norm, sar_norm), both
    float32 in [0, 1], same H and W.
    """
    optical_norm = preprocess_optical(optical_img, target_size)
    sar_norm = preprocess_sar(sar_img, target_size)
    return optical_norm, sar_norm


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import make_synthetic_pair

    optical, sar = make_synthetic_pair()
    optical_p, sar_p = preprocess_pair(optical, sar)
    print("Optical:", optical_p.shape, optical_p.dtype, "SAR:", sar_p.shape, sar_p.dtype)
