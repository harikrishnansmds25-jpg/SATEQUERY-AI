"""
Stage 1: INPUT
Loads a co-registered optical (Sentinel-2 style RGB) + SAR (Sentinel-1
style grayscale) image pair of the same location.
"""

import numpy as np
from PIL import Image


def load_pair_from_files(optical_path: str, sar_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a real optical/SAR pair from two local image files.

    Returns:
        (optical_img, sar_img):
            optical_img -> HxWx3 uint8 RGB array
            sar_img     -> HxW uint8 single-channel array
    """
    optical_img = np.array(Image.open(optical_path).convert("RGB"))
    sar_img = np.array(Image.open(sar_path).convert("L"))
    return optical_img, sar_img


def make_synthetic_pair(size: int = 256, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic optical + SAR pair for testing the pipeline
    before you have real Sentinel-1/2 imagery.

    - optical: smooth RGB terrain-like image with a bright rectangular
      "building" region.
    - sar: grayscale image with realistic speckle noise, where the same
      structure appears as a bright return (SAR is sensitive to man-made
      structures) so the fusion step has something meaningful to combine.
    """
    rng = np.random.default_rng(seed)

    # optical: smooth base + a visible structure
    optical = rng.integers(60, 140, size=(size, size, 3), dtype=np.uint8)
    x0, y0, x1, y1 = size // 3, size // 3, size // 3 + 50, size // 3 + 70
    optical[y0:y1, x0:x1] = [180, 170, 150]  # looks like a rooftop

    # sar: speckled grayscale, same structure shows as a strong bright return
    sar_base = rng.integers(20, 60, size=(size, size), dtype=np.uint8)
    speckle = (rng.random((size, size)) * 30).astype(np.uint8)
    sar = np.clip(sar_base.astype(int) + speckle, 0, 255).astype(np.uint8)
    sar[y0:y1, x0:x1] = np.clip(200 + speckle[y0:y1, x0:x1], 0, 255).astype(np.uint8)

    return optical, sar


if __name__ == "__main__":
    optical, sar = make_synthetic_pair()
    print("Optical shape:", optical.shape, "SAR shape:", sar.shape)
