"""
Stage 2: QUALITY AND ALIGNMENT CHECK
Checks that the optical image isn't too cloudy, the SAR image isn't
excessively noisy, and both images match in size (co-registered).
"""

import cv2
import numpy as np


def check_alignment(optical_img: np.ndarray, sar_img: np.ndarray) -> tuple[bool, str]:
    """Checks the two images cover the same pixel grid size. A production
    system would also check geolocation metadata (lat/lon bounds match),
    but shape matching catches the most common demo-time failure.
    """
    if optical_img.shape[:2] != sar_img.shape[:2]:
        return False, f"Size mismatch: optical {optical_img.shape[:2]} vs SAR {sar_img.shape[:2]}"
    return True, "ok"


def check_cloud_cover(optical_img: np.ndarray, brightness_threshold: int = 200,
                       max_cloud_ratio: float = 0.4) -> tuple[bool, float]:
    """Same heuristic as the change detection quality gate: very bright,
    low-saturation pixels are likely cloud.
    """
    hsv = cv2.cvtColor(optical_img, cv2.COLOR_RGB2HSV)
    bright_mask = hsv[:, :, 2] > brightness_threshold
    low_sat_mask = hsv[:, :, 1] < 40
    cloud_mask = bright_mask & low_sat_mask
    cloud_ratio = cloud_mask.mean()
    return cloud_ratio <= max_cloud_ratio, cloud_ratio


def check_sar_noise(sar_img: np.ndarray, max_std_threshold: float = 90.0) -> tuple[bool, float]:
    """Rough SAR quality heuristic: extremely high pixel variance can
    indicate corrupted or unusable SAR data (as opposed to normal speckle,
    which is expected and fine).
    """
    std = float(sar_img.astype(np.float32).std())
    return std <= max_std_threshold, std


def run_quality_gate(optical_img: np.ndarray, sar_img: np.ndarray) -> dict:
    """Runs all checks and returns a single verdict dict, same shape as
    the change detection quality gate for consistency across branches.
    """
    aligned_ok, aligned_msg = check_alignment(optical_img, sar_img)
    if not aligned_ok:
        return {"passed": False, "reason": aligned_msg, "details": {}}

    cloud_ok, cloud_ratio = check_cloud_cover(optical_img)
    sar_ok, sar_std = check_sar_noise(sar_img)

    details = {"cloud_ratio": cloud_ratio, "sar_std": sar_std}

    if not cloud_ok:
        return {"passed": False, "reason": "Optical image too cloudy", "details": details}
    if not sar_ok:
        return {"passed": False, "reason": "SAR image too noisy/corrupted", "details": details}

    return {"passed": True, "reason": None, "details": details}


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import make_synthetic_pair

    optical, sar = make_synthetic_pair()
    print(run_quality_gate(optical, sar))
