"""
MAIN PIPELINE
Single entry point: run_sar_optical_analysis(optical_img, sar_img, query) -> dict

This is the function your team's router agent calls when it decides
the user's query needs the "optical+SAR joint analysis" branch.
"""

import numpy as np

from quality.checker import run_quality_gate
from preprocessing.prep import preprocess_pair
from fusion.fuse import fuse
from analysis.vlm_analyzer import analyze
from output.result_gen import generate_result


def run_sar_optical_analysis(optical_img: np.ndarray, sar_img: np.ndarray, query: str,
                              fusion_method: str = "channel_replace",
                              vlm_call=None) -> dict:
    """
    Args:
        optical_img: raw HxWx3 uint8 optical image.
        sar_img: raw HxW uint8 SAR image.
        query: the user's natural language question.
        fusion_method: "channel_replace" | "alpha_blend" | "edge_overlay"
        vlm_call: function(image: np.ndarray, prompt: str) -> str,
                  the team's shared VLM call. If None, a heuristic
                  fallback answer is used instead.

    Returns:
        On success:
            {
                "success": True,
                "composite_image": np.ndarray,
                "answer": str,
                "confidence_note": str,
            }
        On quality failure:
            {
                "success": False,
                "reason": str,
            }
    """
    # Stage 2: quality and alignment check
    quality = run_quality_gate(optical_img, sar_img)
    if not quality["passed"]:
        return {"success": False, "reason": quality["reason"]}

    # Stage 3: modality-specific preprocessing
    optical_norm, sar_norm = preprocess_pair(optical_img, sar_img)

    # Stage 4: fusion
    composite = fuse(optical_norm, sar_norm, method=fusion_method)

    # Stage 5: joint analysis
    answer = analyze(composite, optical_norm, sar_norm, query, vlm_call=vlm_call)

    # Stage 6: result generation
    result = generate_result(composite, answer, fusion_method)

    return {
        "success": True,
        "composite_image": result["composite_image"],
        "answer": result["answer"],
        "confidence_note": result["confidence_note"],
    }


if __name__ == "__main__":
    from data.loader import make_synthetic_pair

    optical, sar = make_synthetic_pair()
    result = run_sar_optical_analysis(optical, sar, query="What structures are visible in this area?")

    if result["success"]:
        print("Answer:", result["answer"])
        print("Confidence note:", result["confidence_note"])
    else:
        print("Rejected:", result["reason"])
