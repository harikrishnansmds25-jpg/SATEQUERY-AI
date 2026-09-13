"""
Stage 6: RESULT GENERATION
Packages the fusion composite, the VLM's answer, and a confidence note
into the final output shown to the user.
"""

import numpy as np


def generate_result(composite_img: np.ndarray, answer: str, fusion_method: str) -> dict:
    """Builds the final response dict.

    A confidence note is included because fusion via a composite image
    is an approximation, not true feature-level multimodal fusion - it's
    honest to flag this rather than present the answer as definitive.
    """
    composite_uint8 = (np.clip(composite_img, 0, 1) * 255).astype(np.uint8)

    confidence_note = (
        f"Answer generated from a {fusion_method} SAR+optical composite. "
        "This is an approximate fusion method suitable for a prototype; "
        "production systems would use feature-level fusion trained on "
        "paired SAR/optical data for higher reliability."
    )

    return {
        "composite_image": composite_uint8,
        "answer": answer,
        "confidence_note": confidence_note,
    }
