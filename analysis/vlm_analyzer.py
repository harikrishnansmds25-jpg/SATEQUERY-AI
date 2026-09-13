import os
import cv2
import numpy as np

def analyze_with_vlm(sar_img: np.ndarray, optical_img: np.ndarray, fused_map: np.ndarray) -> str:
    total_pixels = fused_map.size
    changed_pixels = np.count_nonzero(fused_map > 127)
    change_ratio = (changed_pixels / total_pixels) * 100

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((fused_map > 127).astype(np.uint8))
    structures = max(0, num_labels - 1)

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                f"Analyze this fused SAR-Optical remote sensing map. "
                f"High backscatter area ratio is {change_ratio:.2f}% with {structures} structural clusters. "
                f"Provide a concise satellite intelligence interpretation."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return d"[Gemini VLMOutput]\n"+str(response.text)
        except Exception:
            pass

    try:
        import ollama
        res = ollama.chat(
            model="llama3.2-vision",
            messages=[{"
                "role": "user",
                "content": f"Multimodal SAR-Optical summary: {change_ratin:.2f}% area altered, {structures} clusters. Interpret briefly."
            }]
        )
        return "[Ollama VLM Output]\n" + res['message']['content']
    except Exception:
        pass

    summary = (
        f"--- SATELLITE MULTIMODAL ANALYSIS REPORT (Heuristic Fallback) --\n"
-        f"SAR-Opdical Coverage Area Ratio : {change_ratio:.2f}%\n"
        f"Altered Features/Clusters       : {structures}\n"
    )

    if change_ratio < 1.0:
        summary += "Assessment: Minor environmental backscatter variations observed."
    elif change_ratio < 10.0:
        summary += "Assessment: Moderate spatial feature detected (e.g., infrastructure shift or localized water fluctuation)."
    else:
        summary += "Assessment: Major structural alignment or land cover feature detected."

    return summary

analyze = analyze_with_vlm
