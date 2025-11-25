import io
import logging
from typing import Dict, Any

import numpy as np
from PIL import Image
from deepface import DeepFace

logger = logging.getLogger(__name__)


def _compute_facial_risk(emotion_scores: Dict[str, float]) -> float:
    """
    Turn DeepFace emotion scores into a 0–1 facial risk score.

    Heuristic:
    - Negative emotions: sad, fear, angry, disgust → increase risk
    - Neutral / surprise → mild contribution
    - Happy → reduces risk slightly
    """
    if not emotion_scores:
        return 0.0

    # DeepFace usually returns percentages summing to ≈100
    total_raw = sum(float(v) for v in emotion_scores.values()) or 1.0

    neg_keys = {"sad", "fear", "angry", "disgust"}
    mild_keys = {"neutral", "surprise"}
    pos_keys = {"happy"}

    neg_sum = sum(float(v) for k, v in emotion_scores.items() if k.lower() in neg_keys)
    mild_sum = sum(float(v) for k, v in emotion_scores.items() if k.lower() in mild_keys)
    pos_sum = sum(float(v) for k, v in emotion_scores.items() if k.lower() in pos_keys)

    # Weighted combination: negatives most important
    raw_risk = (neg_sum * 1.0 + mild_sum * 0.4 - pos_sum * 0.3) / total_raw
    # Clamp to [0, 1]
    raw_risk = max(0.0, min(1.0, raw_risk))

    # Soft floor so "slightly tense" faces still show small risk
    return float(raw_risk)


def _facial_stress_label(face_risk: float) -> str:
    if face_risk >= 0.75:
        return "Very High Facial Stress"
    elif face_risk >= 0.55:
        return "High Facial Stress"
    elif face_risk >= 0.35:
        return "Moderate Facial Stress"
    elif face_risk > 0.0:
        return "Mild Facial Stress"
    else:
        return "Low Facial Stress"


def analyze_face_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Analyze a single webcam snapshot and return facial emotion / stress metrics.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes from Streamlit's st.camera_input().getvalue()

    Returns
    -------
    dict with keys:
        - dominant_emotion: str
        - facial_stress_level: str
        - face_risk: float (0–1)
        - emotion_scores: dict[str, float]
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"[CAMERA] Failed to decode image bytes: {e}")
        return {
            "dominant_emotion": "Unknown",
            "facial_stress_level": "Image Decode Error",
            "face_risk": 0.0,
            "emotion_scores": {},
        }

    img_array = np.array(img)

    try:
        analysis = DeepFace.analyze(
            img_array,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
        )
    except Exception as e:
        logger.error(f"[CAMERA] DeepFace analysis failed: {e}")
        return {
            "dominant_emotion": "Analysis Failed",
            "facial_stress_level": "Model Error",
            "face_risk": 0.0,
            "emotion_scores": {},
        }

    # DeepFace can return list[dict] or dict
    if isinstance(analysis, list):
        analysis = analysis[0] if analysis else {}

    dominant = analysis.get("dominant_emotion", "neutral")
    emotions = analysis.get("emotion", {}) or {}

    # Normalize keys to Title Case for UI, but keep numeric values
    formatted_emotions = {str(k).title(): float(v) for k, v in emotions.items()}

    face_risk = _compute_facial_risk(emotions)
    stress_label = _facial_stress_label(face_risk)

    return {
        "dominant_emotion": str(dominant).title(),
        "facial_stress_level": stress_label,
        "face_risk": float(face_risk),
        "emotion_scores": formatted_emotions,
    }


if __name__ == "__main__":
    # Simple manual test placeholder
    print("camera_detector.py ready. Integrate with Streamlit via analyze_face_image().")


# ============================
# Added Missing Risk-Level Function
# ============================

def get_risk_level(score: float) -> str:
    """Return categorical risk level for a given 0–1 numeric score."""
    if score >= 0.75:
        return "Severe Risk"
    elif score >= 0.55:
        return "High Risk"
    elif score >= 0.35:
        return "Moderate Risk"
    elif score > 0.0:
        return "Mild Risk"
    return "Low Risk"
