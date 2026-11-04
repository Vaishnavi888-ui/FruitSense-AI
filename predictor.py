"""
FruitSense AI - Predictor & Maturity Analysis Engine
Handles trained AI model inference when available, and provides an honest,
clearly-labeled computer-vision color spectrum baseline when no trained
model is connected.
"""

import os
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

# Maturity Categories
CATEGORIES = ["Unripe", "Semi-Ripe", "Ripe"]

# Supported model file extensions
SUPPORTED_EXTENSIONS = [".onnx", ".tflite", ".pt", ".pth", ".h5", ".keras"]


def check_model_status(models_dir: str = "models") -> Dict[str, Any]:
    """
    Checks if a trained model weight file is present in the models directory.
    Returns connection status and details.
    """
    if not os.path.exists(models_dir):
        return {
            "connected": False,
            "model_path": None,
            "model_type": None,
            "filename": None,
            "message": f"Directory '{models_dir}' not found.",
        }

    for file in os.listdir(models_dir):
        _, ext = os.path.splitext(file.lower())
        if ext in SUPPORTED_EXTENSIONS:
            return {
                "connected": True,
                "model_path": os.path.join(models_dir, file),
                "model_type": ext.replace(".", "").upper(),
                "filename": file,
                "message": f"Detected trained model: {file}",
            }

    return {
        "connected": False,
        "model_path": None,
        "model_type": None,
        "filename": None,
        "message": "No trained model weights found in 'models/' directory.",
    }


def analyze_fruit_color_spectrum(image: Image.Image) -> Dict[str, Any]:
    """
    Authentic computer-vision color-space analysis.
    Extracts real RGB and HSV color ratios from the fruit surface.
    
    This is NOT a fake AI score. It is an honest, rule-based baseline
    measuring peel chlorophyll (green) vs carotenoid (yellow/orange) pigmentation.
    """
    # Resize for fast, consistent pixel analysis
    img_rgb = image.convert("RGB").resize((224, 224))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0

    # Convert RGB to HSV
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    # Hue calculation (0 to 360 degrees)
    hue = np.zeros_like(c_max)
    delta_mask = delta > 1e-4

    # Red is max
    r_mask = delta_mask & (c_max == r)
    hue[r_mask] = (60.0 * (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6))

    # Green is max
    g_mask = delta_mask & (c_max == g)
    hue[g_mask] = (60.0 * (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2))

    # Blue is max
    b_mask = delta_mask & (c_max == b)
    hue[b_mask] = (60.0 * (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4))

    # Saturation and Value
    saturation = np.zeros_like(c_max)
    saturation[c_max > 1e-4] = delta[c_max > 1e-4] / c_max[c_max > 1e-4]
    value = c_max

    # Filter out obvious backgrounds (pure white/grey studio backgrounds or very dark shadows)
    fruit_mask = (value > 0.15) & ~((saturation < 0.15) & (value > 0.85))

    total_fruit_pixels = np.sum(fruit_mask)
    if total_fruit_pixels == 0:
        fruit_mask = np.ones_like(value, dtype=bool)
        total_fruit_pixels = fruit_mask.size

    # Segment color bands (Hue ranges)
    # Green: 65° to 155° (chlorophyll dominance -> Unripe)
    green_mask = fruit_mask & (hue >= 65.0) & (hue <= 155.0) & (saturation > 0.15)
    
    # Yellow / Golden: 35° to 65° (carotenoids developing -> Semi-Ripe to Ripe)
    yellow_mask = fruit_mask & (hue >= 35.0) & (hue < 65.0) & (saturation > 0.20)
    
    # Orange / Reddish blush: 10° to 35° (advanced maturity -> Ripe)
    orange_mask = fruit_mask & (hue >= 10.0) & (hue < 35.0) & (saturation > 0.25)

    green_count = np.sum(green_mask)
    yellow_count = np.sum(yellow_mask)
    orange_count = np.sum(orange_mask)

    green_pct = float((green_count / total_fruit_pixels) * 100.0)
    yellow_pct = float((yellow_count / total_fruit_pixels) * 100.0)
    orange_pct = float((orange_count / total_fruit_pixels) * 100.0)
    other_pct = max(0.0, 100.0 - (green_pct + yellow_pct + orange_pct))

    # Determine stage based on surface color ratios
    if (yellow_pct + orange_pct >= 60.0) or (orange_pct >= 25.0 and green_pct < 25.0):
        stage = "Ripe"
        dominant_feature = f"Carotenoid yellow/orange pigments dominate {yellow_pct + orange_pct:.1f}% of surface."
    elif green_pct >= 65.0 and (yellow_pct + orange_pct < 25.0):
        stage = "Unripe"
        dominant_feature = f"Green chlorophyll peel dominates {green_pct:.1f}% of detected fruit surface with minimal breaking color."
    else:
        stage = "Semi-Ripe"
        dominant_feature = f"Breaking color stage: {yellow_pct:.1f}% yellow transition with {green_pct:.1f}% residual green."

    return {
        "stage": stage,
        "green_pct": round(green_pct, 1),
        "yellow_pct": round(yellow_pct, 1),
        "orange_pct": round(orange_pct, 1),
        "other_pct": round(other_pct, 1),
        "dominant_feature": dominant_feature,
    }


def assess_mango(image: Image.Image, models_dir: str = "models") -> Dict[str, Any]:
    """
    Main assessment entry point.
    - If a trained AI model is connected, runs real inference and provides actual confidence.
    - If no model is connected, runs authentic computer-vision color baseline and flags as prototype.
    """
    status = check_model_status(models_dir)

    if status["connected"]:
        # A trained model file exists. Attempt inference.
        # NOTE: If user provides PyTorch, ONNX, or Keras weights, inference happens here.
        try:
            # Placeholder for user model inference hook
            # If the user has a real onnx / tflite / pt file, load and score it
            return _run_trained_model(image, status)
        except Exception as e:
            # If model file fails to load or corrupt, fall back with honest error
            color_data = analyze_fruit_color_spectrum(image)
            return {
                "stage": color_data["stage"],
                "is_deep_learning": False,
                "model_status": {
                    "connected": False,
                    "message": f"Model file found but failed to load ({str(e)}). Fallback to color baseline.",
                },
                "confidence_score": None,  # Rule: No fake confidence score
                "confidence_display": "N/A (Prototype Mode)",
                "color_breakdown": color_data,
                "assessment_method": "Color-Space Baseline (Fallback due to model load error)",
            }

    # No model connected: Run authentic color-space heuristic
    color_data = analyze_fruit_color_spectrum(image)

    return {
        "stage": color_data["stage"],
        "is_deep_learning": False,
        "model_status": status,
        "confidence_score": None,  # NEVER fake a confidence score
        "confidence_display": "N/A (Prototype Baseline)",
        "color_breakdown": color_data,
        "assessment_method": "Computer-Vision Color Baseline (Prototype Heuristic)",
    }


def _run_trained_model(image: Image.Image, status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes actual inference with a connected trained model.
    Only called when a real model file is present in models/.
    """
    model_path = status["model_path"]
    model_type = status["model_type"]

    # When onnxruntime is available and model is ONNX
    if model_type == "ONNX":
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # Preprocess
        img = image.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        arr = np.expand_dims(arr, axis=0)   # (1, C, H, W)

        raw_out = session.run(None, {input_name: arr})[0][0]
        # Softmax
        exp_out = np.exp(raw_out - np.max(raw_out))
        probs = exp_out / np.sum(exp_out)

        idx = int(np.argmax(probs))
        stage = CATEGORIES[idx] if idx < len(CATEGORIES) else "Unknown"
        confidence = float(probs[idx] * 100.0)

        return {
            "stage": stage,
            "is_deep_learning": True,
            "model_status": status,
            "confidence_score": round(confidence, 1),
            "confidence_display": f"{confidence:.1f}%",
            "class_probabilities": {
                CATEGORIES[i]: round(float(probs[i] * 100.0), 1)
                for i in range(min(len(CATEGORIES), len(probs)))
            },
            "color_breakdown": analyze_fruit_color_spectrum(image),
            "assessment_method": f"Deep Learning Inference ({status['filename']})",
        }

    # Other formats: alert that framework loader is ready
    raise NotImplementedError(
        f"Framework loader for '{model_type}' is awaiting framework installation. "
        "Recommend exporting trained model to ONNX for lightweight Streamlit deployment."
    )


def get_stage_guidance(stage: str) -> Dict[str, Any]:
    """
    Returns actionable agricultural post-harvest advice and storage guidance.
    """
    guidance_map = {
        "Unripe": {
            "theme_color": "#2D6A4F",
            "badge_bg": "#D8F3DC",
            "icon": "🟢",
            "shelf_life": "7 to 14 days",
            "storage_temp": "12°C to 14°C (Cool, dry ventilated space)",
            "chilling_risk": "High! Do NOT store below 10°C to prevent chilling injury (pitting, uneven ripening).",
            "commercial_action": "Ideal for long-distance transport, wholesale shipping, or controlled ripening chambers.",
            "culinary_use": "Green salads, pickling, raw chutneys, or allow ripening at room temperature.",
        },
        "Semi-Ripe": {
            "theme_color": "#D97706",
            "badge_bg": "#FEF3C7",
            "icon": "🟡",
            "shelf_life": "3 to 5 days",
            "storage_temp": "20°C to 24°C (Ambient room temperature)",
            "chilling_risk": "Moderate. Keep away from direct sunlight; allow natural ethylene accumulation.",
            "commercial_action": "Optimal for retail supermarket shelves; will reach peak table ripeness in 48 to 72 hours.",
            "culinary_use": "Mildly sweet fresh snacks, semi-firm slices, or short ripening hold.",
        },
        "Ripe": {
            "theme_color": "#EA580C",
            "badge_bg": "#FFEDD5",
            "icon": "🟠",
            "shelf_life": "1 to 3 days",
            "storage_temp": "8°C to 10°C (Refrigeration recommended to prevent over-ripening)",
            "chilling_risk": "Low. Fruit has achieved full sugar conversion and sweet aroma.",
            "commercial_action": "Immediate consumer sale, pulp extraction, or processing into mango nectar and desserts.",
            "culinary_use": "Ready to eat! Maximum sweetness, rich flavor, and soft juicy texture.",
        },
    }
    return guidance_map.get(stage, guidance_map["Semi-Ripe"])
