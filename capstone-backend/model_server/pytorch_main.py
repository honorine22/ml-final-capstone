from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import timm
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
EXPORT_DIR = BASE_DIR / "model_exports"
DEFAULT_MODEL_PATH = EXPORT_DIR / "maizeguard_public_best_model.pt"
DEFAULT_METADATA_PATH = EXPORT_DIR / "maizeguard_model_metadata.json"
MODEL_PATH = Path(os.getenv("PYTORCH_MODEL_PATH", DEFAULT_MODEL_PATH))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", DEFAULT_METADATA_PATH))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="MaizeGuard PyTorch Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3006,https://maizeguard-frontend.vercel.app/").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    metadata = {**metadata, **checkpoint.get("config", {})}
    state_dict = checkpoint["state_dict"]
    CLASS_NAMES = checkpoint.get("classes") or metadata.get("class_names") or metadata.get("classes")
    MODEL_NAME = checkpoint.get("model_name") or metadata.get("model_name") or metadata.get("model_backbone", "mobilenetv3_large_100")
    IMG_SIZE = int(checkpoint.get("img_size") or metadata.get("image_size") or metadata.get("input_image_size", 224))
else:
    state_dict = checkpoint
    CLASS_NAMES = metadata.get("class_names") or metadata.get("classes")
    MODEL_NAME = metadata.get("model_name") or metadata.get("model_backbone", "mobilenetv3_large_100")
    IMG_SIZE = int(metadata.get("image_size") or metadata.get("input_image_size", 224))

MEAN = metadata.get("normalization_mean", [0.485, 0.456, 0.406])
STD = metadata.get("normalization_std", [0.229, 0.224, 0.225])
SAFETY_RULE = metadata.get("deployment_safety_rule", {})
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", SAFETY_RULE.get("needs_review_confidence_below", 0.65)))
TOP2_MARGIN_THRESHOLD = float(os.getenv("TOP2_MARGIN_THRESHOLD", SAFETY_RULE.get("needs_review_top2_margin_below", 0.15)))
MIXED_RISK_REVIEW_THRESHOLD = float(os.getenv("MIXED_RISK_REVIEW_THRESHOLD", SAFETY_RULE.get("mixed_risk_review_threshold", 0.55)))
BATCH_INFERENCE_ENABLED = os.getenv("BATCH_INFERENCE_ENABLED", "true").lower() != "false"
BATCH_TILE_MIN_SIDE = int(os.getenv("BATCH_TILE_MIN_SIDE", "360"))
BATCH_TILE_SCALE = float(os.getenv("BATCH_TILE_SCALE", "0.62"))
BATCH_TILE_RISK_WEIGHT = float(os.getenv("BATCH_TILE_RISK_WEIGHT", "0.35"))

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

valid_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def recommendation_for(label: str, needs_review: bool):
    if needs_review:
        return {
            "risk": "Needs review",
            "action": "Needs review",
            "recommendation": "Retake the photo closer to the maize sample on a clear surface.",
        }

    mapping = {
        "good": {
            "risk": "Low",
            "action": "Store safely or prepare for sale",
            "recommendation": "The maize appears clean. Store in a dry place and monitor normally.",
        },
        "broken": {
            "risk": "Medium",
            "action": "Sort before storage",
            "recommendation": "Remove visibly broken or damaged kernels before storage or sale.",
        },
        "impurity": {
            "risk": "Medium",
            "action": "Clean and re-screen",
            "recommendation": "Remove foreign materials such as stones, husks, dust, or debris.",
        },
        "mold_risk": {
            "risk": "High",
            "action": "Do not store; refer for checking",
            "recommendation": "Possible visible mold-risk evidence needs careful checking before storage or consumption.",
        },
    }
    return mapping.get(
        label,
        {
            "risk": "Medium",
            "action": "Needs review",
            "recommendation": "Could not map prediction safely.",
        },
    )


def needs_review(probabilities: np.ndarray, confidence: float) -> bool:
    sorted_probs = np.sort(probabilities)[::-1]
    top2_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
    return confidence < CONFIDENCE_THRESHOLD or top2_margin < TOP2_MARGIN_THRESHOLD


def mixed_risk_reason(probabilities: np.ndarray, raw_label: str) -> Optional[str]:
    if raw_label != "good":
        return None

    risk_scores = []
    for risk_label in ["mold_risk", "impurity", "broken"]:
        if risk_label in CLASS_NAMES:
            index = CLASS_NAMES.index(risk_label)
            risk_scores.append((risk_label, float(probabilities[index])))

    if not risk_scores:
        return None

    risk_label, risk_score = max(risk_scores, key=lambda item: item[1])
    if risk_score >= MIXED_RISK_REVIEW_THRESHOLD:
        readable_label = risk_label.replace("_", " ")
        return (
            f"The top class is good, but {readable_label} also has "
            f"{round(risk_score * 100, 1)}% probability. Ask for manual review before storage."
        )

    return None


def make_batch_views(image: Image.Image):
    """Use full image plus large tiles for batch photos.

    CK-CNNLW's original idea is segmentation followed by single-object
    classification. For a lightweight capstone API, large tile views approximate
    that object-aware behavior without running an old Mask R-CNN stack.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    views = [("full", rgb)]

    if not BATCH_INFERENCE_ENABLED or min(width, height) < BATCH_TILE_MIN_SIDE:
        return views

    tile_size = int(min(width, height) * BATCH_TILE_SCALE)
    if tile_size < BATCH_TILE_MIN_SIDE:
        return views

    boxes = [
        ("center", ((width - tile_size) // 2, (height - tile_size) // 2)),
        ("top_left", (0, 0)),
        ("top_right", (width - tile_size, 0)),
        ("bottom_left", (0, height - tile_size)),
        ("bottom_right", (width - tile_size, height - tile_size)),
    ]

    for name, (left, top) in boxes:
        right = min(left + tile_size, width)
        bottom = min(top + tile_size, height)
        left = max(right - tile_size, 0)
        top = max(bottom - tile_size, 0)
        if right > left and bottom > top:
            views.append((name, rgb.crop((left, top, right, bottom))))

    return views


@torch.no_grad()
def predict_views(image: Image.Image):
    views = make_batch_views(image)
    probabilities = []

    for _, view in views:
        tensor = valid_tfms(view).unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        probabilities.append(torch.softmax(logits, dim=1)[0].cpu().numpy())

    stacked = np.stack(probabilities, axis=0)
    avg_probs = stacked.mean(axis=0)

    # If a risky class is clearly visible in one tile, gently lift that signal
    # without allowing one noisy tile to dominate the full-image prediction.
    max_probs = stacked.max(axis=0)
    for risk_label in ["mold_risk", "impurity", "broken"]:
        if risk_label in CLASS_NAMES:
            index = CLASS_NAMES.index(risk_label)
            avg_probs[index] = max(
                avg_probs[index],
                (1 - BATCH_TILE_RISK_WEIGHT) * avg_probs[index]
                + BATCH_TILE_RISK_WEIGHT * max_probs[index],
            )

    avg_probs = avg_probs / avg_probs.sum()
    view_summaries = []
    for (name, _), probs in zip(views, stacked):
        index = int(np.argmax(probs))
        view_summaries.append(
            {
                "view": name,
                "label": CLASS_NAMES[index],
                "confidence": round(float(probs[index]), 4),
            }
        )

    return avg_probs, view_summaries


def image_quality_review(image: Image.Image) -> Tuple[bool, Optional[str]]:
    width, height = image.size
    if min(width, height) < 160:
        return True, (
            f"Image resolution is too small ({width}x{height}). "
            "Upload a clear batch photo at least 160 pixels on each side."
        )

    small = image.convert("RGB").resize((96, 96))
    array = np.asarray(small, dtype=np.float32)
    channel_std = float(array.reshape(-1, 3).std(axis=0).mean())
    brightness = array.mean(axis=2)
    very_bright_ratio = float((brightness > 242).mean())
    very_dark_ratio = float((brightness < 25).mean())

    if channel_std < 8:
        return True, "Image has too little visual texture for reliable maize quality assessment."
    if very_bright_ratio > 0.92:
        return True, "Image is mostly blank or over-exposed; retake closer to the maize sample."
    if very_dark_ratio > 0.80:
        return True, "Image is too dark for reliable maize quality assessment."
    return False, None


@app.get("/")
def health_check():
    return {
        "status": "ready",
        "message": "MaizeGuard PyTorch model API is running",
        "model": MODEL_NAME,
        "classes": CLASS_NAMES,
        "image_size": IMG_SIZE,
        "model_path": str(MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top2_margin_threshold": TOP2_MARGIN_THRESHOLD,
        "mixed_risk_review_threshold": MIXED_RISK_REVIEW_THRESHOLD,
        "batch_inference_enabled": BATCH_INFERENCE_ENABLED,
        "batch_tile_min_side": BATCH_TILE_MIN_SIDE,
        "batch_tile_scale": BATCH_TILE_SCALE,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    quality_review, quality_reason = image_quality_review(pil_image)

    avg_probs, view_summaries = predict_views(pil_image)

    raw_index = int(np.argmax(avg_probs))
    raw_label = CLASS_NAMES[raw_index]
    confidence = float(avg_probs[raw_index])
    sorted_probs = np.sort(avg_probs)[::-1]
    top2_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
    mixed_reason = mixed_risk_reason(avg_probs, raw_label)
    review = quality_review or needs_review(avg_probs, confidence) or mixed_reason is not None
    review_reason = quality_reason or mixed_reason

    return {
        "label": raw_label,
        "raw_label": raw_label,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "needs_review": review,
        "review_reason": review_reason,
        "input_width": pil_image.width,
        "input_height": pil_image.height,
        "inference_view": "full_image_plus_tiles" if len(view_summaries) > 1 else "full_image",
        "view_count": len(view_summaries),
        "view_predictions": view_summaries,
        "top2_margin": round(top2_margin, 4),
        "probabilities": {
            CLASS_NAMES[index]: round(float(avg_probs[index]), 4)
            for index in range(len(CLASS_NAMES))
        },
        **recommendation_for(raw_label, review),
    }
