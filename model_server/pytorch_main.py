from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import os
from pathlib import Path

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
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3006").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
CLASS_NAMES = metadata.get("class_names") or metadata.get("classes")
MODEL_NAME = metadata.get("model_name") or metadata.get("model_backbone", "mobilenetv3_large_100")
IMG_SIZE = int(metadata.get("image_size") or metadata.get("input_image_size", 224))
MEAN = metadata.get("normalization_mean", [0.485, 0.456, 0.406])
STD = metadata.get("normalization_std", [0.229, 0.224, 0.225])
SAFETY_RULE = metadata.get("deployment_safety_rule", {})
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", SAFETY_RULE.get("needs_review_confidence_below", 0.65)))
TOP2_MARGIN_THRESHOLD = float(os.getenv("TOP2_MARGIN_THRESHOLD", SAFETY_RULE.get("needs_review_top2_margin_below", 0.15)))
RISK_PRIORITY_THRESHOLD = float(os.getenv("RISK_PRIORITY_THRESHOLD", metadata.get("risk_priority_threshold", 0.55)))

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

valid_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def make_crops(image: Image.Image):
    image = image.convert("RGB")
    width, height = image.size
    crops = [image]

    center_size = int(min(width, height) * 0.75)
    left = (width - center_size) // 2
    top = (height - center_size) // 2
    crops.append(image.crop((left, top, left + center_size, top + center_size)))

    corner_size = int(min(width, height) * 0.60)
    boxes = [
        (0, 0, corner_size, corner_size),
        (width - corner_size, 0, width, corner_size),
        (0, height - corner_size, corner_size, height),
        (width - corner_size, height - corner_size, width, height),
    ]
    crops.extend(image.crop(box) for box in boxes)
    return crops


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


def choose_conservative_label(probabilities: np.ndarray):
    best_index = int(np.argmax(probabilities))
    label = CLASS_NAMES[best_index]
    confidence = float(probabilities[best_index])

    # If a higher-risk class is clearly likely, prefer review over a false "good".
    for risk_label in ["mold_risk", "impurity", "broken"]:
        if risk_label in CLASS_NAMES:
            index = CLASS_NAMES.index(risk_label)
            if probabilities[index] >= RISK_PRIORITY_THRESHOLD:
                return risk_label, float(probabilities[index])

    return label, confidence


def needs_review(probabilities: np.ndarray, confidence: float) -> bool:
    sorted_probs = np.sort(probabilities)[::-1]
    top2_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
    return confidence < CONFIDENCE_THRESHOLD or top2_margin < TOP2_MARGIN_THRESHOLD


def image_quality_review(image: Image.Image) -> tuple[bool, str | None]:
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
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    quality_review, quality_reason = image_quality_review(pil_image)
    crops = make_crops(pil_image)

    crop_probs = []
    with torch.no_grad():
        for crop in crops:
            tensor = valid_tfms(crop).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            crop_probs.append(probs)

    avg_probs = np.mean(crop_probs, axis=0)
    raw_index = int(np.argmax(avg_probs))
    raw_label = CLASS_NAMES[raw_index]
    label, confidence = choose_conservative_label(avg_probs)
    review = quality_review or needs_review(avg_probs, confidence)

    return {
        "label": label,
        "raw_label": raw_label,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "needs_review": review,
        "review_reason": quality_reason,
        "probabilities": {
            CLASS_NAMES[index]: round(float(avg_probs[index]), 4)
            for index in range(len(CLASS_NAMES))
        },
        **recommendation_for(label, review),
    }
