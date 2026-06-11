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
EXPORT_DIR = BASE_DIR / "model_exports"
MODEL_PATH = Path(os.getenv("PYTORCH_MODEL_PATH", EXPORT_DIR / "maizeguard_public_best_model.pt"))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", EXPORT_DIR / "maizeguard_model_metadata.json"))

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
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
CLASS_NAMES = metadata["class_names"]
MODEL_NAME = metadata["model_name"]
IMG_SIZE = int(metadata.get("image_size", 224))
MEAN = metadata.get("normalization_mean", [0.485, 0.456, 0.406])
STD = metadata.get("normalization_std", [0.229, 0.224, 0.225])

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


def recommendation_for(label: str, confidence: float):
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "risk": "Medium",
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

    # If a higher-risk class is reasonably likely, prefer review over a false "good".
    for risk_label in ["mold_risk", "impurity", "broken"]:
        if risk_label in CLASS_NAMES:
            index = CLASS_NAMES.index(risk_label)
            if probabilities[index] >= 0.45:
                return risk_label, float(probabilities[index])

    return label, confidence


@app.get("/")
def health_check():
    return {
        "status": "ready",
        "message": "MaizeGuard PyTorch model API is running",
        "model": MODEL_NAME,
        "classes": CLASS_NAMES,
        "image_size": IMG_SIZE,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    crops = make_crops(pil_image)

    crop_probs = []
    with torch.no_grad():
        for crop in crops:
            tensor = valid_tfms(crop).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            crop_probs.append(probs)

    avg_probs = np.mean(crop_probs, axis=0)
    label, confidence = choose_conservative_label(avg_probs)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "probabilities": {
            CLASS_NAMES[index]: round(float(avg_probs[index]), 4)
            for index in range(len(CLASS_NAMES))
        },
        **recommendation_for(label, confidence),
    }
