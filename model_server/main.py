from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import os

import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_exports", "maizeguard_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model_exports", "class_names.json")

IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

app = FastAPI(title="MaizeGuard Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3006").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)


def normalize_label(label: str) -> str:
    value = label.lower().strip()

    if "good" in value or "healthy" in value or "normal" in value:
        return "good"
    if "broken" in value or "damage" in value or "defect" in value:
        return "broken"
    if "impurity" in value or "dirty" in value or "foreign" in value:
        return "impurity"
    if "discolor" in value or "stain" in value or "dark" in value:
        return "discolored"
    if "mold" in value or "rotten" in value or "fung" in value:
        return "mold"

    return value


def recommendation_for(label: str, confidence: float) -> dict:
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "risk": "Medium",
            "action": "Needs review",
            "recommendation": "The image is uncertain. Retake the photo closer to the maize sample on a plain surface.",
        }

    recommendations = {
        "good": {
            "risk": "Low",
            "action": "Store safely or prepare for sale",
            "recommendation": "The batch appears clean and suitable for normal storage with routine monitoring.",
        },
        "broken": {
            "risk": "Medium",
            "action": "Sort before storage",
            "recommendation": "Remove visibly broken or damaged kernels before storage or sale.",
        },
        "impurity": {
            "risk": "Medium",
            "action": "Clean and re-screen",
            "recommendation": "Separate stones, husks, dust, and foreign matter before aggregation or sale.",
        },
        "discolored": {
            "risk": "Medium",
            "action": "Sell quickly or refer for review",
            "recommendation": "Avoid mixing discolored maize with clean grain. Refer for further checking if needed.",
        },
        "mold": {
            "risk": "High",
            "action": "Do not store; refer for checking",
            "recommendation": "Visible mold-risk maize should not be stored directly. Refer the batch for further assessment.",
        },
    }

    return recommendations.get(
        label,
        {
            "risk": "Medium",
            "action": "Needs review",
            "recommendation": "The model could not map this prediction to a safe recommendation.",
        },
    )


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


@app.get("/")
def health_check():
    return {
        "status": "ready",
        "message": "MaizeGuard model API is running",
        "classes": class_names,
        "image_size": IMG_SIZE,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    input_array = preprocess_image(image_bytes)
    predictions = model.predict(input_array, verbose=0)[0]

    pred_index = int(np.argmax(predictions))
    raw_label = str(class_names[pred_index])
    label = normalize_label(raw_label)
    confidence = float(predictions[pred_index])

    probabilities = {
        str(class_names[index]): float(predictions[index])
        for index in range(len(class_names))
    }

    return {
        "label": label,
        "raw_label": raw_label,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "probabilities": probabilities,
        **recommendation_for(label, confidence),
    }
