import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_server"))

from pytorch_main import CLASS_NAMES, DEVICE, image_quality_review, model, needs_review, valid_tfms  # noqa: E402


TEST_ROOT = ROOT / "data" / "external_test"
OUTPUT_DIR = ROOT / "reports" / "external_test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@torch.no_grad()
def predict(path: Path):
    image = Image.open(path).convert("RGB")
    tensor = valid_tfms(image).unsqueeze(0).to(DEVICE)
    probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    quality_review, quality_reason = image_quality_review(image)
    review = quality_review or needs_review(np.asarray(probabilities), probabilities[index])
    return CLASS_NAMES[index], probabilities[index], probabilities, review, quality_reason


rows = []
for path in sorted(TEST_ROOT.rglob("*")):
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    relative = path.relative_to(TEST_ROOT)
    source = relative.parts[0]
    folder_label = relative.parts[-2]
    expected = folder_label if source == "ckcnn_sanity" else (
        "good" if folder_label == "good" else "needs_manual_review"
    )
    predicted, confidence, probabilities, review, review_reason = predict(path)

    rows.append(
        {
            "file": str(relative),
            "source": source,
            "expected": expected,
            "predicted": predicted,
            "confidence": round(confidence, 6),
            "needs_review": review,
            "review_reason": review_reason or "",
            **{
                f"prob_{label}": round(probabilities[index], 6)
                for index, label in enumerate(CLASS_NAMES)
            },
        }
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUTPUT_DIR / "predictions.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for row in rows:
    groups[(row["source"], row["expected"])].append(row)

summary = {}
for (source, expected), items in groups.items():
    key = f"{source}/{expected}"
    counts = Counter(item["predicted"] for item in items)
    reviewed = sum(str(item["needs_review"]).lower() == "true" for item in items)
    final_items = [item for item in items if str(item["needs_review"]).lower() != "true"]
    summary[key] = {
        "samples": len(items),
        "prediction_counts": dict(counts),
        "needs_review": reviewed,
        "final_decisions": len(final_items),
        "accuracy": (
            round(sum(item["predicted"] == expected for item in items) / len(items), 4)
            if expected in CLASS_NAMES
            else None
        ),
        "final_decision_accuracy": (
            round(sum(item["predicted"] == expected for item in final_items) / len(final_items), 4)
            if expected in CLASS_NAMES and final_items
            else None
        ),
    }

summary_path = OUTPUT_DIR / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"Saved: {csv_path}")
