import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_server"))

from pytorch_main import CLASS_NAMES, DEVICE, health_check, model, valid_tfms  # noqa: E402


REQUIRED_FILES = [
    "README.md",
    "notebooks/maizeguard_public_datasets_pytorch_training.ipynb",
    "model_server/pytorch_main.py",
    "model_server/main.py",
    "model_server/model_exports/maizeguard_public_best_model.pt",
    "model_server/model_exports/maizeguard_model_metadata.json",
    "model_server/model_exports/class_names.json",
    "reports/models/model_metrics_summary.csv",
    "reports/models/classification_report_raw_argmax.csv",
    "reports/models/test_predictions_and_errors_raw_argmax.csv",
    "reports/models/api_response_example.json",
    "docs/deployment-plan.md",
    "docs/api-test.http",
]

REQUIRED_DIAGRAMS = [
    "01_research_model.png",
    "02_system_architecture.png",
    "03_use_case_diagram.png",
    "04_class_diagram.png",
    "05_erd.png",
    "06_sequence_diagram.png",
    "07_gantt_chart.png",
]

REQUIRED_SCREENSHOTS = [
    "00_app_interface_home.png",
    "01_class_distribution_by_split.png",
    "03_sample_images_by_class.png",
    "06_training_validation_loss.png",
    "08_confusion_matrix_raw_argmax.png",
    "10_per_class_metrics.png",
]


def status_line(ok: bool, label: str) -> str:
    return f"{'PASS' if ok else 'FAIL'}  {label}"


def file_exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def read_metrics():
    metrics_path = ROOT / "reports" / "models" / "model_metrics_summary.csv"
    if not metrics_path.exists():
        return None

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


@torch.no_grad()
def smoke_predict():
    candidates = [
        ROOT / "data" / "external_test" / "ckcnn_sanity" / "good",
        ROOT / "data" / "processed" / "test" / "good",
    ]
    image_path = None
    for folder in candidates:
        if folder.exists():
            image_path = next(iter(sorted(folder.glob("*"))), None)
        if image_path:
            break

    if image_path is None:
        return None

    image = Image.open(image_path).convert("RGB")
    tensor = valid_tfms(image).unsqueeze(0).to(DEVICE)
    probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    index = max(range(len(probabilities)), key=probabilities.__getitem__)

    return {
        "file": str(image_path.relative_to(ROOT)),
        "label": CLASS_NAMES[index],
        "confidence_percent": round(probabilities[index] * 100, 2),
    }


def main():
    failures = []

    print("MaizeGuard capstone readiness check\n")

    for path in REQUIRED_FILES:
        ok = file_exists(path)
        print(status_line(ok, path))
        if not ok:
            failures.append(path)

    for name in REQUIRED_DIAGRAMS:
        path = f"docs/diagrams/{name}"
        ok = file_exists(path)
        print(status_line(ok, path))
        if not ok:
            failures.append(path)

    for name in REQUIRED_SCREENSHOTS:
        path = f"docs/screenshots/{name}"
        ok = file_exists(path)
        print(status_line(ok, path))
        if not ok:
            failures.append(path)

    print("\nModel server health")
    print(json.dumps(health_check(), indent=2))

    metrics = read_metrics()
    print("\nLatest notebook metrics")
    if metrics:
        print(json.dumps(metrics, indent=2))
    else:
        print("No metrics summary found.")
        failures.append("reports/models/model_metrics_summary.csv")

    prediction = smoke_predict()
    print("\nLocal model smoke prediction")
    if prediction:
        print(json.dumps(prediction, indent=2))
    else:
        print("No smoke-test image found.")
        failures.append("data/external_test/ckcnn_sanity/good")

    external_summary = ROOT / "reports" / "external_test" / "summary.json"
    if external_summary.exists():
        print("\nExternal/domain-shift test summary")
        print(external_summary.read_text(encoding="utf-8"))

    print("\nResult")
    if failures:
        print(f"FAIL: {len(failures)} required item(s) need attention.")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print("PASS: Required capstone demo files and model smoke test are ready.")


if __name__ == "__main__":
    main()
