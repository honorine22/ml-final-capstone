import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]


def load_split(split_dir: Path, image_size: int):
    classes = sorted([path.name for path in split_dir.iterdir() if path.is_dir()])
    rows = []
    labels = []
    for label_index, class_name in enumerate(classes):
        for path in sorted((split_dir / class_name).glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            with Image.open(path) as image:
                image = image.convert("RGB").resize((image_size, image_size))
                arr = np.asarray(image, dtype=np.float32) / 255.0
                color_mean = arr.mean(axis=(0, 1))
                color_std = arr.std(axis=(0, 1))
                hist_features = []
                for channel in range(3):
                    hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0, 1), density=True)
                    hist_features.extend(hist.tolist())
                small = Image.fromarray((arr * 255).astype(np.uint8)).resize((32, 32))
                pixels = np.asarray(small, dtype=np.float32).reshape(-1) / 255.0
                rows.append(np.concatenate([color_mean, color_std, hist_features, pixels]))
                labels.append(label_index)
    return np.asarray(rows), np.asarray(labels), classes


def evaluate(name, model, x_train, y_train, x_test, y_test, classes):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            pred,
            target_names=classes,
            zero_division=0,
            output_dict=True,
        ),
        "estimator": model,
    }


def main():
    parser = argparse.ArgumentParser(description="Train lightweight classical baselines from image features.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--out", default="reports/sklearn_model_comparison.json")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    x_train, y_train, classes = load_split(data_dir / "train", args.image_size)
    x_val, y_val, val_classes = load_split(data_dir / "val", args.image_size)
    x_test, y_test, test_classes = load_split(data_dir / "test", args.image_size)
    if classes != val_classes or classes != test_classes:
        raise SystemExit("Class folders must match across train/val/test.")

    x_train_full = np.concatenate([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    candidates = [
        ("logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("linear_svm", make_pipeline(StandardScaler(), LinearSVC(max_iter=5000))),
        ("random_forest", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
    ]
    results = []
    best = None
    for name, model in candidates:
        row = evaluate(name, model, x_train_full, y_train_full, x_test, y_test, classes)
        estimator = row.pop("estimator")
        results.append(row)
        if best is None or row["f1_macro"] > best[0]["f1_macro"]:
            best = (row, estimator)
        print(json.dumps(row, indent=2))

    ranked = sorted(results, key=lambda item: item["f1_macro"], reverse=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"classes": classes, "recommended_model": ranked[0]["model"], "ranking": ranked}, indent=2), encoding="utf-8")
    with (ROOT / "reports" / "sklearn_best_model.pkl").open("wb") as file:
        pickle.dump({"classes": classes, "model": best[1], "image_size": args.image_size}, file)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
