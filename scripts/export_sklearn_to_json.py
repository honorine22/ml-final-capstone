import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    model_path = ROOT / "reports" / "sklearn_best_model.pkl"
    out_path = ROOT / "public" / "model" / "maize_linear_model.json"
    if not model_path.exists():
        raise SystemExit("Missing reports/sklearn_best_model.pkl. Run train_sklearn_baseline.py first.")

    with model_path.open("rb") as file:
        artifact = pickle.load(file)

    pipeline = artifact["model"]
    scaler = pipeline.named_steps["standardscaler"]
    logistic = pipeline.named_steps.get("logisticregression")
    if logistic is None:
        raise SystemExit("Best model is not logistic regression. Re-run with logistic regression or export manually.")

    out = {
        "type": "standard_scaled_logistic_regression",
        "classes": artifact["classes"],
        "imageSize": artifact["image_size"],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": logistic.coef_.tolist(),
        "intercept": logistic.intercept_.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
