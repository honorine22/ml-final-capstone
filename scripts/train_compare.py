import argparse
import json
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare maize quality classifiers.")
    parser.add_argument("--data-dir", default="data/processed", help="Folder with train/val/test class subfolders.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out", default="reports/model_comparison.json")
    return parser.parse_args()


def validate_dataset(data_dir: Path):
    required = ["train", "val", "test"]
    missing = [split for split in required if not (data_dir / split).exists()]
    if missing:
      raise SystemExit(f"Missing dataset split folders: {', '.join(missing)} under {data_dir}")

    class_dirs = sorted([path.name for path in (data_dir / "train").iterdir() if path.is_dir()])
    if len(class_dirs) < 2:
      raise SystemExit(
          "Add at least two class folders under data/processed/train, for example "
          "good, broken, impurity, discolored, mold."
      )
    return class_dirs


def build_model(name, image_size, class_count, tf):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.08),
        ],
        name="augmentation",
    )

    if name == "custom_cnn":
        x = augmentation(inputs)
        x = tf.keras.layers.Rescaling(1.0 / 255)(x)
        for filters in [32, 64, 128]:
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        app = {
            "mobilenet_v2": tf.keras.applications.MobileNetV2,
            "efficientnet_b0": tf.keras.applications.EfficientNetB0,
            "resnet50": tf.keras.applications.ResNet50,
        }[name]
        base = app(include_top=False, weights="imagenet", input_tensor=inputs)
        base.trainable = False
        x = augmentation(inputs)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    class_names = validate_dataset(data_dir)

    try:
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
    except ImportError as error:
        raise SystemExit(
            "Missing ML dependencies. Install tensorflow, scikit-learn, and numpy before running."
        ) from error

    image_size = (args.image_size, args.image_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    results = []

    for model_name in ["custom_cnn", "mobilenet_v2", "efficientnet_b0", "resnet50"]:
        print(f"\nTraining {model_name}...")
        model = build_model(model_name, args.image_size, len(class_names), tf)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True,
            )
        ]
        start = time.perf_counter()
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
        train_seconds = time.perf_counter() - start

        predict_start = time.perf_counter()
        probabilities = model.predict(test_ds)
        inference_seconds = time.perf_counter() - predict_start
        y_pred = np.argmax(probabilities, axis=1)

        model_path = Path("reports/models") / f"{model_name}.keras"
        model.save(model_path)
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        row = {
            "model": model_name,
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "test_accuracy": float(np.mean(y_pred == y_true)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "model_size_mb": round(model_size_mb, 2),
            "inference_ms_per_batch": round((inference_seconds / max(len(test_ds), 1)) * 1000, 2),
            "train_seconds": round(train_seconds, 2),
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0,
                output_dict=True,
            ),
        }
        results.append(row)
        print(json.dumps({key: row[key] for key in row if key != "classification_report"}, indent=2))

    ranked = sorted(results, key=lambda item: (item["f1_macro"], -item["model_size_mb"]), reverse=True)
    output = {
        "classes": class_names,
        "recommended_model": ranked[0]["model"],
        "ranking": ranked,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved comparison to {out_path}")
    print(f"Recommended model: {output['recommended_model']}")


if __name__ == "__main__":
    main()
