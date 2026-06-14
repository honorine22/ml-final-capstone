import argparse
import json
import random
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = ["good", "broken", "impurity", "mold_risk"]

IGNORE_PATTERNS = [
    "mask",
    "segmentation",
    "annotation",
    "label",
    "bbox",
    "bounding",
    "groundtruth",
    "plot",
    "chart",
    "confusion",
    "readme",
    "architecture",
    "fig",
    "figure",
    "grainset-tiny",
    "tiny_data",
]

NON_MAIZE_PATTERNS = [
    "sorg",
    "sorghum",
    "wheat",
    "rice",
    "bean",
]

LABEL_KEYWORDS = {
    "good": ["good", "healthy", "normal", "sound", "clean"],
    "impurity": ["impurity", "foreign", "stone", "stones", "husk", "dust", "trash", "debris"],
    "mold_risk": ["mold", "mould", "fung", "fungus", "fungal", "infected", "infection", "rotten", "rot"],
    "broken": [
        "defective",
        "defect",
        "damaged",
        "damage",
        "broken",
        "unsound",
        "bad",
        "cracked",
        "shrivel",
        "low_quality",
        "low-quality",
    ],
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def should_ignore(path: Path) -> bool:
    text = str(path).lower()
    return any(pattern in text for pattern in IGNORE_PATTERNS)


def is_non_maize(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    text = " ".join(parts)
    return any(re.search(rf"(^|[^a-z]){re.escape(pattern)}([^a-z]|$)", text) for pattern in NON_MAIZE_PATTERNS)


def ckcnn_label(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    if "good" in parts:
        return "good"
    if "impurity" in parts:
        return "impurity"
    if "broken" in parts or "defective" in parts or "damaged" in parts:
        return "broken"
    if "rotten" in parts or "mold" in parts or "fungus" in parts:
        return "mold_risk"
    return None


def infer_label(path: Path) -> str | None:
    if is_non_maize(path):
        return None

    source = source_name(path)
    if source == "CK-CNN":
        label = ckcnn_label(path)
        if label is not None:
            return label

    text = " ".join(part.lower() for part in path.parts[-8:])

    # Priority matters: impurity and mold-risk should not be swallowed by broad "bad" labels.
    for label in ["impurity", "mold_risk", "good", "broken"]:
        for keyword in LABEL_KEYWORDS[label]:
            if re.search(rf"(^|[^a-z]){re.escape(keyword)}([^a-z]|$)", text):
                return label
    return None


def source_name(path: Path) -> str:
    text = str(path).lower()
    if "ck-cnn" in text or "ck_cnn" in text or "ckcnn" in text:
        return "CK-CNN"
    if "grainset" in text:
        return "GrainSet"
    if "efficientmaize" in text or "maize seed" in text or "r6vvm5jkh6" in text:
        return "EfficientMaize"
    return path.parts[-4] if len(path.parts) >= 4 else "public_source"


def collect_manifest(roots: list[Path]) -> pd.DataFrame:
    rows = []
    skipped = []
    for root in roots:
        if not root.exists():
            print(f"Skipping missing root: {root}")
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not is_image(path) or should_ignore(path):
                if path.is_file() and is_image(path):
                    skipped.append({"path": str(path), "reason": "ignored_pattern"})
                continue
            if is_non_maize(path):
                skipped.append({"path": str(path), "reason": "non_maize"})
                continue
            label = infer_label(path)
            if label is None:
                skipped.append({"path": str(path), "reason": "no_safe_label"})
                continue
            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "source": source_name(path),
                    "original_folder": path.parent.name,
                    "file_name": path.name,
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    frame.attrs["skipped"] = skipped
    return frame


def copy_image(src: Path, dst: Path, image_size: int) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as image:
            image = image.convert("RGB")
            image.thumbnail((image_size, image_size))
            canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
            left = (image_size - image.width) // 2
            top = (image_size - image.height) // 2
            canvas.paste(image, (left, top))
            canvas.save(dst, quality=92)
        return True
    except Exception as exc:
        print(f"Skipping unreadable image {src}: {exc}")
        return False


def write_split_images(split_name: str, frame: pd.DataFrame, out_dir: Path, image_size: int) -> pd.DataFrame:
    rows = []
    for index, row in frame.reset_index(drop=True).iterrows():
        src = Path(row["path"])
        dst = out_dir / split_name / row["label"] / f"{row['source'].lower().replace(' ', '_')}_{index:06d}.jpg"
        if copy_image(src, dst, image_size):
            item = row.to_dict()
            item["prepared_path"] = str(dst)
            rows.append(item)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare public maize/corn datasets for MaizeGuard.")
    parser.add_argument("--roots", nargs="+", default=["data/raw"], help="Public dataset folders to scan.")
    parser.add_argument("--out-dir", default="data/public_prepared", help="Prepared split image folder.")
    parser.add_argument("--report-dir", default="reports/public_training", help="Manifest/report folder.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images-per-class", type=int, default=2500)
    parser.add_argument("--min-images-per-class", type=int, default=40)
    parser.add_argument("--exclude-mold-risk", action="store_true", help="Train a safer 3-class model when public mold labels are too weak.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    roots = [Path(root) if Path(root).is_absolute() else ROOT / root for root in args.roots]
    out_dir = ROOT / args.out_dir
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest = collect_manifest(roots)
    skipped = manifest.attrs.get("skipped", [])
    if manifest.empty:
        raise SystemExit("No mapped public images found. Add CK-CNN, GrainSet, or EfficientMaize under the roots.")

    class_names = [name for name in CLASS_NAMES if not (args.exclude_mold_risk and name == "mold_risk")]
    manifest = manifest[manifest["label"].isin(class_names)].copy()

    counts_before = manifest["label"].value_counts().to_dict()
    valid_classes = [
        label
        for label, count in counts_before.items()
        if label in class_names and count >= args.min_images_per_class
    ]
    manifest = manifest[manifest["label"].isin(valid_classes)].copy()

    if "good" not in set(manifest["label"]):
        raise SystemExit("Prepared public dataset must include the good class.")

    balanced = []
    for label, group in manifest.groupby("label"):
        count = min(len(group), args.max_images_per_class)
        balanced.append(group.sample(n=count, random_state=args.seed))
    manifest = pd.concat(balanced).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        manifest,
        test_size=0.30,
        random_state=args.seed,
        stratify=manifest["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)

    prepared = {
        "train": write_split_images("train", train_df, out_dir, args.image_size),
        "val": write_split_images("val", val_df, out_dir, args.image_size),
        "test": write_split_images("test", test_df, out_dir, args.image_size),
    }

    manifest.to_csv(report_dir / "public_dataset_manifest_all.csv", index=False)
    for split, frame in prepared.items():
        frame.to_csv(report_dir / f"{split}_manifest.csv", index=False)

    summary = {
        "class_names": class_names,
        "counts_before_filtering": counts_before,
        "counts_after_balancing": manifest["label"].value_counts().to_dict(),
        "prepared_counts": {
            split: frame["label"].value_counts().to_dict()
            for split, frame in prepared.items()
        },
        "label_policy": {
            "CK-CNN good": "good",
            "CK-CNN defective/broken": "broken",
            "CK-CNN impurity": "impurity",
            "CK-CNN rotten/fungal": "mold_risk only if keeping the 4-class model",
            "GrainSet healthy maize": "good",
            "GrainSet damaged/unsound maize": "broken or mold_risk only when label is clear",
            "EfficientMaize good": "good",
            "EfficientMaize bad": "support only; do not force unclear bad images into mold_risk",
            "ignored": "GrainSet-tiny, sorghum/sorg, wheat, rice, masks, annotations, and unclear labels",
        },
    }
    (report_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (report_dir / "dataset_mapping_audit.json").write_text(
        json.dumps(
            {
                "mapped_by_source_and_label": pd.crosstab(manifest["source"], manifest["label"]).to_dict(),
                "skipped_examples": skipped[:1000],
                "skipped_count": len(skipped),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Prepared dataset: {out_dir}")
    print(f"Reports: {report_dir}")


if __name__ == "__main__":
    main()
