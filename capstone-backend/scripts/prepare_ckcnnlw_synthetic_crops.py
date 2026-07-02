import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


CLASS_MAP = {
    "good": "good",
    "healthy": "good",
    "normal": "good",
    "impurity": "impurity",
    "impurities": "impurity",
    "foreign": "impurity",
    "others": "impurity",
    "other": "impurity",
    "defective": "broken",
    "defect": "broken",
    "broken": "broken",
    "damage": "broken",
    "damaged": "broken",
    "rotten": "mold_risk",
    "rot": "mold_risk",
    "fungus": "mold_risk",
    "fungal": "mold_risk",
    "mold": "mold_risk",
    "mould": "mold_risk",
}


def normalize_label(value):
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    for key, mapped in CLASS_MAP.items():
        if key in text:
            return mapped
    return None


def get_region_label(region):
    shape = region.get("shape_attributes", {})
    attrs = region.get("region_attributes", {})
    candidates = [
        shape.get("class_name"),
        attrs.get("class_name"),
        attrs.get("class"),
        attrs.get("label"),
        attrs.get("name"),
    ]

    for value in candidates:
        mapped = normalize_label(value)
        if mapped:
            return mapped
    return None


def iter_regions(annotation_data):
    for key, item in annotation_data.items():
        filename = item.get("filename")
        regions = item.get("regions", [])

        if isinstance(regions, dict):
            regions = regions.values()

        for index, region in enumerate(regions):
            yield key, filename, index, region


def crop_polygon(image, points_x, points_y, padding, masked):
    width, height = image.size
    min_x = max(min(points_x) - padding, 0)
    min_y = max(min(points_y) - padding, 0)
    max_x = min(max(points_x) + padding, width - 1)
    max_y = min(max(points_y) + padding, height - 1)

    if max_x <= min_x or max_y <= min_y:
        return None

    box = (int(min_x), int(min_y), int(max_x) + 1, int(max_y) + 1)
    crop = image.crop(box).convert("RGB")

    if not masked:
        return crop

    shifted = [(x - box[0], y - box[1]) for x, y in zip(points_x, points_y)]
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(shifted, fill=255)

    background = Image.new("RGB", crop.size, (245, 245, 245))
    background.paste(crop, mask=mask)
    return background


def process_split(source_root, split, output_root, padding, masked, min_size):
    split_root = source_root / split
    annotation_path = split_root / "via_region_data.json"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    annotation_data = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = []
    counts = Counter()
    skipped = Counter()
    image_cache = {}

    for _, filename, region_index, region in iter_regions(annotation_data):
        if not filename:
            skipped["missing_filename"] += 1
            continue

        label = get_region_label(region)
        if not label:
            skipped["unknown_label"] += 1
            continue

        shape = region.get("shape_attributes", {})
        points_x = shape.get("all_points_x") or []
        points_y = shape.get("all_points_y") or []

        if len(points_x) < 3 or len(points_y) < 3:
            skipped["invalid_polygon"] += 1
            continue

        image_path = split_root / filename
        if not image_path.exists():
            skipped["missing_image"] += 1
            continue

        if image_path not in image_cache:
            image_cache[image_path] = Image.open(image_path).convert("RGB")

        crop = crop_polygon(image_cache[image_path], points_x, points_y, padding, masked)
        if crop is None or min(crop.size) < min_size:
            skipped["too_small"] += 1
            continue

        class_dir = output_root / split / label
        class_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem
        output_name = f"ckcnnlw_{stem}_{region_index:04d}.jpg"
        output_path = class_dir / output_name
        crop.save(output_path, quality=92)

        counts[label] += 1
        rows.append(
            {
                "split": split,
                "source_image": str(image_path),
                "output_image": str(output_path),
                "label": label,
                "width": crop.width,
                "height": crop.height,
                "masked": masked,
            }
        )

    return rows, counts, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Extract labeled object crops from CK-CNNLW synthesized cluster images."
    )
    parser.add_argument(
        "--source-root",
        default="data/raw/CK-CNNLW/dataset/synthesized",
        help="Path containing train/ and val/ folders with via_region_data.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/synthetic_ckcnnlw_crops",
        help="Output folder for cropped class folders.",
    )
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--min-size", type=int, default=24)
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Save rectangular crops instead of polygon-masked crops on a neutral background.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    total_counts = Counter()
    total_skipped = Counter()

    for split in ["train", "val"]:
        rows, counts, skipped = process_split(
            source_root=source_root,
            split=split,
            output_root=output_root,
            padding=args.padding,
            masked=not args.no_mask,
            min_size=args.min_size,
        )
        all_rows.extend(rows)
        total_counts.update({f"{split}/{key}": value for key, value in counts.items()})
        total_skipped.update({f"{split}/{key}": value for key, value in skipped.items()})

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "source_image",
                "output_image",
                "label",
                "width",
                "height",
                "masked",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "total_crops": len(all_rows),
        "counts": dict(total_counts),
        "skipped": dict(total_skipped),
        "recommendation": (
            "Use these crops as extra training data only. Keep real external phone images "
            "as validation/test data so synthetic images do not inflate the final metrics."
        ),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
