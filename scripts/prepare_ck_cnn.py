import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


CLASS_MAP = {
    "good": "good",
    "broken": "broken",
    "impurity": "impurity",
    "rotten": "mold",
}


def collect_images(source: Path):
    return sorted([path for path in source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])


def save_image(src: Path, dst: Path, image_size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert("RGB")
        image.thumbnail((image_size, image_size))
        canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        left = (image_size - image.width) // 2
        top = (image_size - image.height) // 2
        canvas.paste(image, (left, top))
        canvas.save(dst, quality=90)


def main():
    parser = argparse.ArgumentParser(description="Prepare CK-CNN images for training.")
    parser.add_argument("--raw-dir", default="data/raw/ck-cnn/CK-CNN-master/dataset/individual")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = ROOT / args.raw_dir
    out_dir = ROOT / args.out_dir
    if not raw_dir.exists():
        raise SystemExit(f"Missing CK-CNN folder: {raw_dir}. Run scripts/download_datasets.py --ck-cnn first.")

    random.seed(args.seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for source_name, target_name in CLASS_MAP.items():
        images = collect_images(raw_dir / source_name)
        if not images:
            print(f"No images found for {source_name}")
            continue
        random.shuffle(images)
        train_end = int(len(images) * 0.7)
        val_end = int(len(images) * 0.85)
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }
        for split, split_images in splits.items():
            for index, image_path in enumerate(split_images):
                dst = out_dir / split / target_name / f"ckcnn_{source_name}_{index:04d}.jpg"
                save_image(image_path, dst, args.image_size)
        print(f"{source_name} -> {target_name}: {len(images)} images")


if __name__ == "__main__":
    main()
