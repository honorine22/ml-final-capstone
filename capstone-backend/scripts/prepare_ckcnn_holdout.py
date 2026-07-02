import csv
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "models" / "test_manifest.csv"
SOURCE_ROOT = ROOT / "data" / "raw" / "ck-cnn" / "CK-CNN-master" / "dataset" / "individual"
OUTPUT_ROOT = ROOT / "data" / "external_test" / "ckcnn_sanity"


if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

counts = {}
with MANIFEST.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        source_folder = row["original_folder"]
        source = SOURCE_ROOT / source_folder / row["file_name"]
        destination_dir = OUTPUT_ROOT / row["label"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"ckcnn_{source_folder}_{source.stem}.png"

        if not source.exists():
            raise FileNotFoundError(source)

        with Image.open(source) as image:
            image.convert("RGB").save(destination, format="PNG", optimize=True)

        counts[row["label"]] = counts.get(row["label"], 0) + 1

print(f"Prepared exact notebook holdout at {OUTPUT_ROOT}")
print(counts)
