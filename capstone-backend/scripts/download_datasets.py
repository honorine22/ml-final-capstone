import argparse
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def download(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"Already exists: {out}")
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, out.open("wb") as file:
        shutil.copyfileobj(response, file)


def unzip(zip_path: Path, out_dir: Path):
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"Already extracted: {out_dir}")
        return
    print(f"Extracting {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)
    marker.write_text("ok", encoding="utf-8")


def fetch_ck_cnn():
    zip_path = RAW / "ck-cnn" / "CK-CNN-master.zip"
    download("https://codeload.github.com/vision-cidis/CK-CNN/zip/refs/heads/master", zip_path)
    unzip(zip_path, RAW / "ck-cnn")


def fetch_grainset_metadata():
    out = RAW / "grainset" / "maize_figshare_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen("https://api.figshare.com/v2/articles/22987562") as response:
        out.write_bytes(response.read())
    metadata = json.loads(out.read_text(encoding="utf-8"))
    print("GrainSet maize files:")
    for file in metadata.get("files", []):
        print(f"- {file['name']}: {file['size'] / (1024 ** 3):.2f} GB, {file['download_url']}")
    print("Large download note: maize.zip is about 6 GB. Download it only when you are ready.")


def fetch_grainset_maize():
    out = RAW / "grainset" / "maize.zip"
    download("https://ndownloader.figshare.com/files/40737164", out)
    unzip(out, RAW / "grainset" / "maize")


def main():
    parser = argparse.ArgumentParser(description="Download available MaizeGuard datasets.")
    parser.add_argument("--ck-cnn", action="store_true", help="Download CK-CNN dataset from GitHub.")
    parser.add_argument("--grainset-metadata", action="store_true", help="Save GrainSet Figshare metadata.")
    parser.add_argument("--grainset-maize", action="store_true", help="Download the 6GB GrainSet maize zip.")
    parser.add_argument("--all-light", action="store_true", help="Download light/manageable sources only.")
    args = parser.parse_args()

    if args.all_light:
        args.ck_cnn = True
        args.grainset_metadata = True

    if args.ck_cnn:
        fetch_ck_cnn()
    if args.grainset_metadata:
        fetch_grainset_metadata()
    if args.grainset_maize:
        fetch_grainset_maize()

    if not any([args.ck_cnn, args.grainset_metadata, args.grainset_maize]):
        parser.print_help()


if __name__ == "__main__":
    main()
