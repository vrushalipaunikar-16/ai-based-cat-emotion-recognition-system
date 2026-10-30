"""
scripts/preprocess_images.py
Resize and copy images from the raw dataset into data/processed/images/,
preserving the class folder structure (train/valid × 7 classes).

Usage:
  python scripts/preprocess_images.py
"""

import sys
import shutil
from pathlib import Path
from PIL import Image

ROOT      = Path(__file__).resolve().parent.parent   # meowmood/
SRC_DIR   = ROOT / "data" / "raw" / "images"
DST_DIR   = ROOT / "data" / "processed" / "images"
IMG_SIZE  = (128, 128)


def process_split(split: str):
    src_split = SRC_DIR / split
    dst_split = DST_DIR / split
    if not src_split.exists():
        print(f"[SKIP] {src_split} not found.")
        return

    for cls_dir in sorted(src_split.iterdir()):
        if not cls_dir.is_dir():
            continue
        out_dir = dst_split / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                img.save(out_dir / img_path.name)
            except Exception as e:
                print(f"[WARN] {img_path.name}: {e}")

        print(f"  {split}/{cls_dir.name}: {len(imgs)} images processed → {out_dir}")


if __name__ == "__main__":
    print("Preprocessing images...")
    for split in ["train", "valid"]:
        process_split(split)
    print("Done.")
