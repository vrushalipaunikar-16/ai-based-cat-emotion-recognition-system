"""
scripts/preprocess_audio.py
Extract MFCC features from all wav files in data/raw/audio/
and save them as .npy arrays in data/processed/audio/.

Also generates data/labels.csv mapping each file to its emotion label.

Usage:
  python scripts/preprocess_audio.py
"""

import sys
import csv
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent   # meowmood/
sys.path.insert(0, str(ROOT))
from backend.utils import extract_mfcc, AUDIO_LABEL_MAP

SRC_DIR  = ROOT / "data" / "raw" / "audio"
DST_DIR  = ROOT / "data" / "processed" / "audio"
CSV_OUT  = ROOT / "data" / "labels.csv"


def run():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    wav_files = sorted(SRC_DIR.glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No wav files found in {SRC_DIR}")
        return

    for wav in wav_files:
        prefix = wav.stem[0]
        if prefix not in AUDIO_LABEL_MAP:
            print(f"[SKIP] Unknown prefix '{prefix}' in {wav.name}")
            continue
        label = AUDIO_LABEL_MAP[prefix]
        try:
            mfcc = extract_mfcc(str(wav))          # (MAX_PAD, N_MFCC)
            out_path = DST_DIR / (wav.stem + ".npy")
            np.save(str(out_path), mfcc)
            rows.append({"file": wav.name, "npy": out_path.name, "label": label})
        except Exception as e:
            print(f"[WARN] {wav.name}: {e}")

    # Write labels CSV
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "npy", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} audio files.")
    print(f"Features saved → {DST_DIR}")
    print(f"Labels CSV     → {CSV_OUT}")


if __name__ == "__main__":
    run()
