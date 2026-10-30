"""
scripts/data_summary.py
EDA: generate class balance plots and a sample mel-spectrogram.
Saves all figures to outputs/results/.

Usage:
  python scripts/data_summary.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent   # meowmood/
sys.path.insert(0, str(ROOT))
from backend.utils import AUDIO_LABEL_MAP, IMAGE_CLASSES

IMAGE_DIR   = ROOT / "data" / "raw" / "images"
AUDIO_DIR   = ROOT / "data" / "raw" / "audio"
RESULTS_DIR = ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SR = 22050


# ── Image class distribution ───────────────────────────────────────────────────
def plot_image_distribution():
    counts = {}
    for split in ["train", "valid"]:
        for cls in IMAGE_CLASSES:
            p = IMAGE_DIR / split / cls
            if p.exists():
                counts.setdefault(cls, {})[split] = len(list(p.glob("*.jpg")))

    df = pd.DataFrame(counts).T.fillna(0).astype(int)
    ax = df.plot(kind="bar", figsize=(11, 5), color=["#FF6B6B", "#4ECDC4"], edgecolor="white")
    ax.set_title("Image Dataset — Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Number of Images")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = RESULTS_DIR / "eda_image_distribution.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")
    print(df.to_string())


# ── Audio label distribution ───────────────────────────────────────────────────
def plot_audio_distribution():
    label_counts: dict = {}
    for wav in AUDIO_DIR.glob("*.wav"):
        prefix = wav.stem[0]
        if prefix in AUDIO_LABEL_MAP:
            lbl = AUDIO_LABEL_MAP[prefix]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    if not label_counts:
        print("[SKIP] No audio files found.")
        return

    series = pd.Series(label_counts).sort_values(ascending=False)
    colors = ["#A29BFE", "#FD79A8", "#FDCB6E"]
    ax = series.plot(kind="bar", figsize=(7, 4), color=colors[:len(series)], edgecolor="white")
    ax.set_title("Audio Dataset — Label Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = RESULTS_DIR / "eda_audio_distribution.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")
    print(series.to_string())


# ── Sample mel-spectrogram ─────────────────────────────────────────────────────
def plot_sample_spectrogram():
    wav_files = list(AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        print("[SKIP] No audio files for spectrogram.")
        return
    y, sr = librosa.load(str(wav_files[0]), sr=SR)
    S     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db  = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(9, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                   ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Mel-Spectrogram — {wav_files[0].name}", fontsize=12)
    plt.tight_layout()
    out = RESULTS_DIR / "eda_sample_spectrogram.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("=== Image Distribution ===")
    plot_image_distribution()
    print("\n=== Audio Distribution ===")
    plot_audio_distribution()
    print("\n=== Sample Spectrogram ===")
    plot_sample_spectrogram()
    print("\nEDA complete.")
