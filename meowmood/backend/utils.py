"""
backend/utils.py
Helper functions for preprocessing images and audio used by both
training scripts and the FastAPI backend.
"""

import os
import json
import tempfile
import numpy as np
import librosa
from pathlib import Path
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE  = (128, 128)
SR        = 22050
N_MFCC    = 40
MAX_PAD   = 174   # fixed time-steps for LSTM

IMAGE_CLASSES = ["Angry", "Disgusted", "Happy", "Normal", "Sad", "Scared", "Surprised"]

# Audio filename prefix → emotion label
# B_ = Brushing context → Normal
# F_ = Waiting for food → Happy
# I_ = Isolation        → Sad
AUDIO_LABEL_MAP = {"B": "Normal", "F": "Happy", "I": "Sad"}


# ── Image helpers ──────────────────────────────────────────────────────────────

def preprocess_image_array(img: Image.Image) -> np.ndarray:
    """Resize, convert to RGB, normalise → (1, H, W, 3) ready for CNN."""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis]   # add batch dim


# ── Audio helpers ──────────────────────────────────────────────────────────────

def extract_mfcc(file_path: str, n_mfcc: int = N_MFCC, max_pad: int = MAX_PAD) -> np.ndarray:
    """
    Load a wav file and extract MFCC features.
    Returns array of shape (max_pad, n_mfcc) — time-first for LSTM.
    """
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)   # (n_mfcc, T)
    if mfcc.shape[1] < max_pad:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad]
    return mfcc.T   # (max_pad, n_mfcc)


def preprocess_audio_bytes(wav_bytes: bytes) -> np.ndarray:
    """
    Accept raw wav bytes, extract MFCC features.
    Returns array of shape (1, max_pad, n_mfcc) ready for LSTM.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        feat = extract_mfcc(tmp_path)
    finally:
        os.unlink(tmp_path)
    return feat[np.newaxis].astype(np.float32)   # (1, MAX_PAD, N_MFCC)


def load_audio_labels(labels_path: str) -> list:
    """Load the saved audio label list from JSON."""
    with open(labels_path) as f:
        return json.load(f)
