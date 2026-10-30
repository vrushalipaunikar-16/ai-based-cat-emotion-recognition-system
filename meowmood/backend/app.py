"""
backend/app.py
FastAPI backend with two prediction endpoints:
  POST /predict_image  — accepts an image file, returns emotion + confidence scores
  POST /predict_audio  — accepts a wav file,   returns emotion + confidence scores

Run:
  uvicorn backend.app:app --reload   (from meowmood/ directory)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# ── Allow imports from project root ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # meowmood/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from backend.utils import (
    preprocess_image_array,
    preprocess_audio_bytes,
    load_audio_labels,
    IMAGE_CLASSES,
)

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_IMG   = ROOT / "models" / "image_model.keras"
MODEL_AUD   = ROOT / "models" / "audio_model.keras"
LABELS_AUD  = ROOT / "models" / "audio_labels.json"

app = FastAPI(title="MeowMood API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model loading ─────────────────────────────────────────────────────────
_img_model  = None
_aud_model  = None
_aud_labels = None


def get_image_model():
    global _img_model
    if _img_model is None:
        if not MODEL_IMG.exists():
            raise HTTPException(status_code=503, detail="Image model not trained yet.")
        import tensorflow as tf
        _img_model = tf.keras.models.load_model(str(MODEL_IMG))
    return _img_model


def get_audio_model():
    global _aud_model, _aud_labels
    if _aud_model is None:
        if not MODEL_AUD.exists():
            raise HTTPException(status_code=503, detail="Audio model not trained yet.")
        import tensorflow as tf
        _aud_model  = tf.keras.models.load_model(str(MODEL_AUD))
        _aud_labels = load_audio_labels(str(LABELS_AUD))
    return _aud_model, _aud_labels


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "MeowMood API is running 🐱"}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)) -> Dict:
    """
    Accept a JPG/PNG image of a cat face.
    Returns predicted emotion label and per-class confidence scores.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are accepted.")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    arr = preprocess_image_array(img)

    model = get_image_model()
    probs = model.predict(arr, verbose=0)[0].tolist()
    top_idx = int(np.argmax(probs))

    return {
        "predicted_emotion": IMAGE_CLASSES[top_idx],
        "confidence": round(probs[top_idx] * 100, 2),
        "scores": {cls: round(p * 100, 2) for cls, p in zip(IMAGE_CLASSES, probs)},
    }


@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)) -> Dict:
    """
    Accept a WAV audio clip of a cat vocalization.
    Returns predicted emotion label and per-class confidence scores.
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV audio files are accepted.")

    wav_bytes = await file.read()
    feat = preprocess_audio_bytes(wav_bytes)

    model, labels = get_audio_model()
    probs = model.predict(feat, verbose=0)[0].tolist()
    top_idx = int(np.argmax(probs))

    return {
        "predicted_emotion": labels[top_idx],
        "confidence": round(probs[top_idx] * 100, 2),
        "scores": {lbl: round(p * 100, 2) for lbl, p in zip(labels, probs)},
    }
