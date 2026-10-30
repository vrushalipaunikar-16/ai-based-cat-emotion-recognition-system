"""
backend/train/audio_model.py
Train a Bidirectional LSTM for cat vocalization emotion classification.

Audio label mapping (from filename prefix):
  B_ → Normal   (brushing / calm context)
  F_ → Happy    (waiting for food)
  I_ → Sad      (isolation)

Steps:
  1. Load wav files from data/raw/audio/, extract MFCC features
  2. Split 80/20 train/test
  3. Build Bidirectional LSTM model
  4. Train with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  5. Evaluate: classification report + confusion matrix
  6. Save model  → models/audio_model.keras
  7. Save labels → models/audio_labels.json
  8. Save plots  → outputs/results/
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ── Allow imports from project root ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # meowmood/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from backend.utils import extract_mfcc, AUDIO_LABEL_MAP, N_MFCC, MAX_PAD

# ── Config ─────────────────────────────────────────────────────────────────────
AUDIO_DIR   = ROOT / "data" / "raw" / "audio"
MODEL_OUT   = ROOT / "models" / "audio_model.keras"
LABELS_OUT  = ROOT / "models" / "audio_labels.json"
RESULTS_DIR = ROOT / "outputs" / "results"
LOGS_DIR    = ROOT / "outputs" / "logs"
EPOCHS      = 60
BATCH_SIZE  = 16


# ── Step 1: Load dataset ───────────────────────────────────────────────────────
def load_dataset():
    """Scan audio dir, extract MFCCs, map labels from filename prefix."""
    X, y = [], []
    for wav in sorted(AUDIO_DIR.glob("*.wav")):
        prefix = wav.stem[0]
        if prefix not in AUDIO_LABEL_MAP:
            continue
        try:
            mfcc = extract_mfcc(str(wav))
            X.append(mfcc)
            y.append(AUDIO_LABEL_MAP[prefix])
        except Exception as e:
            print(f"[WARN] Skipping {wav.name}: {e}")
    return np.array(X, dtype=np.float32), y


# ── Step 3: Model architecture ─────────────────────────────────────────────────
def build_model(input_shape: tuple, num_classes: int) -> Model:
    """
    Bidirectional LSTM classifier.
    input_shape: (MAX_PAD, N_MFCC) = (174, 40)
    """
    inp = Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inp, outputs=out)


# ── Step 4: Training ───────────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_OUT.parent, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Step 1: load
    X, y_str = load_dataset()
    labels = sorted(set(y_str))
    label2idx = {l: i for i, l in enumerate(labels)}
    y = np.array([label2idx[l] for l in y_str])

    print(f"Audio classes: {labels}")
    print(f"Dataset size: {X.shape}")

    # Save label list
    with open(LABELS_OUT, "w") as f:
        json.dump(labels, f)

    # Step 2: split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    num_classes = len(labels)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes)

    model = build_model((MAX_PAD, N_MFCC), num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.3, patience=5, monitor="val_loss", verbose=1),
        ModelCheckpoint(str(MODEL_OUT), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.CSVLogger(str(LOGS_DIR / "audio_training_log.csv")),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # ── Step 5: Evaluation ─────────────────────────────────────────────────────
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels,
                yticklabels=labels, cmap="Blues")
    plt.title("Audio Model — Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "audio_confusion_matrix.png", dpi=120)
    plt.close()

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "audio_training_curves.png", dpi=120)
    plt.close()

    print(f"\nModel saved  → {MODEL_OUT}")
    print(f"Labels saved → {LABELS_OUT}")
    print(f"Results saved → {RESULTS_DIR}")


if __name__ == "__main__":
    train()
