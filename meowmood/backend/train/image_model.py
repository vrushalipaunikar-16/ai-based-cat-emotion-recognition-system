"""
backend/train/image_model.py
Train a MobileNetV2-based CNN for 7-class cat facial emotion classification.

Steps:
  1. Load augmented image generators from data/raw/images (train + valid splits)
  2. Build MobileNetV2 with custom classification head
  3. Fine-tune top 30 layers; freeze the rest
  4. Train with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  5. Evaluate: classification report + confusion matrix
  6. Save model  → models/image_model.keras
  7. Save plots  → outputs/results/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Allow imports from project root ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # meowmood/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from backend.utils import IMG_SIZE, IMAGE_CLASSES

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_TRAIN  = ROOT / "data" / "raw" / "images" / "train"
DATA_VALID  = ROOT / "data" / "raw" / "images" / "valid"
MODEL_OUT   = ROOT / "models" / "image_model.keras"
RESULTS_DIR = ROOT / "outputs" / "results"
LOGS_DIR    = ROOT / "outputs" / "logs"
BATCH_SIZE  = 32
EPOCHS      = 30
NUM_CLASSES = len(IMAGE_CLASSES)


# ── Step 1: Data generators ────────────────────────────────────────────────────
def get_generators(batch_size: int = BATCH_SIZE):
    """Return augmented train and clean validation generators."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
    )
    valid_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_flow = train_gen.flow_from_directory(
        str(DATA_TRAIN),
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    valid_flow = valid_gen.flow_from_directory(
        str(DATA_VALID),
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_flow, valid_flow


# ── Step 2 & 3: Model architecture ────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES) -> Model:
    """MobileNetV2 backbone + custom head. Top 30 layers are trainable."""
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze all except the last 30 layers for fine-tuning
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=out)


# ── Step 4: Training ───────────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_OUT.parent, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    train_flow, valid_flow = get_generators()

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.3, patience=3, monitor="val_loss", verbose=1),
        ModelCheckpoint(str(MODEL_OUT), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.CSVLogger(str(LOGS_DIR / "image_training_log.csv")),
    ]

    history = model.fit(
        train_flow,
        validation_data=valid_flow,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # ── Step 5: Evaluation ─────────────────────────────────────────────────────
    valid_flow.reset()
    y_pred = np.argmax(model.predict(valid_flow), axis=1)
    y_true = valid_flow.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=IMAGE_CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=IMAGE_CLASSES,
                yticklabels=IMAGE_CLASSES, cmap="YlOrRd")
    plt.title("Image Model — Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "image_confusion_matrix.png", dpi=120)
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
    plt.savefig(RESULTS_DIR / "image_training_curves.png", dpi=120)
    plt.close()

    print(f"\nModel saved → {MODEL_OUT}")
    print(f"Results saved → {RESULTS_DIR}")


if __name__ == "__main__":
    train()
