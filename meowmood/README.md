# 🐱 MeowMood — AI Cat Emotion Recognition

Detect cat emotions from facial images and vocalizations using deep learning.

## Project Structure

```
meowmood/
├── backend/
│   ├── train/
│   │   ├── image_model.py     # CNN (MobileNetV2) training
│   │   └── audio_model.py     # BiLSTM training for audio
│   ├── app.py                 # FastAPI with /predict_image & /predict_audio
│   └── utils.py               # Shared preprocessing helpers
├── data/
│   ├── raw/
│   │   ├── images/            # Original image dataset (train/ + valid/)
│   │   └── audio/             # Raw .wav files
│   ├── processed/
│   │   ├── images/            # Resized images
│   │   └── audio/             # Extracted MFCC .npy files
│   └── labels.csv             # File → emotion label mapping
├── frontend/
│   └── app.py                 # Streamlit UI
├── models/
│   ├── image_model.keras      # Saved CNN weights
│   ├── audio_model.keras      # Saved LSTM weights
│   └── audio_labels.json      # Audio class label list
├── outputs/
│   ├── logs/                  # Training CSV logs
│   └── results/               # Confusion matrices, training curves, EDA plots
├── scripts/
│   ├── preprocess_images.py   # Resize & copy images to processed/
│   ├── preprocess_audio.py    # Extract MFCCs → .npy files + labels.csv
│   └── data_summary.py        # EDA plots (class balance, spectrogram)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy datasets into data/raw/
#    images → data/raw/images/train/<class>/ and data/raw/images/valid/<class>/
#    audio  → data/raw/audio/*.wav

# 3. (Optional) Preprocess
python scripts/preprocess_images.py
python scripts/preprocess_audio.py
python scripts/data_summary.py

# 4. Train models
python backend/train/image_model.py
python backend/train/audio_model.py

# 5. Run the Streamlit app
streamlit run frontend/app.py

# 6. (Optional) Run the FastAPI backend
uvicorn backend.app:app --reload
```

## Emotion Classes

| Image Model (7 classes) | Audio Model (3 classes) |
|-------------------------|-------------------------|
| Angry, Disgusted, Happy, Normal, Sad, Scared, Surprised | Normal, Happy, Sad |
