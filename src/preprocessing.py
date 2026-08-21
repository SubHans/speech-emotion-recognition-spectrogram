"""
Stage 1: Preprocessing Module
Extracts Mel-Spectrogram features from audio files and formats dataset for deep learning.
"""

import os
import numpy as np
import pandas as pd
import librosa
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAVDESS_EMOTION_MAPPING = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


def extract_mel_spectrogram(file_path: str, sample_rate: int = 22050, n_mels: int = 128, fmax: int = 8000, resize_shape: tuple = (300, 300)) -> np.ndarray:
    """
    Ekstraksi fitur Mel-Spectrogram dari file audio .wav.
    """
    audio_data, sr = librosa.load(file_path, sr=sample_rate)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels, fmax=fmax)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    resized_spectrogram = resize(log_mel_spectrogram, resize_shape)
    return resized_spectrogram


def load_ravdess_dataset(dataset_path: str, sample_rate: int = 22050, n_mels: int = 128, fmax: int = 8000, resize_shape: tuple = (300, 300)) -> pd.DataFrame:
    """
    Membaca dan mengekstraksi seluruh fitur dari dataset RAVDESS.
    """
    emotions = []
    features = []

    print(f"Loading RAVDESS Dataset from: {dataset_path}")
    for root, _, files in os.walk(dataset_path):
        for file in tqdm(files, desc="Processing RAVDESS Files"):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion_label = RAVDESS_EMOTION_MAPPING.get(emotion_code, 'unknown')
                    if emotion_label != 'unknown':
                        spectrogram = extract_mel_spectrogram(file_path, sample_rate, n_mels, fmax, resize_shape)
                        features.append(spectrogram)
                        emotions.append(emotion_label)

    return pd.DataFrame({
        'spectrogram': features,
        'class': emotions
    })


def load_tess_dataset(dataset_path: str, sample_rate: int = 22050, n_mels: int = 128, fmax: int = 8000, resize_shape: tuple = (300, 300)) -> pd.DataFrame:
    """
    Membaca dan mengekstraksi seluruh fitur dari dataset TESS.
    """
    emotions = []
    features = []

    print(f"Loading TESS Dataset from: {dataset_path}")
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            emotion_label = folder.split('_')[-1].lower()
            for file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    spectrogram = extract_mel_spectrogram(file_path, sample_rate, n_mels, fmax, resize_shape)
                    features.append(spectrogram)
                    emotions.append(emotion_label)

    return pd.DataFrame({
        'spectrogram': features,
        'class': emotions
    })


def prepare_dataset(data_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Mengubah format spectrogram ke 3-channel RGB (300x300x3),
    melakukan Label Encoding, dan membagi dataset (Train/Test).
    """
    print("Preparing features tensor & label encoding...")
    X = np.array([s for s in data_df['spectrogram']])
    X = np.expand_dims(X, axis=-1)
    X = np.repeat(X, 3, axis=-1)  # Expand 1-channel to 3-channel RGB

    y = np.array(data_df['class'])
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    print(f"Data shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    return X_train, X_test, y_train, y_test, label_encoder
