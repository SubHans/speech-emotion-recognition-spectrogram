"""
Stage 3: Evaluation & Visualization Utility Module
Provides plotting utilities for Mel-Spectrograms, training history,
confusion matrices, and prints evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


def visualize_spectrogram(spectrogram: np.ndarray, label: str, sr: int = 22050, save_path: str = None):
    """
    Visualisasi plot gambar Mel-Spectrogram tunggal.
    """
    plt.figure(figsize=(10, 4))
    if len(spectrogram.shape) == 3:
        spec_display = spectrogram[:, :, 0]
    else:
        spec_display = spectrogram

    librosa.display.specshow(spec_display, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel-Spectrogram - Emotion: {label}")
    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved spectrogram plot to {save_path}")
    plt.show()


def plot_training_history(history, save_path: str = None):
    """
    Plot grafik pergerakan Akurasi dan Loss selama proses pelatihan.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', color='red', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved training history plot to {save_path}")
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, title: str = "Confusion Matrix", save_path: str = None):
    """
    Plot heatmap Confusion Matrix menggunakan Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix plot to {save_path}")
    plt.show()


def print_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list):
    """
    Menampilkan metrik evaluasi lengkap (Accuracy, Weighted F1-Score, Classification Report).
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\n" + "=" * 60)
    print("                      EVALUATION RESULTS                    ")
    print("=" * 60)
    print(f"Accuracy Score    : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Weighted F1-Score : {f1:.4f}")
    print("-" * 60)
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("=" * 60 + "\n")

    return acc, f1
