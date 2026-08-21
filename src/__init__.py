"""
Speech Emotion Recognition (SER) Package
"""

from .preprocessing import extract_mel_spectrogram, load_ravdess_dataset, load_tess_dataset, prepare_dataset
from .models import build_efficientnet_base, fine_tune_efficientnet, extract_deep_features, optimize_xgboost_optuna, train_final_xgboost
from .utils import visualize_spectrogram, plot_training_history, plot_confusion_matrix, print_evaluation_report

__all__ = [
    'extract_mel_spectrogram',
    'load_ravdess_dataset',
    'load_tess_dataset',
    'prepare_dataset',
    'build_efficientnet_base',
    'fine_tune_efficientnet',
    'extract_deep_features',
    'optimize_xgboost_optuna',
    'train_final_xgboost',
    'visualize_spectrogram',
    'plot_training_history',
    'plot_confusion_matrix',
    'print_evaluation_report'
]
