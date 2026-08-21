"""
Main Execution Script (CLI Entrypoint)
Pipeline Runner for Speech Emotion Recognition (SER).

Example Usage:
  python main.py --dataset ravdess --data_path ./ravdess-emotional-speech-audio --mode enhanced
  python main.py --dataset tess --data_path ./toronto-emotional-speech-set-tess --mode baseline
"""

import argparse
import sys
import numpy as np

from src.preprocessing import load_ravdess_dataset, load_tess_dataset, prepare_dataset
from src.models import (
    build_efficientnet_base,
    fine_tune_efficientnet,
    extract_deep_features,
    optimize_xgboost_optuna,
    train_final_xgboost
)
from src.utils import (
    plot_training_history,
    plot_confusion_matrix,
    print_evaluation_report,
    visualize_spectrogram
)


def parse_args():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition Pipeline Runner")

    parser.add_argument('--dataset', type=str, choices=['ravdess', 'tess'], default='ravdess',
                        help='Dataset to process (ravdess or tess)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory containing .wav audio files')
    parser.add_argument('--mode', type=str, choices=['baseline', 'enhanced'], default='enhanced',
                        help='Model pipeline mode: baseline (EfficientNetB3) or enhanced (EfficientNetB3 + Optuna + XGBoost)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs for EfficientNetB3 fine-tuning (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for model training (default: 32)')
    parser.add_argument('--n_trials', type=int, default=15,
                        help='Number of trials for Optuna hyperparameter optimization (default: 15)')
    parser.add_argument('--save_plots', action='store_true',
                        help='Flag to save evaluation plots as PNG files')

    return parser.parse_args()


def run_pipeline(args):
    print("=" * 70)
    print("           SPEECH EMOTION RECOGNITION (SER) PIPELINE RUNNER           ")
    print("=" * 70)
    print(f"Dataset Selected   : {args.dataset.upper()}")
    print(f"Dataset Path       : {args.data_path}")
    print(f"Pipeline Mode      : {args.mode.upper()}")
    print("=" * 70 + "\n")

    # STAGE 1: DATA LOADING & PREPROCESSING
    print(">>> STAGE 1: Loading & Preprocessing Audio Data...")
    if args.dataset == 'ravdess':
        df = load_ravdess_dataset(args.data_path)
    else:
        df = load_tess_dataset(args.data_path)

    if len(df) == 0:
        print(f"ERROR: No audio files found in directory '{args.data_path}'. Please check dataset path.")
        sys.exit(1)

    print(f"Successfully loaded {len(df)} audio samples.")

    # Visualize first sample
    visualize_spectrogram(
        df['spectrogram'].iloc[0],
        df['class'].iloc[0],
        save_path="sample_spectrogram.png" if args.save_plots else None
    )

    X_train, X_test, y_train, y_test, label_encoder = prepare_dataset(df)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    # STAGE 2: EFFICIENTNETB3 BASE TRAINING & FINE-TUNING
    print("\n>>> STAGE 2: Building & Fine-Tuning EfficientNetB3 Model...")
    model, base_model = build_efficientnet_base(
        input_shape=(300, 300, 3),
        num_classes=num_classes,
        dense_units=128,
        dropout_rate=0.3,
        learning_rate=1e-3
    )

    print("Initial training (Frozen base model)...")
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=args.batch_size
    )

    print("Fine-tuning top 20 layers of EfficientNetB3...")
    model = fine_tune_efficientnet(model, base_model, trainable_layers_from_top=20, learning_rate=1e-5)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # STAGE 3: EVALUATION / HYBRID CLASSIFICATION
    if args.mode == 'baseline':
        print("\n>>> STAGE 3: Evaluating Baseline EfficientNetB3 Model...")
        plot_training_history(history, save_path="training_history.png" if args.save_plots else None)

        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print_evaluation_report(y_test, y_pred, class_names)
        plot_confusion_matrix(
            y_test, y_pred, class_names,
            title="Baseline EfficientNetB3 Confusion Matrix",
            save_path="confusion_matrix_baseline.png" if args.save_plots else None
        )

    else:
        print("\n>>> STAGE 3: Enhanced Hybrid Pipeline (EfficientNetB3 Features + Optuna + XGBoost)...")

        # Deep Feature Extraction
        X_train_features, X_test_features = extract_deep_features(model, X_train, X_test)

        # Optuna Hyperparameter Optimization
        best_params = optimize_xgboost_optuna(
            X_train_features, y_train,
            X_test_features, y_test,
            n_trials=args.n_trials
        )

        # Train Final XGBoost Classifier
        final_xgb = train_final_xgboost(best_params, X_train_features, y_train)

        # Predict & Evaluate
        y_pred = final_xgb.predict(X_test_features)

        print_evaluation_report(y_test, y_pred, class_names)
        plot_confusion_matrix(
            y_test, y_pred, class_names,
            title="Enhanced EfficientNetB3 + XGBoost Confusion Matrix",
            save_path="confusion_matrix_enhanced.png" if args.save_plots else None
        )

    print("\n[SUCCESS] Pipeline execution complete!")


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
