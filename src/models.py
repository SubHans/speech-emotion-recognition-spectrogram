"""
Stage 2: Model Architecture & Training Module
Contains EfficientNetB3 Transfer Learning, Fine-Tuning, Deep Feature Extraction,
and Optuna Hyperparameter Optimization for XGBoost.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier
import optuna

# Disable unnecessary Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_efficientnet_base(input_shape: tuple = (300, 300, 3), num_classes: int = 8, dense_units: int = 128, dropout_rate: float = 0.3, learning_rate: float = 1e-3):
    """
    Membangun model dasar Transfer Learning EfficientNetB3 (ImageNet weights).
    """
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers initially

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def fine_tune_efficientnet(model: Model, base_model: Model, trainable_layers_from_top: int = 20, learning_rate: float = 1e-5):
    """
    Unfreeze N layer teratas dari EfficientNetB3 untuk fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers_from_top]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def extract_deep_features(model: Model, X_train: np.ndarray, X_test: np.ndarray):
    """
    Mengekstraksi vektor fitur deep learning dari model EfficientNetB3 yang telah dilatih.
    """
    print("Extracting deep features from EfficientNetB3...")
    X_train_features = model.predict(X_train, batch_size=32)
    X_test_features = model.predict(X_test, batch_size=32)
    return X_train_features, X_test_features


def optimize_xgboost_optuna(X_train_features: np.ndarray, y_train: np.ndarray, X_test_features: np.ndarray, y_test: np.ndarray, n_trials: int = 15, random_state: int = 42):
    """
    Tuning hyperparameter otomatis untuk XGBoost Classifier menggunakan Optuna.
    """
    print(f"Starting Optuna Hyperparameter Optimization ({n_trials} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'eval_metric': 'logloss',
            'random_state': random_state
        }

        xgb_model = XGBClassifier(**params)
        xgb_model.fit(X_train_features, y_train)
        y_pred = xgb_model.predict(X_test_features)
        accuracy = np.mean(y_pred == y_test)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters found:", study.best_params)
    print(f"Best trial validation accuracy: {study.best_value:.4f}")

    return study.best_params


def train_final_xgboost(best_params: dict, X_train_features: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> XGBClassifier:
    """
    Melatih model akhir XGBoost dengan hyperparameter terbaik hasil Optuna.
    """
    print("Training final XGBoost Classifier with best hyperparameters...")
    final_xgb = XGBClassifier(**best_params, random_state=random_state)
    final_xgb.fit(X_train_features, y_train)
    return final_xgb
