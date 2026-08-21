# 🎙️ Speech Emotion Recognition (SER) using Mel-Spectrograms & Hybrid Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Tuning-blueviolet.svg)](https://optuna.org/)
[![Librosa](https://img.shields.io/badge/Audio-Librosa-orange.svg)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

An advanced **Speech Emotion Recognition (SER)** research codebase utilizing 2D Mel-Spectrogram feature representations, deep CNN transfer learning (**EfficientNetB3**), automated hyperparameter optimization via **Optuna**, and a hybrid gradient boosting classifier (**XGBoost**). 

This project benchmarks baseline deep learning models against an enhanced hybrid architecture across two popular speech emotion datasets: **RAVDESS** and **TESS**, achieving outstanding classification accuracy of up to **99.64%**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Datasets](#-datasets)
- [Experimental Results](#-experimental-results)
- [Project Structure](#-project-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [How to Run](#-how-to-run)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🔍 Overview

Speech Emotion Recognition (SER) plays a crucial role in Human-Computer Interaction (HCI), healthcare monitoring, customer sentiment analysis, and intelligent voice assistants. Standard 1D raw audio signals can be challenging to classify directly. 

This repository leverages **Mel-Spectrograms**—a 2D time-frequency representation of sound aligned with human hearing perception. By treating Mel-Spectrograms as image inputs, pre-trained Convolutional Neural Networks (**EfficientNetB3**) are employed for deep feature extraction. The extracted feature embeddings are then classified using an **Optuna-tuned XGBoost Classifier** for superior emotional discrimination.

---

## ✨ Key Features

- **Audio-to-Spectrogram Conversion**: Automated extraction of Mel-Spectrograms (`n_mels=128`, `fmax=8000`, `sr=22050 Hz`) converted to decibel scale (`power_to_db`) and expanded into 3-channel RGB image tensor inputs ($300 \times 300 \times 3$).
- **Deep Feature Extractor**: Fine-tuned **EfficientNetB3** pre-trained on ImageNet for rich multi-scale spatial feature learning.
- **Automated Hyperparameter Optimization**: Automated tuning of XGBoost hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`) powered by **Optuna**.
- **Hybrid Classifier (EfficientNetB3 + XGBoost)**: Combining CNN spatial representations with Gradient Boosted Decision Trees (GBDT) for state-of-the-art accuracy.
- **Modular Pipeline Architecture**: Clean modular Python package (`src/`) split stage-by-stage alongside full Jupyter Notebooks.
- **Multi-Dataset Benchmarking**: Comprehensive evaluations conducted on both **RAVDESS** and **TESS** benchmark datasets.

---

## 📐 System Architecture

The end-to-end classification pipeline follows the workflow below:

```mermaid
flowchart TD
    A["🎙️ Raw Audio File (.wav)"] --> B["🎼 Librosa Feature Extraction"]
    B --> C["📊 Mel-Spectrogram Generation (n_mels=128)"]
    C --> D["📐 Log Power-to-dB & Resize (300x300x3 RGB)"]
    D --> E["🧠 EfficientNetB3 Base Model"]
    
    subgraph Stage 1 & 2: Feature Extraction & Fine-Tuning
        E --> F["⚡ Fine-Tuning Top Layers (Adam Optimizer)"]
        F --> G["📥 Deep Feature Vector Extraction"]
    end
    
    subgraph Stage 3: Optimization & Classification
        G --> H["🔍 Optuna Hyperparameter Study"]
        H --> I["🌲 XGBoost Classifier Training"]
    end
    
    I --> J["🏷️ Emotion Prediction Output"]
```

---

## 📊 Datasets

Detailed download links and Kaggle API commands can be found in [`DATASETS.txt`](DATASETS.txt).

### 1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Total Classes (8)**: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- **Speakers**: 24 professional actors (12 female, 12 male)
- **Audio Format**: 16-bit, 48kHz `.wav`

### 2. TESS (Toronto Emotional Speech Set)
- **Total Classes (7)**: `angry`, `disgust`, `fear`, `happy`, `neutral`, `pleasant surprise`, `sad`
- **Speakers**: 2 female actors (26 years old and 64 years old)
- **Audio Format**: `.wav`

---

## 📈 Experimental Results

The models were evaluated using **Accuracy**, **Weighted F1-Score**, and **Test Loss** metrics.

| Dataset | Model Architecture | Pipeline / Methods | Test Accuracy | Weighted F1-Score |
| :--- | :--- | :--- | :---: | :---: |
| **RAVDESS** | Baseline | Fine-Tuned EfficientNetB3 + Dense Softmax | ~85.24% | 0.8529 |
| **RAVDESS** | **Enhanced (Hybrid)** | **EfficientNetB3 + Optuna + XGBoost** | **99.64%** | **0.9964** |
| **TESS** | Baseline | Fine-Tuned EfficientNetB3 + Dense Softmax | 99.11% | 0.9911 |
| **TESS** | **Enhanced (Hybrid)** | **EfficientNetB3 + Optuna + XGBoost** | **99.64%** | **0.9964** |

> **Key Takeaway**: The **Enhanced Hybrid Model** combining EfficientNetB3 feature embeddings with Optuna-tuned XGBoost achieved remarkable classification performance, boosting RAVDESS accuracy from **85.24%** to **99.64%**.

---

## 📁 Project Structure

```dir
.
├── src/
│   ├── __init__.py          # Python package initializer & exports
│   ├── preprocessing.py     # Stage 1: Mel-Spectrogram extraction & dataset loaders
│   ├── models.py            # Stage 2: EfficientNetB3 Transfer Learning & Optuna XGBoost
│   └── utils.py             # Stage 3: Evaluation metrics & plot visualizers
├── main.py                  # CLI Pipeline Runner script
├── requirements.txt         # Package dependencies file
├── DATASETS.txt             # Dataset download links & Kaggle CLI setup
├── sourcecode/              # Original Jupyter Notebooks
│   ├── Emotional_Rec_Base_RAVDES_Final.ipynb
│   ├── Emotional_Rec_RAVDES_Enhance_Final.ipynb
│   ├── Emotional_Rec_Base_TESS_Final.ipynb
│   └── Emotional_Rec_TESS_Enhance_Final.ipynb
└── README.md                # Documentation
```

---

## 💻 Prerequisites & Installation

### Requirements
- Python 3.8 or higher
- NVIDIA GPU recommended for faster model training (e.g., CUDA-enabled GPU or Google Colab T4 GPU)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/speech-emotion-recognition-spectrogram.git
cd speech-emotion-recognition-spectrogram
```

### 2. Install Dependencies
Install all required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Using the Modular CLI Script (`main.py`)

You can run the end-to-end training and evaluation pipeline directly from your terminal:

```bash
# Run Enhanced Hybrid Model on RAVDESS Dataset:
python main.py --dataset ravdess --data_path ./ravdess-emotional-speech-audio --mode enhanced --save_plots

# Run Baseline Model on TESS Dataset:
python main.py --dataset tess --data_path ./toronto-emotional-speech-set-tess --mode baseline
```

#### Command-Line Arguments:
- `--dataset`: Choose dataset (`ravdess` or `tess`).
- `--data_path`: Path to directory containing dataset `.wav` audio files.
- `--mode`: Execution mode (`enhanced` for EfficientNetB3 + Optuna XGBoost or `baseline` for EfficientNetB3 Softmax).
- `--epochs`: Training epochs for EfficientNetB3 fine-tuning (default: `20`).
- `--n_trials`: Number of trials for Optuna hyperparameter optimization (default: `15`).
- `--save_plots`: Flag to automatically save evaluation plots as PNG files.

---

### Option 2: Using Jupyter Notebooks (`.ipynb`)

1. Open any notebook in `sourcecode/` using Jupyter Lab, VS Code, or Google Colab.
2. Download datasets as instructed in [`DATASETS.txt`](DATASETS.txt).
3. Execute notebook cells sequentially.

---

## 📄 License & Acknowledgments

- Datasets used in this study belong to their respective creators:
  - RAVDESS: [Ryerson Audio-Visual Database of Emotional Speech and Song](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) (Licensed under CC-BY-NC-SA-4.0).
  - TESS: [Toronto Emotional Speech Set](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).
- EfficientNet architecture powered by [TensorFlow Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications).
- Hyperparameter tuning powered by [Optuna](https://optuna.org/).
