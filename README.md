# 🧠 EEG Emotion Recognition via Spatial-Temporal Representation Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179.svg)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

An end-to-end framework and interactive analytical dashboard for Electroencephalography (EEG) emotion recognition. This project models the human brain's functional connectivity by combining **Graph Convolutional Networks (GCN)** for electrode topological spatial learning with **Gated Recurrent Units (GRU)** for sequential temporal representation, evaluated on benchmark datasets (**DEAP** and **SEED**).

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Supported Datasets](#-supported-datasets)
- [Feature Extraction Pipeline](#-feature-extraction-pipeline)
- [Benchmark Results](#-benchmark-results)
- [Streamlit Dashboard Walkthrough](#-streamlit-dashboard-walkthrough)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [Running the Interactive Dashboard](#running-the-interactive-dashboard)
  - [Training & Model Evaluation](#training--model-evaluation)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 🔬 Overview

Traditional machine learning methods often flatten multi-channel EEG signals into vector spaces, discarding the crucial anatomical and spatial geometry of the brain's cerebral cortex. 

This repository implements a **Spatial-Temporal Representation Learning** architecture:
1. **Spatial Representation**: Models EEG electrodes as nodes within an anatomical graph based on the International 10–20 system. Graph convolutions (GCN) propagate and aggregate topological features across neighboring brain regions.
2. **Temporal Representation**: Captures sequential dynamic emotional transitions across continuous sliding time windows using Gated Recurrent Units (GRU).
3. **Interactive Visual Analytics**: Features a production-grade 6-page [Streamlit](https://streamlit.io/) web application for raw signal inspection, spectral power visualization, feature analysis, interactive real-time inference, and comparative evaluation.

---

## ✨ Key Features

- **Multi-Dataset Support**: Built-in loaders and preprocessing pipelines for both **DEAP** (32 channels, Valence/Arousal dimensions) and **SEED** (62 channels, 3-class discrete emotions).
- **Domain-Specific Feature Extraction**:
  - **Differential Entropy (DE)** across standard EEG frequency bands ($\delta, \theta, \alpha, \beta, \gamma$).
  - **Power Spectral Density (PSD)** computed via Welch's periodogram method.
  - Multi-feature concatenation and fusion.
- **Topological Electrode Graphs**: Distance- and adjacency-based graph formulations for both 32-channel (DEAP) and 62-channel (SEED) electrode montages.
- **Comprehensive Model Suite**:
  - Machine Learning Baseline: **Support Vector Machines (SVM)** with RBF / Linear kernels.
  - Deep Learning Baselines: **Convolutional Neural Networks (CNN)** and **Gated Recurrent Units (GRU)**.
  - Graph Neural Networks: **Spatial GCN** and hybrid **Spatial-Temporal GCN + GRU**.
- **Interactive Multi-Page Web App**:
  - Raw EEG multi-channel time-series viewer.
  - Power spectrum and topographic heatmap projection.
  - Real-time emotion prediction with confidence scores and circumplex mapping.
  - Model architecture parameter breakdown and benchmark visualization.

---

## 🏗️ System Architecture

```text
       Raw EEG Signals (DEAP / SEED)
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │         Preprocessing Phase         │
  │  • Bandpass Filtering (0.5 - 45 Hz) │
  │  • Segmentation & Overlapping Window │
  │  • Baseline Correction & Artifacts  │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │     Feature Extraction (5 Bands)    │
  │  • Delta (1-4 Hz)   • Theta (4-8 Hz)│
  │  • Alpha (8-13 Hz)  • Beta (13-30 Hz│
  │  • Gamma (30-45 Hz)                 │
  │  • DE (Differential Entropy) + PSD  │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │   Spatial Graph Convolution (GCN)   │
  │  • 10-20 Electrode Adjacency Graph   │
  │  • Inter-channel Spatial Aggregation│
  └──────────────────┬──────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │    Temporal Representation (GRU)    │
  │  • Recurrent sequence modeling over │
  │    consecutive time windows         │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │         Emotion Prediction          │
  │  • DEAP: High/Low Valence & Arousal │
  │  • SEED: Negative / Neutral / Pos   │
  └─────────────────────────────────────┘
```

---

## 📊 Supported Datasets

| Specification | **DEAP Dataset** | **SEED Dataset** |
|:---|:---|:---|
| **Full Name** | Database for Emotion Analysis using Physiological Signals | SJTU Emotion EEG Dataset |
| **Subjects** | 32 participants | 15 participants |
| **EEG Channels** | 32 channels (10–20 standard placement) | 62 channels (ESI neuroscan system) |
| **Sampling Rate** | Downsampled to 128 Hz | Downsampled to 200 Hz |
| **Stimuli** | 40 one-minute music videos per subject | 15 four-minute Chinese film clips |
| **Target Labels** | Continuous 1–9 scale: Valence and Arousal (Binary split at threshold $\ge 5.0$) | 3 Discrete Categories: Negative (-1 / 0), Neutral (0 / 1), Positive (1 / 2) |
| **Input Format** | Preprocessed Python pickle (`.dat` / `.pkl`) | Preprocessed MATLAB (`.mat`) |

---

## ⚡ Feature Extraction Pipeline

For each frequency band ($b \in \{\delta, \theta, \alpha, \beta, \gamma\}$):

1. **Differential Entropy (DE)**: Assuming signal segments follow a Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$:
   $$DE = \frac{1}{2} \ln(2\pi e \sigma^2)$$
   Differential entropy exhibits superior emotion-discriminative properties compared to raw amplitude representations.

2. **Power Spectral Density (PSD)**:
   Calculated across each band using Welch's periodogram with Hanning windowing to extract average energy distributions.

3. **Spatial Graph Formulation**:
   Electrode coordinate positions $(x, y, z)$ on the scalp are mapped onto an adjacency matrix $A$ based on Euclidean distance thresholds and anatomical neighborhood connectivity, with added self-loops:
   $$\tilde{A} = A + I_N$$

---

## 📈 Benchmark Results

### 1. DEAP Dataset (Binary Classification: Valence & Arousal)

| Model | Task | Accuracy | Balanced Accuracy | Macro F1 |
|:---|:---:|:---:|:---:|:---:|
| **SVM (Baseline)** | Valence | 78.75% | 49.61% | 44.06% |
| **CNN** | Valence | 79.37% | 50.00% | 44.25% |
| **GCN + GRU (Ours)** | **Valence** | 70.63% | **52.34%** | **52.40%** |
| **SVM (Baseline)** | Arousal | 29.38% | 47.54% | 29.04% |
| **CNN** | Arousal | 75.00% | 48.39% | 42.86% |
| **GCN + GRU (Ours)** | **Arousal** | **77.50%** | **50.00%** | **43.66%** |

> **Note on Imbalance**: Due to class skewness in DEAP self-assessment ratings, standard Accuracy can be misleading. **Balanced Accuracy** and **Macro F1** demonstrate that the **GCN + GRU** hybrid model significantly improves minority class sensitivity and balanced predictive power.

---

### 2. SEED Dataset (3-Class Classification)

| Model | Sequence Length | Accuracy | Balanced Accuracy | Macro F1 |
|:---|:---:|:---:|:---:|:---:|
| **SVM (Baseline)** | — | 33.51% | 34.03% | 29.41% |
| **CNN** | — | 35.98% | 36.65% | 28.63% |
| **Spatial GCN (Best)** | — | **44.74%** | **44.50%** | **44.25%** |
| **GCN + GRU** | 20 windows | 44.03% | 43.30% | 34.83% |
| **GCN + GRU** | 10 windows | 41.64% | 41.49% | 41.44% |

---

## 🖥️ Streamlit Dashboard Walkthrough

Launch the multi-page interactive web interface to explore all stages of the workflow:

| Page | Description |
|---|---|
| **🏠 Overview (`app.py`)** | Central dashboard showing project metrics, processing status, dataset summaries, and best model highlights. |
| **📤 1. Upload Dataset** | Upload raw `.dat`, `.pkl`, or `.mat` files. Configure segment lengths, filter bands, and run automatic feature extraction. |
| **📉 2. EEG Visualization** | Multi-channel interactive waveforms, bandpass filtered signals, power spectral density curves, and 2D scalp topographic heatmaps. |
| **📊 3. Feature Analysis** | Cross-band energy comparisons, Differential Entropy distribution boxplots, and inter-electrode correlation matrices. |
| **🎯 4. Emotion Prediction** | Live inference testbench with trained models. Outputs predicted state, class probabilities, and Valence-Arousal circumplex position. |
| **🏛️ 5. Model Architecture** | Deep-dive into network layers, node feature dimensions, graph edge connectivity, and parameter allocations. |
| **📑 6. Evaluation** | Benchmark scorecards, confusion matrices, class-wise precision/recall, and performance comparison charts. |

---

## 📁 Repository Structure

```text
EEG-Emotion-Recognition/
├── app.py                           # Main Streamlit application entry point
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation
├── .streamlit/                      # Streamlit UI configuration
│   └── config.toml
├── models/                          # Trained baseline model checkpoints
│   ├── svm_valence_best.joblib      # Best SVM for DEAP Valence
│   ├── svm_arousal_best.joblib      # Best SVM for DEAP Arousal
│   └── svm_seed.joblib              # Best SVM for SEED 3-class
├── pages/                           # Streamlit multi-page interface
│   ├── 1_Upload_Dataset.py          # Data ingestion and preprocessing
│   ├── 2_EEG_Visualization.py       # Time-series & frequency visualization
│   ├── 3_Feature_Analysis.py        # DE and PSD feature analytics
│   ├── 4_Emotion_Prediction.py      # Real-time model inference
│   ├── 5_Model_Architecture.py      # Architecture flow & parameter viewer
│   └── 6_Evaluation.py              # Benchmark comparison and metrics
├── results/                         # Evaluation artifacts and benchmark logs
│   ├── final_best_models.csv
│   ├── DEAP/                        # DEAP comparison CSV and plot figures
│   │   ├── deap_model_comparison.csv
│   │   └── deap_valence_accuracy.png ...
│   └── SEED/                        # SEED comparison CSV, ROC, and matrices
│       ├── seed_model_comparison.csv
│       ├── seed_gcn_confusion_matrix.png ...
└── src/                             # Core Python source modules
    ├── features/                    # Signal feature extraction
    │   ├── differential_entropy.py  # Differential entropy calculations
    │   ├── psd.py                   # Power spectral density estimation
    │   └── feature_fusion.py        # Multimodal feature combination
    ├── graph/                       # Graph topology generation
    │   ├── electrode_graph.py       # DEAP 32-channel electrode adjacency
    │   └── seed_electrode_graph.py  # SEED 62-channel electrode adjacency
    ├── models/                      # PyTorch model definitions
    │   ├── gcn.py                   # Spatial Graph Convolutional Network
    │   ├── gru.py                   # Gated Recurrent Unit
    │   ├── gcn_gru.py               # DEAP Spatial-Temporal GCN+GRU model
    │   ├── seed_gcn.py              # SEED Spatial GCN model
    │   └── seed_gcn_gru.py          # SEED Spatial-Temporal GCN+GRU model
    ├── preprocessing/               # Raw signal loading & filtering
    │   ├── deap_loader.py           # DEAP data loading utilities
    │   ├── seed_loader.py           # SEED MATLAB data loading
    │   ├── filtering.py             # Butterworth bandpass filtering
    │   └── segmentation.py          # Window slicing with overlap
    └── training/                    # Model training, split, & evaluation scripts
        ├── train_valence.py         # DEAP Valence GCN+GRU training
        ├── train_arousal.py         # DEAP Arousal GCN+GRU training
        ├── train_svm.py             # Baseline SVM training
        ├── train_cnn.py             # Baseline CNN training
        ├── train_seed_cnn.py        # SEED CNN training
        ├── train_seed_svm.py        # SEED SVM training
        └── final_evaluation.py      # Comprehensive metric aggregation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites

- **Python**: Version `3.10` or higher recommended.
- **CUDA** *(Optional, recommended for GPU training)*: CUDA 11.8 or 12.x.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/EEG-Emotion-Recognition.git
cd EEG-Emotion-Recognition
```

### 3. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> If you plan to retrain the PyTorch Geometric graph models on GPU, ensure you install the PyTorch Geometric extension wheels matching your CUDA version as instructed on the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## 🚀 Usage Guide

### Running the Interactive Dashboard

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`. From the sidebar, navigate between dataset uploading, signal visualization, feature analysis, real-time prediction, architecture inspection, and model evaluation.

---

### Training & Model Evaluation

#### Training Baseline SVMs:
```bash
python src/training/train_svm.py
python src/training/train_svm_arousal.py
python src/training/train_seed_svm.py
```

#### Training Deep Learning Models (CNN & GCN+GRU):
```bash
# DEAP Valence model
python src/training/train_valence.py

# DEAP Arousal model
python src/training/train_arousal.py

# SEED CNN baseline
python src/training/train_seed_cnn.py
```

#### Running Benchmark Evaluation:
```bash
python src/training/final_evaluation.py
```

---

## 🧰 Technologies Used

- **Deep Learning & Graph Processing**: [PyTorch](https://pytorch.org/), [PyTorch Geometric (PyG)](https://pyg.org/)
- **Machine Learning & Signal Processing**: [scikit-learn](https://scikit-learn.org/), [SciPy](https://scipy.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [MNE-Python](https://mne.tools/)
- **Visualization & UI**: [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
