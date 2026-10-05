# 🧠 EEG Emotion Recognition Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Machine Learning](https://img.shields.io/badge/ML-GCN%20%7C%20GRU-green)

A spatial-temporal representation learning framework for EEG-based emotion recognition. This project provides an interactive web dashboard to process, visualize, and classify human emotions from EEG signals using Graph Convolutional Networks (GCN) and Gated Recurrent Units (GRU).

## 🚀 Live Demo
**[Play with the Live App on Streamlit Cloud](https://eeg-emotion-recognition-5hj2sba95stfzywvpecpyf.streamlit.app/)**

## ✨ Features
* **Interactive Dashboard:** Built with Streamlit for a seamless user experience.
* **Multi-Dataset Support:** Compatible with **DEAP** and **SEED** datasets.
* **Spatial-Temporal Learning:** Uses GCN to capture spatial brain electrode relationships and GRU for temporal signal sequences.
* **Automated Preprocessing:** Handles filtering, differential entropy extraction, and PSD.
* **Real-time Visualization:** View topographic maps, electrode graphs, and signal properties interactively.
* **Downloadable Predictions:** Easily export emotion classification results as CSV.

## 📁 Repository Structure
* pp.py: Main dashboard entry point.
* pages/: UI screens for Dataset Upload, Visualization, Feature Analysis, Models, and Evaluation.
* src/preprocessing/: Data loaders and signal filtering logic.
* src/features/: Extraction of PSD and Differential Entropy (DE) features.
* src/models/: Implementation of the GCN and GRU architectures.
* src/training/: Training and evaluation scripts.

## 🛠️ Local Setup
1. **Clone the repository:**
   `ash
   git clone https://github.com/Bhumika13-web/EEG-Emotion-Recognition.git
   cd EEG-Emotion-Recognition
   `
2. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   `
3. **Run the application:**
   `ash
   streamlit run app.py
   `

## 🤝 Contributors
* Bhumika
* Arsh
