"""
Final Evaluation and Comparison
EEG Emotion Recognition using Spatial-Temporal Representation Learning

This script:
1. Stores the completed DEAP and SEED experiment results.
2. Creates comparison CSV files.
3. Creates Accuracy comparison plots.
4. Creates Balanced Accuracy comparison plots.
5. Creates Macro F1 comparison plots.
6. Creates SEED confusion matrix plots.
7. Creates a combined model comparison.

IMPORTANT:
- This script DOES NOT retrain any model.
- Values below are only results already obtained from the experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
DEAP_DIR = RESULTS_DIR / "DEAP"
SEED_DIR = RESULTS_DIR / "SEED"

DEAP_DIR.mkdir(parents=True, exist_ok=True)
SEED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEAP RESULTS
# ============================================================
#
# DEAP:
# Valence:
#   SVM:
#       Test Accuracy       = 0.7875
#       Balanced Accuracy   = 0.4961
#       Macro F1             = 0.4406
#
#   CNN:
#       Test Accuracy       = 0.7937
#       Balanced Accuracy   = 0.5000
#       Macro F1             = 0.4425
#
#   GCN + GRU:
#       Test Accuracy       = 0.7063
#       Balanced Accuracy   = 0.5234
#       Macro F1             = 0.5240
#
# Arousal:
#   SVM:
#       Test Accuracy       = 0.2938
#       Balanced Accuracy   = 0.4754
#       Macro F1             = 0.2904
#
#   CNN:
#       Test Accuracy       = 0.7500
#       Balanced Accuracy   = 0.4839
#       Macro F1             = 0.4286
#
#   GCN + GRU:
#       Test Accuracy       = 0.7750
#       Balanced Accuracy   = 0.5000
#       Macro F1             = 0.4366
#
# ============================================================

deap_results = pd.DataFrame([
    {
        "Model": "SVM",
        "Task": "Valence",
        "Accuracy": 0.7875,
        "Balanced Accuracy": 0.4961,
        "Macro F1": 0.4406,
    },
    {
        "Model": "CNN",
        "Task": "Valence",
        "Accuracy": 0.7937,
        "Balanced Accuracy": 0.5000,
        "Macro F1": 0.4425,
    },
    {
        "Model": "GCN + GRU",
        "Task": "Valence",
        "Accuracy": 0.7063,
        "Balanced Accuracy": 0.5234,
        "Macro F1": 0.5240,
    },
    {
        "Model": "SVM",
        "Task": "Arousal",
        "Accuracy": 0.2938,
        "Balanced Accuracy": 0.4754,
        "Macro F1": 0.2904,
    },
    {
        "Model": "CNN",
        "Task": "Arousal",
        "Accuracy": 0.7500,
        "Balanced Accuracy": 0.4839,
        "Macro F1": 0.4286,
    },
    {
        "Model": "GCN + GRU",
        "Task": "Arousal",
        "Accuracy": 0.7750,
        "Balanced Accuracy": 0.5000,
        "Macro F1": 0.4366,
    },
])


# ============================================================
# SEED RESULTS
# ============================================================
#
# Actual completed experiments:
#
# SVM:
#   Accuracy           = 0.3351
#   Balanced Accuracy  = 0.3403
#   Macro F1           = 0.2941
#
# CNN:
#   Accuracy           = 0.3598
#   Balanced Accuracy  = 0.3665
#   Macro F1           = 0.2863
#
# GCN:
#   Accuracy           = 0.4474
#   Balanced Accuracy  = 0.4450
#   Macro F1           = 0.4425
#
# GCN + GRU (20):
#   Accuracy           = 0.4403
#   Balanced Accuracy  = 0.4330
#   Macro F1           = 0.3483
#
# GCN + GRU (10):
#   Accuracy           = 0.4164
#   Balanced Accuracy  = 0.4149
#   Macro F1           = 0.4144
#
# ============================================================

seed_results = pd.DataFrame([
    {
        "Model": "SVM",
        "Sequence Length": "-",
        "Accuracy": 0.3351,
        "Balanced Accuracy": 0.3403,
        "Macro F1": 0.2941,
    },
    {
        "Model": "CNN",
        "Sequence Length": "-",
        "Accuracy": 0.3598,
        "Balanced Accuracy": 0.3665,
        "Macro F1": 0.2863,
    },
    {
        "Model": "GCN",
        "Sequence Length": "-",
        "Accuracy": 0.4474,
        "Balanced Accuracy": 0.4450,
        "Macro F1": 0.4425,
    },
    {
        "Model": "GCN + GRU",
        "Sequence Length": 20,
        "Accuracy": 0.4403,
        "Balanced Accuracy": 0.4330,
        "Macro F1": 0.3483,
    },
    {
        "Model": "GCN + GRU",
        "Sequence Length": 10,
        "Accuracy": 0.4164,
        "Balanced Accuracy": 0.4149,
        "Macro F1": 0.4144,
    },
])


# ============================================================
# SAVE CSV FILES
# ============================================================

deap_csv = DEAP_DIR / "deap_model_comparison.csv"
seed_csv = SEED_DIR / "seed_model_comparison.csv"

deap_results.to_csv(deap_csv, index=False)
seed_results.to_csv(seed_csv, index=False)

print("=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

print("\nDEAP RESULTS")
print("-" * 80)
print(deap_results.to_string(index=False))

print("\nSEED RESULTS")
print("-" * 80)
print(seed_results.to_string(index=False))

print(f"\nSaved: {deap_csv}")
print(f"Saved: {seed_csv}")


# ============================================================
# HELPER FUNCTION FOR BAR CHARTS
# ============================================================

def create_bar_chart(
    dataframe,
    model_column,
    metric,
    title,
    output_path,
    percentage=True
):
    plt.figure(figsize=(10, 6))

    values = dataframe[metric].values
    labels = []

    for _, row in dataframe.iterrows():
        model = str(row[model_column])

        if "Sequence Length" in dataframe.columns:
            seq = row["Sequence Length"]

            if seq != "-":
                model = f"{model}\n(T={seq})"

        labels.append(model)

    x = np.arange(len(labels))

    bars = plt.bar(x, values)

    plt.xticks(x, labels)
    plt.ylabel(metric)
    plt.title(title)

    if percentage:
        plt.ylim(0, 1)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value * 100:.2f}%",
                ha="center",
                va="bottom"
            )

    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


# ============================================================
# SEED PLOTS
# ============================================================

create_bar_chart(
    seed_results,
    "Model",
    "Accuracy",
    "SEED Model Accuracy Comparison",
    SEED_DIR / "seed_accuracy_comparison.png"
)

create_bar_chart(
    seed_results,
    "Model",
    "Balanced Accuracy",
    "SEED Balanced Accuracy Comparison",
    SEED_DIR / "seed_balanced_accuracy_comparison.png"
)

create_bar_chart(
    seed_results,
    "Model",
    "Macro F1",
    "SEED Macro F1 Comparison",
    SEED_DIR / "seed_macro_f1_comparison.png"
)


# ============================================================
# DEAP VALENCE
# ============================================================

deap_valence = deap_results[
    deap_results["Task"] == "Valence"
].copy()

create_bar_chart(
    deap_valence,
    "Model",
    "Accuracy",
    "DEAP Valence Accuracy Comparison",
    DEAP_DIR / "deap_valence_accuracy.png"
)

create_bar_chart(
    deap_valence,
    "Model",
    "Balanced Accuracy",
    "DEAP Valence Balanced Accuracy Comparison",
    DEAP_DIR / "deap_valence_balanced_accuracy.png"
)

create_bar_chart(
    deap_valence,
    "Model",
    "Macro F1",
    "DEAP Valence Macro F1 Comparison",
    DEAP_DIR / "deap_valence_macro_f1.png"
)


# ============================================================
# DEAP AROUSAL
# ============================================================

deap_arousal = deap_results[
    deap_results["Task"] == "Arousal"
].copy()

create_bar_chart(
    deap_arousal,
    "Model",
    "Accuracy",
    "DEAP Arousal Accuracy Comparison",
    DEAP_DIR / "deap_arousal_accuracy.png"
)

create_bar_chart(
    deap_arousal,
    "Model",
    "Balanced Accuracy",
    "DEAP Arousal Balanced Accuracy Comparison",
    DEAP_DIR / "deap_arousal_balanced_accuracy.png"
)

create_bar_chart(
    deap_arousal,
    "Model",
    "Macro F1",
    "DEAP Arousal Macro F1 Comparison",
    DEAP_DIR / "deap_arousal_macro_f1.png"
)


# ============================================================
# CONFUSION MATRIX PLOTTING
# ============================================================

def plot_confusion_matrix(
    matrix,
    class_names,
    title,
    output_path
):
    matrix = np.asarray(matrix)

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix, interpolation="nearest")

    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    threshold = matrix.max() / 2

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black"
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved confusion matrix: {output_path}")


# ============================================================
# SEED GCN CONFUSION MATRIX
# ============================================================

seed_gcn_cm = np.array([
    [1448, 792, 1120],
    [1104, 1104, 1104],
    [1324, 183, 2003]
])

plot_confusion_matrix(
    seed_gcn_cm,
    ["Negative", "Neutral", "Positive"],
    "SEED GCN Confusion Matrix",
    SEED_DIR / "seed_gcn_confusion_matrix.png"
)


# ============================================================
# SEED GCN + GRU (20) CONFUSION MATRIX
# ============================================================

seed_gcn_gru20_cm = np.array([
    [6, 40, 116],
    [0, 52, 104],
    [0, 12, 156]
])

plot_confusion_matrix(
    seed_gcn_gru20_cm,
    ["Negative", "Neutral", "Positive"],
    "SEED GCN + GRU (T=20) Confusion Matrix",
    SEED_DIR / "seed_gcn_gru20_confusion_matrix.png"
)


# ============================================================
# SEED GCN + GRU (10) CONFUSION MATRIX
# ============================================================

seed_gcn_gru10_cm = np.array([
    [142, 78, 110],
    [108, 108, 108],
    [140, 39, 166]
])

plot_confusion_matrix(
    seed_gcn_gru10_cm,
    ["Negative", "Neutral", "Positive"],
    "SEED GCN + GRU (T=10) Confusion Matrix",
    SEED_DIR / "seed_gcn_gru10_confusion_matrix.png"
)


# ============================================================
# BEST MODEL IDENTIFICATION
# ============================================================

best_seed = seed_results.loc[
    seed_results["Macro F1"].idxmax()
]

print("\n" + "=" * 80)
print("BEST SEED MODEL")
print("=" * 80)

print(f"Model: {best_seed['Model']}")

if best_seed["Sequence Length"] != "-":
    print(f"Sequence Length: {best_seed['Sequence Length']}")

print(
    f"Accuracy: {best_seed['Accuracy'] * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{best_seed['Balanced Accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{best_seed['Macro F1'] * 100:.2f}%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "Dataset": "SEED",
        "Best Model": "GCN",
        "Accuracy": 0.4474,
        "Balanced Accuracy": 0.4450,
        "Macro F1": 0.4425,
    }
])

summary_path = RESULTS_DIR / "final_best_models.csv"

summary.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 80)
print("FINAL BEST MODEL SUMMARY")
print("=" * 80)

print(summary.to_string(index=False))

print(f"\nSaved: {summary_path}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)