import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models"

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.training.eeg_dataset import EEGSequenceDataset
from src.models.gcn_gru import EEGGCNGRU, create_edge_index


# ============================================================
# FILES
# ============================================================

TEST_FILE = DATA_PATH / "deap_test.npz"

CHECKPOINT_FILE = (
    MODEL_PATH / "gcn_gru_valence_balanced_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BALANCED GCN + GRU — TEST EVALUATION")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    if not CHECKPOINT_FILE.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{CHECKPOINT_FILE}"
        )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    test_data = np.load(TEST_FILE)

    X_test = test_data["features"]
    y_test = test_data["valence"]

    print()
    print("=" * 70)
    print("TEST DATA")
    print("=" * 70)

    print(
        "Test features:",
        X_test.shape
    )

    print(
        "Test labels:",
        y_test.shape
    )

    print()
    print("Test class distribution:")

    print(
        "Class 0:",
        np.sum(y_test == 0)
    )

    print(
        "Class 1:",
        np.sum(y_test == 1)
    )

    # ========================================================
    # DATASET
    # ========================================================

    test_dataset = EEGSequenceDataset(
        X_test,
        y_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # GRAPH
    # ========================================================

    edge_index, adjacency = create_edge_index(
        device
    )

    print()
    print("=" * 70)
    print("EEG GRAPH")
    print("=" * 70)

    print(
        "Nodes:",
        adjacency.shape[0]
    )

    print(
        "Edge index:",
        edge_index.shape
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = EEGGCNGRU(
        input_features=10,
        gcn_hidden=64,
        gcn_embedding=32,
        gru_hidden=64,
        gru_layers=2,
        num_classes=2,
        dropout=0.3
    ).to(device)

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print("=" * 70)
    print("CHECKPOINT")
    print("=" * 70)

    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation accuracy:",
        f"{checkpoint['val_accuracy']:.4f}"
    )

    print(
        "Validation balanced accuracy:",
        f"{checkpoint['val_balanced_accuracy']:.4f}"
    )

    print(
        "Validation Macro F1:",
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # ========================================================
    # TEST PREDICTION
    # ========================================================

    predictions = []
    targets = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            outputs = model(
                X_batch,
                edge_index
            )

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                y_batch.numpy()
            )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        targets,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            targets,
            predictions
        )
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        targets,
        predictions
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy  : {balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1           : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1        : {weighted_f1:.4f}"
    )

    print(
        f"Macro Precision    : {precision:.4f}"
    )

    print(
        f"Macro Recall       : {recall:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "                 Predicted"
    )

    print(
        "                 Low   High"
    )

    print(
        f"Actual Low       "
        f"{cm[0,0]:4d}  {cm[0,1]:4d}"
    )

    print(
        f"Actual High      "
        f"{cm[1,0]:4d}  {cm[1,1]:4d}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            targets,
            predictions,
            target_names=[
                "Low Valence",
                "High Valence"
            ],
            digits=4,
            zero_division=0
        )
    )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    unique, counts = np.unique(
        predictions,
        return_counts=True
    )

    print("=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    for class_id, count in zip(
        unique,
        counts
    ):

        label = (
            "Low Valence"
            if class_id == 0
            else "High Valence"
        )

        print(
            f"{label}: {count}"
        )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    baseline_predictions = np.zeros_like(
        y_test
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    baseline_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            baseline_predictions
        )
    )

    baseline_macro_f1 = f1_score(
        y_test,
        baseline_predictions,
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 70)
    print("MAJORITY BASELINE COMPARISON")
    print("=" * 70)

    print(
        f"Baseline Accuracy          : "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Baseline Balanced Accuracy : "
        f"{baseline_balanced_accuracy:.4f}"
    )

    print(
        f"Baseline Macro F1          : "
        f"{baseline_macro_f1:.4f}"
    )

    print()
    print("Balanced GCN + GRU:")

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{macro_f1:.4f}"
    )

    # ========================================================
    # END
    # ========================================================

    print()
    print("=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()