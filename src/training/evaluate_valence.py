import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(
    str(SRC_DIR)
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from models.gcn_gru import (
    EEGGCNGRU,
    create_edge_index,
)

from training.eeg_dataset import (
    EEGSequenceDataset,
)


# ============================================================
# FILE PATHS
# ============================================================

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deap_test.npz"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "gcn_gru_valence_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8

NUM_CLASSES = 2


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 60)

print(
    "GCN + GRU VALENCE TEST EVALUATION"
)

print("=" * 60)

print()

print(
    "Device:",
    device,
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# CHECK FILES
# ============================================================

print()

print("=" * 60)

print("CHECKING FILES")

print("=" * 60)

print()

print(
    "Test dataset:"
)

print(
    TEST_FILE
)

print()

print(
    "Model checkpoint:"
)

print(
    MODEL_FILE
)


if not TEST_FILE.exists():

    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_FILE}"
    )


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_FILE}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print()

print("=" * 60)

print("LOADING TEST DATA")

print("=" * 60)


test_data = np.load(
    TEST_FILE
)


test_features = (
    test_data["features"]
)

test_labels = (
    test_data["valence"]
)


print()

print(
    "Test features:",
    test_features.shape,
)

print(
    "Test labels:",
    test_labels.shape,
)


# ============================================================
# TEST LABEL DISTRIBUTION
# ============================================================

test_class_0 = int(
    np.sum(
        test_labels == 0
    )
)

test_class_1 = int(
    np.sum(
        test_labels == 1
    )
)


print()

print("=" * 60)

print("TEST LABEL DISTRIBUTION")

print("=" * 60)

print()

print(
    "Class 0:",
    test_class_0,
)

print(
    "Class 1:",
    test_class_1,
)


# ============================================================
# CREATE DATASET
# ============================================================

test_dataset = (
    EEGSequenceDataset(
        test_features,
        test_labels,
    )
)


# ============================================================
# CREATE DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


print()

print("=" * 60)

print("TEST DATALOADER")

print("=" * 60)

print()

print(
    "Test sequences:",
    len(test_dataset),
)

print(
    "Test batches:",
    len(test_loader),
)

print(
    "Batch size:",
    BATCH_SIZE,
)


# ============================================================
# CREATE EEG GRAPH
# ============================================================

print()

print("=" * 60)

print("CREATING EEG ELECTRODE GRAPH")

print("=" * 60)


edge_index, adjacency = (
    create_edge_index(
        device
    )
)


print()

print(
    "Number of EEG nodes:",
    adjacency.shape[0],
)

print(
    "Adjacency shape:",
    adjacency.shape,
)

print(
    "Edge index shape:",
    edge_index.shape,
)


# ============================================================
# CREATE MODEL
# ============================================================

print()

print("=" * 60)

print("CREATING MODEL")

print("=" * 60)


model = EEGGCNGRU(
    input_features=10,
    gcn_hidden=64,
    gcn_embedding=32,
    gru_hidden=64,
    gru_layers=2,
    num_classes=NUM_CLASSES,
    dropout=0.3,
).to(device)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print()

print("=" * 60)

print("LOADING BEST MODEL")

print("=" * 60)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print()

print(
    "Checkpoint epoch:",
    checkpoint.get(
        "epoch",
        "N/A"
    ),
)

print(
    "Validation F1:",
    f"{checkpoint.get('val_f1', 0):.4f}",
)

print(
    "Validation Accuracy:",
    f"{checkpoint.get('val_accuracy', 0):.4f}",
)

print()

print(
    "Model loaded successfully."
)


# ============================================================
# EVALUATION
# ============================================================

print()

print("=" * 60)

print("RUNNING TEST EVALUATION")

print("=" * 60)


model.eval()


all_predictions = []

all_labels = []


with torch.no_grad():

    for batch_index, (X, y) in enumerate(
        test_loader
    ):

        X = X.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )


        outputs = model(
            X,
            edge_index,
        )


        predictions = torch.argmax(
            outputs,
            dim=1,
        )


        all_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )

        all_labels.extend(
            y
            .cpu()
            .numpy()
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0,
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0,
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1],
)


# ============================================================
# RESULTS
# ============================================================

print()

print("=" * 60)

print("FINAL TEST RESULTS")

print("=" * 60)

print()

print(
    f"Test Accuracy : {accuracy:.4f}"
)

print(
    f"Test F1 Score : {f1:.4f}"
)

print(
    f"Test Precision : {precision:.4f}"
)

print(
    f"Test Recall    : {recall:.4f}"
)


# ============================================================
# PERCENTAGE RESULTS
# ============================================================

print()

print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Test F1 Score : {f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()

print("=" * 60)

print("CONFUSION MATRIX")

print("=" * 60)

print()

print(
    "Rows    = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "             Predicted"
)

print(
    "             Low  High"
)

print(
    f"Actual Low   {cm[0,0]:3d}  {cm[0,1]:3d}"
)

print(
    f"Actual High  {cm[1,0]:3d}  {cm[1,1]:3d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()

print("=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print()

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Low Valence",
            "High Valence",
        ],
        zero_division=0,
    )
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

predicted_class_0 = int(
    np.sum(
        all_predictions == 0
    )
)

predicted_class_1 = int(
    np.sum(
        all_predictions == 1
    )
)


print()

print("=" * 60)

print("PREDICTION DISTRIBUTION")

print("=" * 60)

print()

print(
    "Predicted Class 0:",
    predicted_class_0,
)

print(
    "Predicted Class 1:",
    predicted_class_1,
)


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 60)

print("TEST EVALUATION COMPLETE")

print("=" * 60)

print()

print(
    "Best GCN + GRU model evaluated successfully."
)

print()

print(
    "Model:",
    MODEL_FILE,
)