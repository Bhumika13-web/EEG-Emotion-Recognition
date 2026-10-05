import sys
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
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
# CONFIGURATION
# ============================================================

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deap_train.npz"
)

VAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deap_val.npz"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_MODEL_FILE = (
    MODEL_DIR
    / "gcn_gru_valence_best.pth"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_CLASSES = 2

RANDOM_SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


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
    "GCN + GRU VALENCE TRAINING"
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

print()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 60)

print("LOADING DATA")

print("=" * 60)

print()

print("Train file:")

print(
    TRAIN_FILE
)

print()

print("Validation file:")

print(
    VAL_FILE
)


train_data = np.load(
    TRAIN_FILE
)

val_data = np.load(
    VAL_FILE
)


# ============================================================
# EXTRACT DATA
# ============================================================

train_features = (
    train_data["features"]
)

train_labels = (
    train_data["valence"]
)

val_features = (
    val_data["features"]
)

val_labels = (
    val_data["valence"]
)


# ============================================================
# DATA SHAPES
# ============================================================

print()

print("=" * 60)

print("DATA SHAPES")

print("=" * 60)

print()

print(
    "Train features:",
    train_features.shape,
)

print(
    "Train labels:",
    train_labels.shape,
)

print(
    "Validation features:",
    val_features.shape,
)

print(
    "Validation labels:",
    val_labels.shape,
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()

print("=" * 60)

print("LABEL DISTRIBUTION")

print("=" * 60)

train_class_0 = int(
    np.sum(
        train_labels == 0
    )
)

train_class_1 = int(
    np.sum(
        train_labels == 1
    )
)

val_class_0 = int(
    np.sum(
        val_labels == 0
    )
)

val_class_1 = int(
    np.sum(
        val_labels == 1
    )
)

print()

print("Training:")

print(
    "Class 0:",
    train_class_0,
)

print(
    "Class 1:",
    train_class_1,
)

print()

print("Validation:")

print(
    "Class 0:",
    val_class_0,
)

print(
    "Class 1:",
    val_class_1,
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = (
    EEGSequenceDataset(
        train_features,
        train_labels,
    )
)

val_dataset = (
    EEGSequenceDataset(
        val_features,
        val_labels,
    )
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


print()

print("=" * 60)

print("DATALOADER")

print("=" * 60)

print()

print(
    "Training sequences:",
    len(train_dataset),
)

print(
    "Validation sequences:",
    len(val_dataset),
)

print(
    "Training batches:",
    len(train_loader),
)

print(
    "Validation batches:",
    len(val_loader),
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

print(
    "CREATING EEG ELECTRODE GRAPH"
)

print("=" * 60)


edge_index, adjacency = (
    create_edge_index(
        device
    )
)


print()

print(
    "Edge index shape:",
    edge_index.shape,
)

print(
    "Adjacency shape:",
    adjacency.shape,
)


# ============================================================
# CREATE MODEL
# ============================================================

print()

print("=" * 60)

print(
    "CREATING GCN + GRU MODEL"
)

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
# MODEL PARAMETERS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print()

print(
    "Total parameters:",
    f"{total_parameters:,}",
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}",
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = torch.tensor(
    [
        0.6226,
        2.5397,
    ],
    dtype=torch.float32,
    device=device,
)


print()

print("=" * 60)

print("CLASS WEIGHTS")

print("=" * 60)

print()

print(
    "Class 0 weight:",
    class_weights[0].item(),
)

print(
    "Class 1 weight:",
    class_weights[1].item(),
)


# ============================================================
# LOSS
# ============================================================

criterion = (
    nn.CrossEntropyLoss(
        weight=class_weights
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    for X, y in train_loader:

        X = X.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(
            X,
            edge_index,
        )

        loss = criterion(
            outputs,
            y,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            y
            .detach()
            .cpu()
            .numpy()
        )


    average_loss = (
        running_loss
        / len(train_loader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    return (
        average_loss,
        accuracy,
        f1,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for X, y in val_loader:

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

            loss = criterion(
                outputs,
                y,
            )

            running_loss += (
                loss.item()
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


    average_loss = (
        running_loss
        / len(val_loader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    return (
        average_loss,
        accuracy,
        f1,
    )


# ============================================================
# TRAINING
# ============================================================

print()

print("=" * 60)

print("STARTING TRAINING")

print("=" * 60)

print()


best_val_f1 = -1.0


for epoch in range(
    1,
    EPOCHS + 1,
):

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    (
        train_loss,
        train_accuracy,
        train_f1,
    ) = train_one_epoch()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        val_loss,
        val_accuracy,
        val_f1,
    ) = validate()


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"  Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"  Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"  Train F1: "
        f"{train_f1:.4f}"
    )

    print(
        f"  Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"  Val Accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"  Val F1: "
        f"{val_f1:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "val_f1":
                    val_f1,

                "val_accuracy":
                    val_accuracy,

                "input_features":
                    10,

                "gcn_hidden":
                    64,

                "gcn_embedding":
                    32,

                "gru_hidden":
                    64,

                "gru_layers":
                    2,

                "num_classes":
                    2,
            },
            BEST_MODEL_FILE,
        )

        print()

        print(
            "  ✓ BEST MODEL SAVED"
        )

        print(
            "  Path:",
            BEST_MODEL_FILE,
        )


    print()

    print("-" * 60)


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 60)

print("TRAINING COMPLETE")

print("=" * 60)

print()

print(
    "Best Validation F1:",
    f"{best_val_f1:.4f}",
)

print()

print(
    "Best model:"
)

print(
    BEST_MODEL_FILE
)

print()

print(
    "Valence training finished successfully."
)