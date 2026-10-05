import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.training.eeg_dataset import EEGSequenceDataset
from src.models.gcn_gru import EEGGCNGRU, create_edge_index


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 8
EPOCHS = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 6

NUM_CLASSES = 2

MODEL_OUTPUT = (
    MODEL_PATH / "gcn_gru_valence_balanced_best.pth"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train_file = DATA_PATH / "deap_train.npz"
    val_file = DATA_PATH / "deap_val.npz"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{train_file}"
        )

    if not val_file.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{val_file}"
        )

    train_data = np.load(train_file)
    val_data = np.load(val_file)

    X_train = train_data["features"]
    y_train = train_data["valence"]

    X_val = val_data["features"]
    y_val = val_data["valence"]

    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print("Train features      :", X_train.shape)
    print("Train labels        :", y_train.shape)

    print("Validation features :", X_val.shape)
    print("Validation labels   :", y_val.shape)

    print()

    print("Train class distribution:")
    print("Class 0:", np.sum(y_train == 0))
    print("Class 1:", np.sum(y_train == 1))

    print()

    print("Validation class distribution:")
    print("Class 0:", np.sum(y_val == 0))
    print("Class 1:", np.sum(y_val == 1))

    return X_train, y_train, X_val, y_val


# ============================================================
# CREATE BALANCED SAMPLER
# ============================================================

def create_balanced_sampler(labels):

    class_counts = np.bincount(labels)

    print()
    print("Class counts:", class_counts)

    # Inverse-frequency class weights
    class_weights = 1.0 / class_counts

    # Assign weight to every training sample
    sample_weights = class_weights[labels]

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("BALANCED GCN + GRU TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X_train, y_train, X_val, y_val = load_data()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = EEGSequenceDataset(
        X_train,
        y_train
    )

    val_dataset = EEGSequenceDataset(
        X_val,
        y_val
    )

    # --------------------------------------------------------
    # BALANCED SAMPLER
    # --------------------------------------------------------

    sampler = create_balanced_sampler(
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # CREATE EEG GRAPH
    # --------------------------------------------------------

    edge_index, adjacency = create_edge_index(
        device
    )

    print()
    print("EEG GRAPH")
    print("-" * 70)
    print("Number of nodes :", adjacency.shape[0])
    print("Adjacency shape :", adjacency.shape)
    print("Edge index shape:", edge_index.shape)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = EEGGCNGRU(
        input_features=10,
        gcn_hidden=64,
        gcn_embedding=32,
        gru_hidden=64,
        gru_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3
    ).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("MODEL")
    print("-" * 70)
    print("Parameters:", total_parameters)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------
    #
    # We are already using WeightedRandomSampler.
    # Therefore we use normal CrossEntropyLoss here
    # instead of applying class weights again.
    #
    # This avoids excessive class balancing.
    # --------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # --------------------------------------------------------
    # BEST MODEL TRACKING
    # --------------------------------------------------------

    best_macro_f1 = -1.0
    best_balanced_accuracy = -1.0

    patience_counter = 0

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_loss = 0.0

        train_predictions = []
        train_targets = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                X_batch,
                edge_index
            )

            # Loss
            loss = criterion(
                outputs,
                y_batch
            )

            # Backpropagation
            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Update model
            optimizer.step()

            # Accumulate loss
            train_loss += (
                loss.item()
                * X_batch.size(0)
            )

            # Predictions
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            train_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            train_targets.extend(
                y_batch.detach()
                .cpu()
                .numpy()
            )

        # Average training loss
        train_loss /= len(train_dataset)

        # Training metrics
        train_accuracy = accuracy_score(
            train_targets,
            train_predictions
        )

        train_balanced_accuracy = (
            balanced_accuracy_score(
                train_targets,
                train_predictions
            )
        )

        train_macro_f1 = f1_score(
            train_targets,
            train_predictions,
            average="macr✓,
            zero_division=0
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0

        val_predictions = []
        val_targets = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(
                    X_batch,
                    edge_index
                )

                loss = criterion(
                    outputs,
                    y_batch
                )

                val_loss += (
                    loss.item()
                    * X_batch.size(0)
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_predictions.extend(
                    predictions
                    .cpu()
                    .numpy()
                )

                val_targets.extend(
                    y_batch
                    .cpu()
                    .numpy()
                )

        # Average validation loss
        val_loss /= len(val_dataset)

        # Validation metrics
        val_accuracy = accuracy_score(
            val_targets,
            val_predictions
        )

        val_balanced_accuracy = (
            balanced_accuracy_score(
                val_targets,
                val_predictions
            )
        )

        val_macro_f1 = f1_score(
            val_targets,
            val_predictions,
            average="macr✓,
            zero_division=0
        )

        # ====================================================
        # LEARNING RATE SCHEDULER
        # ====================================================

        scheduler.step(
            val_macro_f1
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {train_accuracy:.4f} | "
            f"Bal Acc: {train_balanced_accuracy:.4f} | "
            f"Macro F1: {train_macro_f1:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Acc: {val_accuracy:.4f} | "
            f"Bal Acc: {val_balanced_accuracy:.4f} | "
            f"Macro F1: {val_macro_f1:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        is_better = (
            val_macro_f1 > best_macro_f1
            or (
                val_macro_f1 == best_macro_f1
                and
                val_balanced_accuracy
                > best_balanced_accuracy
            )
        )

        if is_better:

            best_macro_f1 = val_macro_f1

            best_balanced_accuracy = (
                val_balanced_accuracy
            )

            patience_counter = 0

            MODEL_PATH.mkdir(
                parents=True,
                exist_ok=True
            )

            checkpoint = {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_macro_f1":
                    val_macro_f1,

                "val_balanced_accuracy":
                    val_balanced_accuracy,

                "val_accuracy":
                    val_accuracy,

                "config": {
                    "input_features": 10,
                    "gcn_hidden": 64,
                    "gcn_embedding": 32,
                    "gru_hidden": 64,
                    "gru_layers": 2,
                    "num_classes": 2,
                    "dropout": 0.3
                }
            }

            torch.save(
                checkpoint,
                MODEL_OUTPUT
            )

            print(
                "✓ BEST MODEL SAVED"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if patience_counter >= PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best Validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Best Validation Balanced Accuracy: "
        f"{best_balanced_accuracy:.4f}"
    )

    print()
    print("Saved model:")

    print(
        MODEL_OUTPUT
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()