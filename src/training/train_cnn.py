import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 16
EPOCHS = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 5

NUM_CLASSES = 2

MODEL_OUTPUT = (
    MODEL_PATH / "cnn_valence_best.pth"
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
# CNN MODEL
# ============================================================

class EEGCNN(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # Input:
        # (batch, 30, 32, 10)

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # (batch, 30, 32, 10)

        # Average temporal windows.
        #
        # This creates a simple CNN baseline
        # without giving the CNN a GRU-style
        # temporal model.

        x = x.mean(
            dim=1
        )

        # (batch, 32, 10)

        # Add channel dimension
        x = x.unsqueeze(1)

        # (batch, 1, 32, 10)

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train_file = DATA_PATH / "deap_train.npz"
    val_file = DATA_PATH / "deap_val.npz"
    test_file = DATA_PATH / "deap_test.npz"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{train_file}"
        )

    if not val_file.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n{val_file}"
        )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found:\n{test_file}"
        )

    train = np.load(train_file)
    val = np.load(val_file)
    test = np.load(test_file)

    X_train = train["features"]
    y_train = train["valence"]

    X_val = val["features"]
    y_val = val["valence"]

    X_test = test["features"]
    y_test = test["valence"]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_loader(X, y, shuffle):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )

    return loader


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    predictions = []
    targets = []

    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch
            )

            total_loss += (
                loss.item()
                * X_batch.size(0)
            )

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                y_batch.cpu().numpy()
            )

    loss = (
        total_loss /
        len(loader.dataset)
    )

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

    return {
        "loss": loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
        "predictions": predictions,
        "targets": targets
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CNN BASELINE — DEAP VALENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    print()
    print("=" * 70)
    print("DATA")
    print("=" * 70)

    print(
        "Train:",
        X_train.shape,
        y_train.shape
    )

    print(
        "Validation:",
        X_val.shape,
        y_val.shape
    )

    print(
        "Test:",
        X_test.shape,
        y_test.shape
    )

    print()
    print("Train class distribution:")
    print(
        "Low:",
        np.sum(y_train == 0)
    )
    print(
        "High:",
        np.sum(y_train == 1)
    )

    print()
    print("Validation class distribution:")
    print(
        "Low:",
        np.sum(y_val == 0)
    )
    print(
        "High:",
        np.sum(y_val == 1)
    )

    print()
    print("Test class distribution:")
    print(
        "Low:",
        np.sum(y_test == 0)
    )
    print(
        "High:",
        np.sum(y_test == 1)
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = create_loader(
        X_train,
        y_train,
        shuffle=True
    )

    val_loader = create_loader(
        X_val,
        y_val,
        shuffle=False
    )

    test_loader = create_loader(
        X_test,
        y_test,
        shuffle=False
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = EEGCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        "Parameters:",
        total_parameters
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_macro_f1 = -1.0
    best_balanced_accuracy = -1.0

    patience_counter = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        train_predictions = []
        train_targets = []

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * X_batch.size(0)
            )

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            train_predictions.extend(
                predicted.detach()
                .cpu()
                .numpy()
            )

            train_targets.extend(
                y_batch.detach()
                .cpu()
                .numpy()
            )

        train_loss = (
            running_loss /
            len(train_loader.dataset)
        )

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
            average="macro",
            zero_division=0
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_results = evaluate(
            model,
            val_loader,
            device
        )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_results["macro_f1"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

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
            f"Val Loss:   {val_results['loss']:.4f} | "
            f"Acc: {val_results['accuracy']:.4f} | "
            f"Bal Acc: {val_results['balanced_accuracy']:.4f} | "
            f"Macro F1: {val_results['macro_f1']:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        # ----------------------------------------------------
        # SAVE BEST
        # ----------------------------------------------------

        is_better = (
            val_results["macro_f1"]
            > best_macro_f1
            or (
                val_results["macro_f1"]
                == best_macro_f1
                and
                val_results["balanced_accuracy"]
                > best_balanced_accuracy
            )
        )

        if is_better:

            best_macro_f1 = (
                val_results["macro_f1"]
            )

            best_balanced_accuracy = (
                val_results["balanced_accuracy"]
            )

            patience_counter = 0

            MODEL_PATH.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_accuracy":
                        val_results["accuracy"],
                    "val_balanced_accuracy":
                        val_results["balanced_accuracy"],
                    "val_macro_f1":
                        val_results["macro_f1"],
                    "config": {
                        "num_classes": 2,
                        "input_shape": [
                            30,
                            32,
                            10
                        ]
                    }
                },
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

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING BEST CNN")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_OUTPUT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

    print(
        "Best validation Macro F1:",
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test_results = evaluate(
        model,
        test_loader,
        device
    )

    print()
    print("=" * 70)
    print("CNN TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{test_results['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{test_results['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{test_results['weighted_f1']:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{test_results['precision']:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{test_results['recall']:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        test_results["targets"],
        test_results["predictions"]
    )

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
        f"{cm[0, 0]:4d}  {cm[0, 1]:4d}"
    )

    print(
        f"Actual High      "
        f"{cm[1, 0]:4d}  {cm[1, 1]:4d}"
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
            test_results["targets"],
            test_results["predictions"],
            target_names=[
                "Low Valence",
                "High Valence"
            ],
            digits=4,
            zero_division=0
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("=" * 70)
    print("CNN EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        "Model saved at:"
    )

    print(
        MODEL_OUTPUT
    )

    print()
    print(
        f"Test Accuracy       : "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Test Balanced Acc.  : "
        f"{test_results['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Macro F1       : "
        f"{test_results['macro_f1']:.4f}"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()