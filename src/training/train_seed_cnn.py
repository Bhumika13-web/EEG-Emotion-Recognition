import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TRAIN_FILE = DATA_DIR / "seed_train.npz"
VAL_FILE = DATA_DIR / "seed_val.npz"
TEST_FILE = DATA_DIR / "seed_test.npz"

MODEL_FILE = MODEL_DIR / "cnn_seed_best.pth"

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING SEED DATA")
    print("=" * 70)

    train = np.load(TRAIN_FILE)
    val = np.load(VAL_FILE)
    test = np.load(TEST_FILE)

    X_train = train["features"]
    y_train = train["labels"]

    X_val = val["features"]
    y_val = val["labels"]

    X_test = test["features"]
    y_test = test["labels"]

    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# PYTORCH DATA LOADERS
# ============================================================

def create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    # CNN expects:
    #
    # (batch, channels, height, width)
    #
    # We have:
    #
    # (samples, 5, 62)
    #
    # Add one channel dimension:
    #
    # (samples, 1, 5, 62)

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    ).unsqueeze(1)

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    ).unsqueeze(1)

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    ).unsqueeze(1)

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# CNN MODEL
# ============================================================

class SEEDCNN(nn.Module):

    def __init__(self):

        super().__init__()

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
                3
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    split_name
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(X)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macr✓
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted"
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\n" + "=" * 70)
    print(f"SEED CNN - {split_name}")
    print("=" * 70)

    print(
        f"Accuracy          : "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{weighted_f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "Negative",
                "Neutral",
                "Positive"
            ],
            digits=4
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader
):

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    # SEED is already reasonably balanced, but calculating
    # weights from the training data keeps the code robust.

    all_train_labels = []

    for _, y in train_loader:

        all_train_labels.extend(
            y.numpy()
        )

    all_train_labels = np.array(
        all_train_labels
    )

    class_counts = np.bincount(
        all_train_labels,
        minlength=3
    )

    class_weights = (
        len(all_train_labels)
        /
        (
            3 * class_counts
        )
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE
    )

    print("\nClass counts:")
    print(class_counts)

    print("\nClass weights:")
    print(class_weights)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_macro_f1 = -1.0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("TRAINING SEED CNN")
    print("=" * 70)

    print("Device:", DEVICE)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for X, y in train_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X)

            loss = criterion(
                outputs,
                y
            )

            loss.backward()

            optimizer.step()

            batch_size = X.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

        train_loss = (
            running_loss
            /
            total_samples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_results = evaluate(
            model,
            val_loader,
            f"VALIDATION - Epoch {epoch}"
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{val_results['macro_f1']:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            val_results["macro_f1"]
            >
            best_macro_f1
        ):

            best_macro_f1 = (
                val_results["macro_f1"]
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_macro_f1":
                        best_macro_f1
                },
                MODEL_FILE
            )

            print(
                f"✓ Best model saved "
                f"(Macro F1: "
                f"{best_macro_f1:.4f})"
            )

    return best_macro_f1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEED CNN BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SEEDCNN().to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", total_parameters)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_model(
        model,
        train_loader,
        val_loader
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

    print(
        "Best validation Macro F1:",
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    val_results = evaluate(
        model,
        val_loader,
        "FINAL VALIDATION"
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    test_results = evaluate(
        model,
        test_loader,
        "FINAL TEST"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SEED CNN SUMMARY")
    print("=" * 70)

    print(
        f"Validation Accuracy : "
        f"{val_results['accuracy']:.4f}"
    )

    print(
        f"Validation Balanced Accuracy : "
        f"{val_results['balanced_accuracy']:.4f}"
    )

    print(
        f"Validation Macro F1 : "
        f"{val_results['macro_f1']:.4f}"
    )

    print()

    print(
        f"Test Accuracy : "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Test Balanced Accuracy : "
        f"{test_results['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Macro F1 : "
        f"{test_results['macro_f1']:.4f}"
    )

    print(
        f"Test Weighted F1 : "
        f"{test_results['weighted_f1']:.4f}"
    )

    print("\nModel saved at:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()