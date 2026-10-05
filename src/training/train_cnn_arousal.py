from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_DIR / "deap_train.npz"
VAL_FILE = DATA_DIR / "deap_val.npz"
TEST_FILE = DATA_DIR / "deap_test.npz"

MODEL_PATH = MODEL_DIR / "cnn_arousal_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 2

# ============================================================
# CNN MODEL
# ============================================================

class EEGCNN(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # Input:
        # (batch, 1, 32, 10)

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

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path):

    data = np.load(path)

    features = data["features"]

    # IMPORTANT:
    # Arousal labels are already binary:
    #
    # 0 = Low Arousal
    # 1 = High Arousal

    labels = data["arousal"].astype(
        np.int64
    )

    return features, labels


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(features):

    """
    Original:

        (N, 30, 32, 10)

    The CNN baseline does NOT explicitly model
    the temporal sequence.

    We average the 30 temporal windows:

        (N, 30, 32, 10)
              ↓
        (N, 32, 10)

    Then add a CNN channel dimension:

        (N, 32, 10)
              ↓
        (N, 1, 32, 10)
    """

    print(
        f"Original feature shape: {features.shape}"
    )

    # Temporal averaging
    features = features.mean(
        axis=1
    )

    print(
        f"After temporal averaging: "
        f"{features.shape}"
    )

    # Add CNN channel dimension
    features = np.expand_dims(
        features,
        axis=1
    )

    print(
        f"CNN input shape: "
        f"{features.shape}"
    )

    return features.astype(
        np.float32
    )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def print_distribution(name, labels):

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    print(f"\n{name} distribution:")
    print("-" * 50)

    for label, count in zip(
        unique,
        counts
    ):

        if label == 0:
            class_name = "Low Arousal"

        elif label == 1:
            class_name = "High Arousal"

        else:
            class_name = f"Unknown Class {label}"

        percentage = (
            count / len(labels) * 100
        )

        print(
            f"  {class_name}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    name
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

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"{name} RESULTS")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{macro_recall:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "Low Arousal",
                "High Arousal"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm,
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for X, y in loader:

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

        running_loss += (
            loss.item() * X.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            y.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DEAP AROUSAL - CNN BASELINE")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading datasets...")

    train_features, train_labels = load_dataset(
        TRAIN_FILE
    )

    val_features, val_labels = load_dataset(
        VAL_FILE
    )

    test_features, test_labels = load_dataset(
        TEST_FILE
    )

    print("\nRaw dataset shapes:")

    print(
        f"Train:      {train_features.shape}"
    )

    print(
        f"Validation: {val_features.shape}"
    )

    print(
        f"Test:       {test_features.shape}"
    )

    # ========================================================
    # LABEL CHECK
    # ========================================================

    print("\nChecking labels...")

    print(
        "Train unique labels:",
        np.unique(train_labels)
    )

    print(
        "Validation unique labels:",
        np.unique(val_labels)
    )

    print(
        "Test unique labels:",
        np.unique(test_labels)
    )

    if len(np.unique(train_labels)) < 2:

        raise ValueError(
            "Training data contains only one class. "
            "Expected Low Arousal (0) and "
            "High Arousal (1)."
        )

    print(
        "\n✓ Training data contains both "
        "arousal classes."
    )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    print_distribution(
        "Train",
        train_labels
    )

    print_distribution(
        "Validation",
        val_labels
    )

    print_distribution(
        "Test",
        test_labels
    )

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    print("\n" + "=" * 70)
    print("PREPARING CNN FEATURES")
    print("=" * 70)

    X_train = prepare_features(
        train_features
    )

    X_val = prepare_features(
        val_features
    )

    X_test = prepare_features(
        test_features
    )

    # ========================================================
    # CONVERT TO TORCH
    # ========================================================

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        train_labels,
        dtype=torch.long
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val = torch.tensor(
        val_labels,
        dtype=torch.long
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        test_labels,
        dtype=torch.long
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

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

    # ========================================================
    # MODEL
    # ========================================================

    model = EEGCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print("\n" + "=" * 70)
    print("CNN MODEL")
    print("=" * 70)

    print(model)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_counts = np.bincount(
        train_labels,
        minlength=NUM_CLASSES
    )

    class_weights = (
        len(train_labels) /
        (
            NUM_CLASSES *
            class_counts
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

    # ========================================================
    # LOSS + OPTIMIZER
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # TRAIN
    # ========================================================

    best_val_macro_f1 = -1.0
    best_epoch = 0

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_acc, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        # Validation
        model.eval()

        val_predictions = []
        val_targets = []

        with torch.no_grad():

            for X, y in val_loader:

                X = X.to(DEVICE)

                outputs = model(X)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_predictions.extend(
                    predictions.cpu().numpy()
                )

                val_targets.extend(
                    y.numpy()
                )

        val_acc = accuracy_score(
            val_targets,
            val_predictions
        )

        val_balanced_acc = balanced_accuracy_score(
            val_targets,
            val_predictions
        )

        val_macro_f1 = f1_score(
            val_targets,
            val_predictions,
            average="macro",
            zero_division=0
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Bal Acc: {val_balanced_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_macro_f1 > best_val_macro_f1:

            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch

            MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_macro_f1,
                    "val_accuracy": val_acc,
                    "val_balanced_accuracy": val_balanced_acc,
                    "num_classes": NUM_CLASSES,
                },
                MODEL_PATH
            )

            print(
                f"  ✓ Best model saved "
                f"(epoch {epoch})"
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation Macro F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    evaluate(
        model,
        val_loader,
        "VALIDATION"
    )

    # ========================================================
    # TEST RESULTS
    # ========================================================

    test_results = evaluate(
        model,
        test_loader,
        "TEST"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    print(
        f"Accuracy          : "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{test_results['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{test_results['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{test_results['weighted_f1']:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{test_results['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{test_results['macro_recall']:.4f}"
    )

    print("\nBest epoch:", best_epoch)

    print(
        "\nModel saved at:"
    )

    print(MODEL_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()