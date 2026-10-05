from pathlib import Path

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

MODEL_PATH = MODEL_DIR / "svm_arousal_best.joblib"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):

    data = np.load(path)

    features = data["features"]

    # IMPORTANT:
    # The NPZ files already contain binary arousal labels.
    arousal = data["arousal"]

    return features, arousal


# ============================================================
# USE EXISTING BINARY LABELS
# ============================================================

def create_binary_labels(arousal):

    """
    The processed DEAP NPZ files already contain
    binary arousal labels.

    0 = Low Arousal
    1 = High Arousal

    DO NOT apply the 5.0 threshold again.
    """

    labels = arousal.astype(np.int64)

    return labels


# ============================================================
# PRINT LABEL DISTRIBUTION
# ============================================================

def print_distribution(name, labels):

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    print(f"\n{name} distribution:")
    print("-" * 50)

    for label, count in zip(unique, counts):

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
# PREPARE FEATURES FOR SVM
# ============================================================

def prepare_features(features):

    """
    Original feature shape:

        (N, 30, 32, 10)

    N  = EEG sequences/trials
    30 = temporal windows
    32 = EEG electrodes
    10 = DE + PSD features

    For the SVM baseline:

        Step 1:
        Average the 30 temporal windows

        (N, 30, 32, 10)
                  
        (N, 32, 10)

        Step 2:
        Flatten electrode  feature dimensions

        (N, 32, 10)
                  
        (N, 320)
    """

    print(
        f"Original feature shape: "
        f"{features.shape}"
    )

    # Average temporal dimension
    features = features.mean(axis=1)

    print(
        f"After temporal averaging: "
        f"{features.shape}"
    )

    # Flatten
    features = features.reshape(
        features.shape[0],
        -1
    )

    print(
        f"Final SVM feature shape: "
        f"{features.shape}"
    )

    return features


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, X, y, name):

    predictions = model.predict(X)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macr,
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        y,
        predictions,
        average="macr,
        zero_division=0
    )

    macro_recall = recall_score(
        y,
        predictions,
        average="macr,
        zero_division=0
    )

    confusion = confusion_matrix(
        y,
        predictions
    )

    # --------------------------------------------------------
    # Results
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

    print(confusion)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
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
        "confusion_matrix": confusion,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DEAP AROUSAL - SVM BASELINE")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading datasets...")

    train_features, train_arousal = load_dataset(
        TRAIN_FILE
    )

    val_features, val_arousal = load_dataset(
        VAL_FILE
    )

    test_features, test_arousal = load_dataset(
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
    # CREATE BINARY LABELS
    # ========================================================

    train_labels = create_binary_labels(
        train_arousal
    )

    val_labels = create_binary_labels(
        val_arousal
    )

    test_labels = create_binary_labels(
        test_arousal
    )

    # ========================================================
    # SAFETY CHECK
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

    # Make sure training has both classes
    if len(np.unique(train_labels)) < 2:

        raise ValueError(
            "Training data contains only one class. "
            "Expected both Low Arousal (0) and "
            "High Arousal (1)."
        )

    print(
        "\n Training data contains both "
        "arousal classes."
    )

    # ========================================================
    # DISTRIBUTION
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
    print("PREPARING FEATURES FOR SVM")
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
    # CREATE SVM
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING SVM MODEL")
    print("=" * 70)

    svm_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Kernel          : RBF")
    print("C               : 1.0")
    print("Gamma           : scale")
    print("Class weighting : balanced")

    # ========================================================
    # TRAIN
    # ========================================================

    print("\nTraining SVM...")
    print("-" * 70)

    svm_model.fit(
        X_train,
        train_labels
    )

    print(" Training completed.")

    # ========================================================
    # VALIDATION
    # ========================================================

    val_results = evaluate(
        svm_model,
        X_val,
        val_labels,
        "VALIDATION"
    )

    # ========================================================
    # TEST
    # ========================================================

    test_results = evaluate(
        svm_model,
        X_test,
        test_labels,
        "TEST"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        svm_model,
        MODEL_PATH
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Model saved at:\n{MODEL_PATH}"
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

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()