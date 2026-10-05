import sys
from pathlib import Path

import numpy as np

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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models"

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = DATA_PATH / "deap_train.npz"
VAL_FILE = DATA_PATH / "deap_val.npz"
TEST_FILE = DATA_PATH / "deap_test.npz"

MODEL_OUTPUT = MODEL_PATH / "svm_valence_best.joblib"


# ============================================================
# FEATURE AGGREGATION
# ============================================================

def aggregate_features(features):
    """
    Convert:

        (N, 30, 32, 10)

    into:

        (N, 320)

    by averaging across the 30 temporal windows.

    Each sample therefore contains:

        32 EEG electrodes × 10 DE/PSD features
    """

    if features.ndim != 4:
        raise ValueError(
            f"Expected 4D features, got {features.shape}"
        )

    if features.shape[2] != 32:
        raise ValueError(
            f"Expected 32 EEG channels, got "
            f"{features.shape[2]}"
        )

    if features.shape[3] != 10:
        raise ValueError(
            f"Expected 10 features, got "
            f"{features.shape[3]}"
        )

    # Average temporal windows
    aggregated = np.mean(
        features,
        axis=1
    )

    # Shape:
    # (N, 32, 10)

    # Flatten:
    # (N, 320)

    aggregated = aggregated.reshape(
        aggregated.shape[0],
        -1
    )

    return aggregated


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    if not VAL_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n{VAL_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found:\n{TEST_FILE}"
        )

    train = np.load(TRAIN_FILE)
    val = np.load(VAL_FILE)
    test = np.load(TEST_FILE)

    X_train = aggregate_features(
        train["features"]
    )

    y_train = train["valence"]

    X_val = aggregate_features(
        val["features"]
    )

    y_val = val["valence"]

    X_test = aggregate_features(
        test["features"]
    )

    y_test = test["valence"]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# PRINT METRICS
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    name
):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions
        )
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    precision = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1           : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{precision:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{recall:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Low Valence",
                "High Valence"
            ],
            digits=4,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SVM BASELINE — DEAP VALENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_dataset()

    print()
    print("DATA")
    print("-" * 70)

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
    print("Train classes:")
    print(
        "Low:",
        np.sum(y_train == 0)
    )
    print(
        "High:",
        np.sum(y_train == 1)
    )

    print()
    print("Validation classes:")
    print(
        "Low:",
        np.sum(y_val == 0)
    )
    print(
        "High:",
        np.sum(y_val == 1)
    )

    print()
    print("Test classes:")
    print(
        "Low:",
        np.sum(y_test == 0)
    )
    print(
        "High:",
        np.sum(y_test == 1)
    )

    # ========================================================
    # SVM MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING SVM")
    print("=" * 70)

    # StandardScaler:
    # normalizes feature magnitudes.
    #
    # SVC:
    # RBF kernel captures nonlinear relationships.
    #
    # class_weight="balanced":
    # compensates for the imbalance in the training set.

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
                    random_state=42
                )
            )
        ]
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("Training SVM...")
    print("Please wait...")

    svm_model.fit(
        X_train,
        y_train
    )

    print("✓ SVM training completed.")

    # ========================================================
    # VALIDATION
    # ========================================================

    val_results = evaluate_model(
        svm_model,
        X_val,
        y_val,
        "SVM VALIDATION RESULTS"
    )

    # ========================================================
    # TEST
    # ========================================================

    test_results = evaluate_model(
        svm_model,
        X_test,
        y_test,
        "SVM TEST RESULTS"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        svm_model,
        MODEL_OUTPUT
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        MODEL_OUTPUT
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SVM SUMMARY")
    print("=" * 70)

    print(
        f"Validation Accuracy : "
        f"{val_results['accuracy']:.4f}"
    )

    print(
        f"Validation Macro F1 : "
        f"{val_results['macro_f1']:.4f}"
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
    print("SVM EXPERIMENT COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()