import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TRAIN_FILE = DATA_DIR / "seed_train.npz"
VAL_FILE = DATA_DIR / "seed_val.npz"
TEST_FILE = DATA_DIR / "seed_test.npz"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING PROCESSED SEED DATA")
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

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# FLATTEN FEATURES
# ============================================================

def flatten_features(X):

    # 5 frequency bands × 62 electrodes = 310 features

    return X.reshape(X.shape[0], -1)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, name):

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    balanced_acc = balanced_accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macr✓
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted"
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
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
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load
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
    # 2. Flatten
    # --------------------------------------------------------

    X_train = flatten_features(X_train)
    X_val = flatten_features(X_val)
    X_test = flatten_features(X_test)

    print("\nFlattened feature shape:")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    # --------------------------------------------------------
    # 3. SVM
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING SVM")
    print("=" * 70)

    model = Pipeline([
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
                class_weight="balanced"
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    print("SVM training completed.")

    # --------------------------------------------------------
    # 4. Validation
    # --------------------------------------------------------

    val_results = evaluate_model(
        model,
        X_val,
        y_val,
        "SEED SVM - VALIDATION"
    )

    # --------------------------------------------------------
    # 5. Test
    # --------------------------------------------------------

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "SEED SVM - TEST"
    )

    # --------------------------------------------------------
    # 6. Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODEL_DIR / "svm_seed.joblib"

    joblib.dump(
        model,
        model_path
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(model_path)

    # --------------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
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

    print(
        f"Test Accuracy       : "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Test Macro F1       : "
        f"{test_results['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()