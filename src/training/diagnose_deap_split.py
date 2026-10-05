from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_FILE = (
    PROCESSED_DIR
    / "deap_train.npz"
)

VAL_FILE = (
    PROCESSED_DIR
    / "deap_val.npz"
)

TEST_FILE = (
    PROCESSED_DIR
    / "deap_test.npz"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("DEAP SPLIT DIAGNOSTIC")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

train = np.load(TRAIN_FILE)
val = np.load(VAL_FILE)
test = np.load(TEST_FILE)


# ============================================================
# SHOW KEYS
# ============================================================

print()
print("=" * 70)
print("AVAILABLE DATA KEYS")
print("=" * 70)

print()
print("Train keys:", train.files)
print("Validation keys:", val.files)
print("Test keys:", test.files)


# ============================================================
# ANALYZE SPLIT
# ============================================================

def analyze_split(name, data):

    features = data["features"]
    labels = data["valence"]
    subjects = data["subject_ids"]
    trials = data["trial_ids"]

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print()
    print("Number of sequences:", len(labels))
    print("Features shape:", features.shape)

    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------

    unique_subjects = np.unique(subjects)

    print()
    print("Subjects:")
    print(unique_subjects)

    print(
        "Number of subjects:",
        len(unique_subjects)
    )

    # --------------------------------------------------------
    # TRIALS
    # --------------------------------------------------------

    unique_trials = np.unique(trials)

    print()
    print(
        "Unique trials:",
        len(unique_trials)
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    class_0 = int(
        np.sum(labels == 0)
    )

    class_1 = int(
        np.sum(labels == 1)
    )

    total = len(labels)

    print()
    print("Class distribution:")

    print(
        f"Class 0: {class_0} "
        f"({class_0 / total * 100:.2f}%)"
    )

    print(
        f"Class 1: {class_1} "
        f"({class_1 / total * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # MAJORITY BASELINE
    # --------------------------------------------------------

    majority_class = (
        0
        if class_0 >= class_1
        else 1
    )

    majority_predictions = np.full(
        len(labels),
        majority_class
    )

    majority_accuracy = accuracy_score(
        labels,
        majority_predictions
    )

    majority_balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            majority_predictions
        )
    )

    majority_macro_f1 = f1_score(
        labels,
        majority_predictions,
        average="macro",
        zero_division=0
    )

    majority_weighted_f1 = f1_score(
        labels,
        majority_predictions,
        average="weighted",
        zero_division=0
    )

    print()
    print("Majority-class baseline:")

    print(
        "Majority class:",
        majority_class
    )

    print(
        f"Accuracy: "
        f"{majority_accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{majority_balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1: "
        f"{majority_macro_f1:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{majority_weighted_f1:.4f}"
    )

    return {
        "features": features,
        "labels": labels,
        "subjects": subjects,
        "trials": trials,
        "class_0": class_0,
        "class_1": class_1,
    }


# ============================================================
# ANALYZE ALL SPLITS
# ============================================================

train_info = analyze_split(
    "TRAINING SET",
    train
)

val_info = analyze_split(
    "VALIDATION SET",
    val
)

test_info = analyze_split(
    "TEST SET",
    test
)


# ============================================================
# SUBJECT OVERLAP
# ============================================================

print()
print("=" * 70)
print("SUBJECT OVERLAP CHECK")
print("=" * 70)

train_subjects = set(
    train_info["subjects"]
)

val_subjects = set(
    val_info["subjects"]
)

test_subjects = set(
    test_info["subjects"]
)

train_val_overlap = (
    train_subjects
    & val_subjects
)

train_test_overlap = (
    train_subjects
    & test_subjects
)

val_test_overlap = (
    val_subjects
    & test_subjects
)

print()

print(
    "Train subjects:",
    sorted(train_subjects)
)

print(
    "Validation subjects:",
    sorted(val_subjects)
)

print(
    "Test subjects:",
    sorted(test_subjects)
)

print()

print(
    "Train ∩ Validation:",
    sorted(train_val_overlap)
)

print(
    "Train ∩ Test:",
    sorted(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    sorted(val_test_overlap)
)


if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):

    print()
    print(
        "✓ NO SUBJECT OVERLAP DETECTED"
    )

else:

    print()
    print(
        "⚠ SUBJECT OVERLAP DETECTED"
    )


# ============================================================
# SUBJECT-WISE CLASS DISTRIBUTION
# ============================================================

def print_subject_distribution(
    name,
    info
):

    print()
    print("=" * 70)
    print(
        f"{name} SUBJECT-WISE "
        "VALENCE DISTRIBUTION"
    )
    print("=" * 70)

    subjects = info["subjects"]
    labels = info["labels"]

    for subject in sorted(
        np.unique(subjects)
    ):

        mask = (
            subjects == subject
        )

        subject_labels = labels[mask]

        low = int(
            np.sum(
                subject_labels == 0
            )
        )

        high = int(
            np.sum(
                subject_labels == 1
            )
        )

        total = len(
            subject_labels
        )

        print(
            f"Subject {subject:02d}: "
            f"Total={total:3d}, "
            f"Low={low:3d}, "
            f"High={high:3d}"
        )


print_subject_distribution(
    "TRAIN",
    train_info
)

print_subject_distribution(
    "VALIDATION",
    val_info
)

print_subject_distribution(
    "TEST",
    test_info
)


# ============================================================
# CHECK SEQUENCES PER SUBJECT
# ============================================================

print()
print("=" * 70)
print("SEQUENCES PER SUBJECT")
print("=" * 70)


for name, info in [
    ("TRAIN", train_info),
    ("VALIDATION", val_info),
    ("TEST", test_info),
]:

    subjects = info["subjects"]

    unique, counts = np.unique(
        subjects,
        return_counts=True
    )

    print()
    print(name)

    for subject, count in zip(
        unique,
        counts
    ):

        print(
            f"  Subject {subject:02d}: "
            f"{count} sequences"
        )


# ============================================================
# CURRENT MODEL RESULTS
# ============================================================

print()
print("=" * 70)
print("CURRENT GCN + GRU TEST RESULTS")
print("=" * 70)


# Based on the actual confusion matrix
# from the completed test evaluation.

current_labels = np.array(
    [0] * 127
    + [1] * 33
)

current_predictions = np.array(
    [0] * 106
    + [1] * 21
    + [0] * 26
    + [1] * 7
)


accuracy = accuracy_score(
    current_labels,
    current_predictions
)

balanced_accuracy = (
    balanced_accuracy_score(
        current_labels,
        current_predictions
    )
)

macro_f1 = f1_score(
    current_labels,
    current_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    current_labels,
    current_predictions,
    average="weighted",
    zero_division=0
)


print()

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1: "
    f"{weighted_f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    current_labels,
    current_predictions,
    labels=[0, 1]
)

print()
print("Confusion Matrix:")

print()

print(
    "             Pred Low   Pred High"
)

print(
    f"Actual Low      "
    f"{cm[0, 0]:3d}         "
    f"{cm[0, 1]:3d}"
)

print(
    f"Actual High     "
    f"{cm[1, 0]:3d}         "
    f"{cm[1, 1]:3d}"
)


# ============================================================
# BASELINE COMPARISON
# ============================================================

test_labels = test_info["labels"]

majority_class = (
    0
    if test_info["class_0"]
    >= test_info["class_1"]
    else 1
)

baseline_predictions = np.full(
    len(test_labels),
    majority_class
)

baseline_accuracy = accuracy_score(
    test_labels,
    baseline_predictions
)

baseline_balanced_accuracy = (
    balanced_accuracy_score(
        test_labels,
        baseline_predictions
    )
)

baseline_macro_f1 = f1_score(
    test_labels,
    baseline_predictions,
    average="macro",
    zero_division=0
)


print()
print("=" * 70)
print("MODEL VS MAJORITY BASELINE")
print("=" * 70)

print()

print(
    f"{'Metric':<25}"
    f"{'Baseline':>15}"
    f"{'GCN + GRU':>15}"
)

print("-" * 55)

print(
    f"{'Accuracy':<25}"
    f"{baseline_accuracy:>15.4f}"
    f"{accuracy:>15.4f}"
)

print(
    f"{'Balanced Accuracy':<25}"
    f"{baseline_balanced_accuracy:>15.4f}"
    f"{balanced_accuracy:>15.4f}"
)

print(
    f"{'Macro F1':<25}"
    f"{baseline_macro_f1:>15.4f}"
    f"{macro_f1:>15.4f}"
)


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

print()
print("=" * 70)
print("FINAL DIAGNOSIS")
print("=" * 70)

print()

if len(train_val_overlap) == 0:

    print(
        "✓ Train and validation subjects "
        "are completely separated."
    )

else:

    print(
        "⚠ Train and validation subjects "
        "overlap."
    )


if len(train_test_overlap) == 0:

    print(
        "✓ Train and test subjects "
        "are completely separated."
    )

else:

    print(
        "⚠ Train and test subjects "
        "overlap."
    )


if len(val_test_overlap) == 0:

    print(
        "✓ Validation and test subjects "
        "are completely separated."
    )

else:

    print(
        "⚠ Validation and test subjects "
        "overlap."
    )


print()

print(
    "The processed dataset preserves "
    "subject_ids and trial_ids."
)

print(
    "Therefore the split can now be "
    "verified directly."
)


print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)