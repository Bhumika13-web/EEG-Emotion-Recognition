from pathlib import Path
import pickle
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "DEAP"
    / "deap-dataset"
    / "data_preprocessed_python"
)

THRESHOLDS = [4.5, 5.0, 5.5]

# Current subject-wise split used in the project
TRAIN_SUBJECTS = [
    0, 3, 5, 6, 7, 9, 10, 11,
    12, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 31
]

VAL_SUBJECTS = [2, 4, 14, 30]

TEST_SUBJECTS = [1, 8, 13, 29]


# ============================================================
# LOAD DEAP LABELS
# ============================================================

def load_all_labels():

    all_labels = []
    subject_labels = {}

    print("=" * 70)
    print("LOADING DEAP LABELS")
    print("=" * 70)

    for subject_id in range(32):

        file_path = DEAP_DIR / f"s{subject_id + 1:02d}.dat"

        if not file_path.exists():
            print(f"WARNING: Missing {file_path}")
            continue

        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        labels = np.asarray(data["labels"])

        subject_labels[subject_id] = labels
        all_labels.append(labels)

        print(
            f"Subject {subject_id:02d} | "
            f"Trials: {len(labels)} | "
            f"Valence mean: {labels[:, 0].mean():.2f} | "
            f"Arousal mean: {labels[:, 1].mean():.2f}"
        )

    all_labels = np.concatenate(all_labels, axis=0)

    return all_labels, subject_labels


# ============================================================
# BASIC STATISTICS
# ============================================================

def print_basic_statistics(labels):

    print("\n" + "=" * 70)
    print("BASIC DEAP LABEL STATISTICS")
    print("=" * 70)

    names = [
        "Valence",
        "Arousal",
        "Dominance",
        "Liking"
    ]

    for i, name in enumerate(names):

        values = labels[:, i]

        print(
            f"{name:10s} | "
            f"Min: {values.min():.2f} | "
            f"Max: {values.max():.2f} | "
            f"Mean: {values.mean():.2f} | "
            f"Median: {np.median(values):.2f}"
        )


# ============================================================
# BINARY DISTRIBUTION
# ============================================================

def analyze_binary(labels):

    print("\n" + "=" * 70)
    print("BINARY VALENCE / AROUSAL DISTRIBUTION")
    print("=" * 70)

    for threshold in THRESHOLDS:

        print(f"\nThreshold = {threshold}")

        # Valence
        valence = labels[:, 0]
        low_v = np.sum(valence < threshold)
        high_v = np.sum(valence >= threshold)

        print(
            f"Valence -> "
            f"Low: {low_v} ({low_v / len(valence) * 100:.2f}%) | "
            f"High: {high_v} ({high_v / len(valence) * 100:.2f}%)"
        )

        # Arousal
        arousal = labels[:, 1]
        low_a = np.sum(arousal < threshold)
        high_a = np.sum(arousal >= threshold)

        print(
            f"Arousal -> "
            f"Low: {low_a} ({low_a / len(arousal) * 100:.2f}%) | "
            f"High: {high_a} ({high_a / len(arousal) * 100:.2f}%)"
        )


# ============================================================
# FOUR QUADRANT ANALYSIS
# ============================================================

def analyze_four_quadrants(labels, threshold=5.0):

    print("\n" + "=" * 70)
    print(f"FOUR-QUADRANT EMOTION DISTRIBUTION (Threshold = {threshold})")
    print("=" * 70)

    valence = labels[:, 0]
    arousal = labels[:, 1]

    low_v = valence < threshold
    high_v = valence >= threshold

    low_a = arousal < threshold
    high_a = arousal >= threshold

    classes = {
        "Low Valence + Low Arousal": low_v & low_a,
        "Low Valence + High Arousal": low_v & high_a,
        "High Valence + Low Arousal": high_v & low_a,
        "High Valence + High Arousal": high_v & high_a,
    }

    total = len(labels)

    for name, mask in classes.items():

        count = np.sum(mask)

        print(
            f"{name:30s} -> "
            f"{count:4d} "
            f"({count / total * 100:.2f}%)"
        )


# ============================================================
# SUBJECT-WISE ANALYSIS
# ============================================================

def analyze_subjects(subject_labels, threshold=5.0):

    print("\n" + "=" * 70)
    print(f"SUBJECT-WISE VALENCE DISTRIBUTION (Threshold = {threshold})")
    print("=" * 70)

    for subject_id in sorted(subject_labels):

        labels = subject_labels[subject_id]

        valence = labels[:, 0]

        low = np.sum(valence < threshold)
        high = np.sum(valence >= threshold)

        print(
            f"Subject {subject_id:02d} | "
            f"Low: {low:2d} | "
            f"High: {high:2d}"
        )


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

def analyze_split(name, subject_ids, subject_labels, threshold=5.0):

    selected = []

    for subject_id in subject_ids:

        if subject_id in subject_labels:
            selected.append(subject_labels[subject_id])

    labels = np.concatenate(selected)

    valence = labels[:, 0]
    arousal = labels[:, 1]

    low_v = np.sum(valence < threshold)
    high_v = np.sum(valence >= threshold)

    low_a = np.sum(arousal < threshold)
    high_a = np.sum(arousal >= threshold)

    print(f"\n{name}")
    print("-" * 50)

    print(
        f"Trials: {len(labels)}"
    )

    print(
        f"Valence -> "
        f"Low: {low_v} ({low_v / len(labels) * 100:.2f}%) | "
        f"High: {high_v} ({high_v / len(labels) * 100:.2f}%)"
    )

    print(
        f"Arousal -> "
        f"Low: {low_a} ({low_a / len(labels) * 100:.2f}%) | "
        f"High: {high_a} ({high_a / len(labels) * 100:.2f}%)"
    )


# ============================================================
# CHECK SUBJECT SPLIT
# ============================================================

def check_subject_split():

    print("\n" + "=" * 70)
    print("SUBJECT SPLIT CHECK")
    print("=" * 70)

    train = set(TRAIN_SUBJECTS)
    val = set(VAL_SUBJECTS)
    test = set(TEST_SUBJECTS)

    print("Train subjects:", sorted(train))
    print("Validation subjects:", sorted(val))
    print("Test subjects:", sorted(test))

    print("\nTrain ∩ Validation:", train & val)
    print("Train ∩ Test:", train & test)
    print("Validation ∩ Test:", val & test)

    all_subjects = train | val | test

    print(
        "\nTotal unique subjects:",
        len(all_subjects)
    )

    if len(all_subjects) == 32:
        print("✓ All 32 DEAP subjects are included.")

    if not (train & val or train & test or val & test):
        print("✓ No subject overlap detected.")

    print("\nSplit sizes:")
    print(f"Train: {len(train)} subjects")
    print(f"Validation: {len(val)} subjects")
    print(f"Test: {len(test)} subjects")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 70)
    print("# DEAP LABEL & EXPERIMENT PROTOCOL ANALYSIS")
    print("#" * 70)

    labels, subject_labels = load_all_labels()

    print_basic_statistics(labels)

    analyze_binary(labels)

    analyze_four_quadrants(
        labels,
        threshold=5.0
    )

    analyze_subjects(
        subject_labels,
        threshold=5.0
    )

    analyze_split(
        "TRAIN",
        TRAIN_SUBJECTS,
        subject_labels,
        threshold=5.0
    )

    analyze_split(
        "VALIDATION",
        VAL_SUBJECTS,
        subject_labels,
        threshold=5.0
    )

    analyze_split(
        "TEST",
        TEST_SUBJECTS,
        subject_labels,
        threshold=5.0
    )

    check_subject_split()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()