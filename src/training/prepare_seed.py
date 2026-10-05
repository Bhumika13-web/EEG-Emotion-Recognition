import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

SEED_DIR = Path("data/SEED")
OUTPUT_DIR = Path("data/processed")

DATA_FILE = SEED_DIR / "DatasetCaricatoNoImage.npz"
LABEL_FILE = SEED_DIR / "LabelsNoImage.npz"
SUBJECT_FILE = SEED_DIR / "SubjectsNoImage.npz"


# ============================================================
# LOAD DATA
# ============================================================

def load_seed():

    print("=" * 70)
    print("LOADING SEED DATASET")
    print("=" * 70)

    data_npz = np.load(DATA_FILE, allow_pickle=True)
    label_npz = np.load(LABEL_FILE, allow_pickle=True)
    subject_npz = np.load(SUBJECT_FILE, allow_pickle=True)

    data = data_npz[data_npz.files[0]]
    labels = label_npz[label_npz.files[0]]
    subjects = subject_npz[subject_npz.files[0]]

    print("Data shape     :", data.shape)
    print("Labels shape   :", labels.shape)
    print("Subjects shape :", subjects.shape)

    return data.astype(np.float32), labels.astype(np.int64), subjects.astype(np.int64)


# ============================================================
# DATA CHECK
# ============================================================

def check_data(data, labels, subjects):

    print("\n" + "=" * 70)
    print("DATA CHECK")
    print("=" * 70)

    assert data.shape[0] == labels.shape[0]
    assert data.shape[0] == subjects.shape[0]

    print("Number of samples :", len(data))
    print("Number of subjects:", len(np.unique(subjects)))
    print("Number of classes :", len(np.unique(labels)))
    print("Feature shape     :", data.shape[1:])

    print("\nNaN values :", np.isnan(data).sum())
    print("Inf values :", np.isinf(data).sum())

    print("\nLabels:")

    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        print(
            f"  Label {label}: "
            f"{count} samples "
            f"({count / len(labels) * 100:.2f}%)"
        )


# ============================================================
# SUBJECT-WISE SPLIT
# ============================================================

def split_by_subject(data, labels, subjects):

    print("\n" + "=" * 70)
    print("SUBJECT-WISE SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Train: subjects 0-9
    # Validation: subjects 10-11
    # Test: subjects 12-14
    # --------------------------------------------------------

    train_subjects = list(range(0, 10))
    val_subjects = list(range(10, 12))
    test_subjects = list(range(12, 15))

    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)

    X_train = data[train_mask]
    y_train = labels[train_mask]
    s_train = subjects[train_mask]

    X_val = data[val_mask]
    y_val = labels[val_mask]
    s_val = subjects[val_mask]

    X_test = data[test_mask]
    y_test = labels[test_mask]
    s_test = subjects[test_mask]

    print("Train subjects:", train_subjects)
    print("Val subjects  :", val_subjects)
    print("Test subjects :", test_subjects)

    print("\nSamples:")
    print("Train:", len(X_train))
    print("Val  :", len(X_val))
    print("Test :", len(X_test))

    return (
        X_train,
        y_train,
        s_train,
        X_val,
        y_val,
        s_val,
        X_test,
        y_test,
        s_test,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(X_train, X_val, X_test):

    print("\n" + "=" * 70)
    print("FEATURE NORMALIZATION")
    print("=" * 70)

    print("Original shape:", X_train.shape)

    # X shape:
    #
    # (samples, 5 frequency bands, 62 electrodes)
    #
    # We fit the scaler ONLY on training subjects.
    # This prevents information leakage from validation/test subjects.

    n_train, n_bands, n_channels = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    # Flatten each sample:
    #
    # 5  62 = 310 features
    #
    X_train_flat = X_train.reshape(n_train, -1)
    X_val_flat = X_val.reshape(n_val, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    scaler = StandardScaler()

    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)

    # Restore original 5  62 structure

    X_train = X_train_flat.reshape(
        n_train,
        n_bands,
        n_channels
    ).astype(np.float32)

    X_val = X_val_flat.reshape(
        n_val,
        n_bands,
        n_channels
    ).astype(np.float32)

    X_test = X_test_flat.reshape(
        n_test,
        n_bands,
        n_channels
    ).astype(np.float32)

    print("Normalized shape:", X_train.shape)

    print("\nTraining statistics after normalization:")
    print("Mean:", X_train.mean())
    print("Std :", X_train.std())

    return X_train, X_val, X_test


# ============================================================
# SAVE DATA
# ============================================================

def save_dataset(
    filename,
    X,
    y,
    subjects
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / filename

    np.savez_compressed(
        output_path,
        features=X,
        labels=y,
        subject_ids=subjects
    )

    print(f"Saved: {output_path}")
    print("  Features:", X.shape)
    print("  Labels  :", y.shape)
    print("  Subjects:", subjects.shape)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    data, labels, subjects = load_seed()

    # --------------------------------------------------------
    # 2. Check
    # --------------------------------------------------------

    check_data(
        data,
        labels,
        subjects
    )

    # --------------------------------------------------------
    # 3. Subject-wise split
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        s_train,
        X_val,
        y_val,
        s_val,
        X_test,
        y_test,
        s_test,
    ) = split_by_subject(
        data,
        labels,
        subjects
    )

    # --------------------------------------------------------
    # 4. Normalize
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test
    ) = normalize_features(
        X_train,
        X_val,
        X_test
    )

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)

    save_dataset(
        "seed_train.npz",
        X_train,
        y_train,
        s_train
    )

    save_dataset(
        "seed_val.npz",
        X_val,
        y_val,
        s_val
    )

    save_dataset(
        "seed_test.npz",
        X_test,
        y_test,
        s_test
    )

    # --------------------------------------------------------
    # 6. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEED PREPARATION COMPLETE")
    print("=" * 70)

    print("""
Train:
    Subjects : 0-9
    Samples  : 33,940

Validation:
    Subjects : 10-11
    Samples  : 6,788

Test:
    Subjects : 12-14
    Samples  : 10,182

Feature representation:
    5 frequency bands  62 EEG electrodes

Classes:
    0 = Negative
    1 = Neutral
    2 = Positive
""")


if __name__ == "__main__":
    main()