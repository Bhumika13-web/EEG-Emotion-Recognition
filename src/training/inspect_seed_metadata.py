import numpy as np
from pathlib import Path

SEED_DIR = Path("data/SEED")

DATA_FILE = SEED_DIR / "DatasetCaricatoNoImage.npz"
LABEL_FILE = SEED_DIR / "LabelsNoImage.npz"
SUBJECT_FILE = SEED_DIR / "SubjectsNoImage.npz"


def inspect_npz(name, path):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("File:", path)

    data = np.load(path, allow_pickle=True)

    print("NPZ keys:", data.files)

    for key in data.files:
        arr = data[key]

        print(f"\nKey: {key}")
        print("  Type:", type(arr))
        print("  Shape:", getattr(arr, "shape", "N/A"))
        print("  Dtype:", getattr(arr, "dtype", "N/A"))

        if arr.dtype == object:
            print("  Object array detected")

            try:
                print("  First object:")
                print(arr.flat[0])
            except Exception as e:
                print("  Could not inspect object:", e)


def main():

    # ---------------------------------------------------------
    # 1. Inspect all NPZ files
    # ---------------------------------------------------------

    inspect_npz("SEED DATA", DATA_FILE)
    inspect_npz("SEED LABELS", LABEL_FILE)
    inspect_npz("SEED SUBJECTS", SUBJECT_FILE)

    # ---------------------------------------------------------
    # 2. Load actual arrays
    # ---------------------------------------------------------

    data_npz = np.load(DATA_FILE, allow_pickle=True)
    label_npz = np.load(LABEL_FILE, allow_pickle=True)
    subject_npz = np.load(SUBJECT_FILE, allow_pickle=True)

    data = data_npz[data_npz.files[0]]
    labels = label_npz[label_npz.files[0]]
    subjects = subject_npz[subject_npz.files[0]]

    print("\n" + "=" * 70)
    print("BASIC DATA CHECK")
    print("=" * 70)

    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    print("Subjects shape:", subjects.shape)

    # ---------------------------------------------------------
    # 3. Check first sample
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST SAMPLE")
    print("=" * 70)

    print("Sample shape:", data[0].shape)
    print("Subject:", subjects[0])
    print("Label:", labels[0])

    print("\nFirst sample:")
    print(data[0])

    # ---------------------------------------------------------
    # 4. Check dimensions
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DIMENSION ANALYSIS")
    print("=" * 70)

    if data.ndim == 3:
        samples, dim1, dim2 = data.shape

        print("Samples:", samples)
        print("Dimension 1:", dim1)
        print("Dimension 2:", dim2)

        print("\nIMPORTANT:")
        print("The current file confirms a 3D representation:")
        print(f"({samples}, {dim1}, {dim2})")

        print("The semantic meaning of the two inner dimensions")
        print("must be verified from the dataset source/metadata.")

    # ---------------------------------------------------------
    # 5. Check whether samples are very similar
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONSECUTIVE SAMPLE DIFFERENCE")
    print("=" * 70)

    for i in range(5):
        diff = np.mean(np.abs(data[i + 1] - data[i]))
        print(f"Sample {i} -> {i+1}: mean absolute difference = {diff:.6f}")

    # ---------------------------------------------------------
    # 6. Subject boundaries
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUBJECT BOUNDARIES")
    print("=" * 70)

    unique_subjects = np.unique(subjects)

    for subject in unique_subjects:
        indices = np.where(subjects == subject)[0]

        print(
            f"Subject {subject}: "
            f"{len(indices)} samples | "
            f"index {indices[0]} -> {indices[-1]}"
        )

    # ---------------------------------------------------------
    # 7. Labels
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        percentage = count / len(labels) * 100
        print(
            f"Label {label}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

    # ---------------------------------------------------------
    # 8. Subject  Label
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUBJECT  LABEL")
    print("=" * 70)

    for subject in unique_subjects:
        subject_mask = subjects == subject

        print(f"\nSubject {subject}:")

        sub_labels, sub_counts = np.unique(
            labels[subject_mask],
            return_counts=True
        )

        for label, count in zip(sub_labels, sub_counts):
            print(f"  Label {label}: {count}")

    # ---------------------------------------------------------
    # 9. NaN / Inf
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA QUALITY")
    print("=" * 70)

    print("NaN:", np.isnan(data).sum())
    print("Inf:", np.isinf(data).sum())

    print("\nInspection completed.")


if __name__ == "__main__":
    main()