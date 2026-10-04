"""
SEED Dataset Analysis

This script analyzes the actual SEED dataset loaded from:

data/SEED/
    DatasetCaricatoNoImage.npz
    LabelsNoImage.npz
    SubjectsNoImage.npz

IMPORTANT:
We do NOT assume that the 5 x 62 representation is raw EEG.
The purpose of this script is to understand the dataset structure first.
"""

import os
import sys
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT LOADER
# ============================================================

from src.preprocessing.seed_loader import SEEDLoader


# ============================================================
# PRINT HELPERS
# ============================================================

def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def subsection(title):

    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


# ============================================================
# LOAD ACTUAL ARRAYS
# ============================================================

def load_seed_arrays():

    section("LOADING SEED DATASET")

    loader = SEEDLoader()

    print("\nSEEDLoader object attributes:")

    attributes = [
        attr
        for attr in dir(loader)
        if not attr.startswith("_")
    ]

    for attr in attributes:
        try:
            value = getattr(loader, attr)

            if callable(value):
                print(f"  {attr}()")

            else:
                print(
                    f"  {attr}: "
                    f"type={type(value).__name__}"
                )

        except Exception:
            pass

    print("\nAttempting to locate the actual NumPy arrays...")

    # --------------------------------------------------------
    # Try common attribute names
    # --------------------------------------------------------

    data = None
    labels = None
    subjects = None

    possible_data_names = [
        "X",
        "x",
        "data_array",
        "dataset",
        "features",
        "eeg_data",
        "samples"
    ]

    possible_label_names = [
        "y",
        "labels_array",
        "targets",
        "target",
        "emotion_labels"
    ]

    possible_subject_names = [
        "subject_ids",
        "subject_array",
        "subject_labels",
        "subjects_array"
    ]

    # --------------------------------------------------------
    # Search data
    # --------------------------------------------------------

    for name in possible_data_names:

        if hasattr(loader, name):

            value = getattr(loader, name)

            if isinstance(value, np.ndarray):

                data = value

                print(
                    f"\n✓ Data found in loader.{name}"
                )

                break

    # --------------------------------------------------------
    # Search labels
    # --------------------------------------------------------

    for name in possible_label_names:

        if hasattr(loader, name):

            value = getattr(loader, name)

            if isinstance(value, np.ndarray):

                labels = value

                print(
                    f"✓ Labels found in loader.{name}"
                )

                break

    # --------------------------------------------------------
    # Search subjects
    # --------------------------------------------------------

    for name in possible_subject_names:

        if hasattr(loader, name):

            value = getattr(loader, name)

            if isinstance(value, np.ndarray):

                subjects = value

                print(
                    f"✓ Subjects found in loader.{name}"
                )

                break

    # --------------------------------------------------------
    # Direct fallback:
    # Load the NPZ files directly.
    #
    # This is the reliable method for your dataset.
    # --------------------------------------------------------

    seed_dir = os.path.join(
        PROJECT_ROOT,
        "data",
        "SEED"
    )

    data_path = os.path.join(
        seed_dir,
        "DatasetCaricatoNoImage.npz"
    )

    labels_path = os.path.join(
        seed_dir,
        "LabelsNoImage.npz"
    )

    subjects_path = os.path.join(
        seed_dir,
        "SubjectsNoImage.npz"
    )

    if data is None:

        print(
            "\nData was not directly exposed by SEEDLoader."
        )

        print(
            "Loading DatasetCaricatoNoImage.npz directly..."
        )

        with np.load(
            data_path,
            allow_pickle=True
        ) as npz:

            print(
                f"NPZ keys: {npz.files}"
            )

            # Get first stored array
            data = npz[npz.files[0]]

        print(
            f"✓ Data loaded directly: "
            f"{data.shape}"
        )

    if labels is None:

        print(
            "\nLoading LabelsNoImage.npz directly..."
        )

        with np.load(
            labels_path,
            allow_pickle=True
        ) as npz:

            print(
                f"NPZ keys: {npz.files}"
            )

            labels = npz[npz.files[0]]

        print(
            f"✓ Labels loaded directly: "
            f"{labels.shape}"
        )

    if subjects is None:

        print(
            "\nLoading SubjectsNoImage.npz directly..."
        )

        with np.load(
            subjects_path,
            allow_pickle=True
        ) as npz:

            print(
                f"NPZ keys: {npz.files}"
            )

            subjects = npz[npz.files[0]]

        print(
            f"✓ Subjects loaded directly: "
            f"{subjects.shape}"
        )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    data = np.asarray(data)

    labels = np.asarray(labels)

    subjects = np.asarray(subjects)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print("\nFinal loaded shapes:")

    print(
        f"Data     : {data.shape}"
    )

    print(
        f"Labels   : {labels.shape}"
    )

    print(
        f"Subjects : {subjects.shape}"
    )

    if data.ndim == 0:

        raise ValueError(
            "Data is still a 0-dimensional object. "
            "Please check the NPZ file structure."
        )

    if labels.ndim == 0:

        raise ValueError(
            "Labels is still a 0-dimensional object."
        )

    if subjects.ndim == 0:

        raise ValueError(
            "Subjects is still a 0-dimensional object."
        )

    return data, labels, subjects


# ============================================================
# 1. BASIC STRUCTURE
# ============================================================

def analyze_basic_structure(
    data,
    labels,
    subjects
):

    section("1. BASIC DATASET STRUCTURE")

    print(
        f"Data shape      : {data.shape}"
    )

    print(
        f"Data dtype      : {data.dtype}"
    )

    print(
        f"Labels shape    : {labels.shape}"
    )

    print(
        f"Labels dtype    : {labels.dtype}"
    )

    print(
        f"Subjects shape  : {subjects.shape}"
    )

    print(
        f"Subjects dtype  : {subjects.dtype}"
    )

    print(
        f"\nNumber of samples : {len(data)}"
    )

    print(
        f"Number of labels  : {len(labels)}"
    )

    print(
        f"Number of subjects: "
        f"{len(np.unique(subjects))}"
    )

    print(
        f"\nUnique labels   : "
        f"{np.unique(labels)}"
    )

    print(
        f"Unique subjects : "
        f"{np.unique(subjects)}"
    )


# ============================================================
# 2. LABEL DISTRIBUTION
# ============================================================

def analyze_labels(labels):

    section("2. OVERALL LABEL DISTRIBUTION")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    total = len(labels)

    for label, count in zip(
        unique_labels,
        counts
    ):

        percentage = (
            count / total
        ) * 100

        print(
            f"Label {label}: "
            f"{count:6d} samples "
            f"({percentage:6.2f}%)"
        )

    print("\nInterpretation:")

    if len(counts) == 3:

        difference = (
            counts.max() -
            counts.min()
        )

        if difference < 0.05 * total:

            print(
                "✓ Classes are relatively balanced."
            )

        else:

            print(
                "⚠ Classes show noticeable imbalance."
            )


# ============================================================
# 3. SUBJECT DISTRIBUTION
# ============================================================

def analyze_subjects(
    subjects,
    labels
):

    section("3. SUBJECT-WISE DISTRIBUTION")

    unique_subjects = np.unique(subjects)

    print(
        f"Total subjects: "
        f"{len(unique_subjects)}"
    )

    for subject in unique_subjects:

        mask = subjects == subject

        subject_labels = labels[mask]

        unique_labels, counts = np.unique(
            subject_labels,
            return_counts=True
        )

        print(
            f"\nSubject {subject}"
        )

        print(
            f"  Total samples: "
            f"{mask.sum()}"
        )

        for label, count in zip(
            unique_labels,
            counts
        ):

            percentage = (
                count /
                mask.sum()
            ) * 100

            print(
                f"  Label {label}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )


# ============================================================
# 4. SUBJECT ORDER
# ============================================================

def analyze_subject_order(subjects):

    section(
        "4. SUBJECT ORDERING / CONTIGUOUS BLOCK CHECK"
    )

    if len(subjects) < 2:

        return

    sorted_order = np.all(
        np.diff(subjects) >= 0
    )

    print(
        f"Subjects globally sorted: "
        f"{sorted_order}"
    )

    unique_subjects = np.unique(subjects)

    print("\nSubject index ranges:")

    for subject in unique_subjects:

        indices = np.where(
            subjects == subject
        )[0]

        start = indices.min()

        end = indices.max()

        print(
            f"Subject {subject:2d}: "
            f"index {start:5d} -> "
            f"{end:5d} "
            f"({len(indices)} samples)"
        )

    contiguous = True

    for subject in unique_subjects:

        indices = np.where(
            subjects == subject
        )[0]

        expected = np.arange(
            indices.min(),
            indices.max() + 1
        )

        if not np.array_equal(
            indices,
            expected
        ):

            contiguous = False

    if contiguous:

        print(
            "\n✓ Every subject occupies "
            "one contiguous block."
        )

    else:

        print(
            "\n⚠ Subjects are not stored "
            "in single contiguous blocks."
        )


# ============================================================
# 5. LABEL ORDER
# ============================================================

def analyze_label_order(
    labels,
    subjects
):

    section(
        "5. LABEL ORDERING / RUN ANALYSIS"
    )

    transitions = np.sum(
        labels[1:] != labels[:-1]
    )

    print(
        f"Total label transitions: "
        f"{transitions}"
    )

    print(
        f"Number of label runs: "
        f"{1 + transitions}"
    )

    print(
        "\nFirst 100 labels:"
    )

    print(labels[:100])

    print(
        "\nLabel transitions per subject:"
    )

    for subject in np.unique(subjects):

        subject_labels = labels[
            subjects == subject
        ]

        if len(subject_labels) > 1:

            transitions_subject = np.sum(
                subject_labels[1:]
                != subject_labels[:-1]
            )

        else:

            transitions_subject = 0

        print(
            f"Subject {subject:2d}: "
            f"{transitions_subject} transitions"
        )


# ============================================================
# 6. NAN / INF
# ============================================================

def analyze_missing_values(data):

    section("6. NaN / INFINITY CHECK")

    nan_count = np.isnan(data).sum()

    inf_count = np.isinf(data).sum()

    print(
        f"NaN values : {nan_count}"
    )

    print(
        f"Inf values : {inf_count}"
    )

    if nan_count == 0 and inf_count == 0:

        print(
            "✓ No NaN or Inf values detected."
        )

    else:

        print(
            "⚠ Dataset contains invalid "
            "numerical values."
        )


# ============================================================
# 7. GLOBAL STATISTICS
# ============================================================

def analyze_global_statistics(data):

    section(
        "7. GLOBAL DATA STATISTICS"
    )

    print(
        f"Minimum : {np.min(data):.6f}"
    )

    print(
        f"Maximum : {np.max(data):.6f}"
    )

    print(
        f"Mean    : {np.mean(data):.6f}"
    )

    print(
        f"Std     : {np.std(data):.6f}"
    )

    print(
        f"Median  : {np.median(data):.6f}"
    )


# ============================================================
# 8. 5 x 62 REPRESENTATION
# ============================================================

def analyze_representation(data):

    section(
        "8. ANALYSIS OF 5 x 62 REPRESENTATION"
    )

    if data.ndim != 3:

        print(
            f"⚠ Expected 3D data, "
            f"received {data.ndim}D."
        )

        return

    samples, dim1, dim2 = data.shape

    print(
        f"Samples : {samples}"
    )

    print(
        f"Dim-1   : {dim1}"
    )

    print(
        f"Dim-2   : {dim2}"
    )

    print(
        f"\nEach sample has shape "
        f"({dim1}, {dim2})"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "We are NOT assuming what the "
        "5 dimensions represent."
    )

    # --------------------------------------------------------
    # Row statistics
    # --------------------------------------------------------

    subsection_title = (
        f"Statistics for each of "
        f"the {dim1} rows"
    )

    subsection(subsection_title)

    for row in range(dim1):

        row_data = data[:, row, :]

        print(
            f"Row {row}: "
            f"mean={np.mean(row_data):.6f}, "
            f"std={np.std(row_data):.6f}, "
            f"min={np.min(row_data):.6f}, "
            f"max={np.max(row_data):.6f}"
        )

    # --------------------------------------------------------
    # Column statistics
    # --------------------------------------------------------

    subsection_title = (
        f"Statistics for each of "
        f"the {dim2} columns"
    )

    subsection(subsection_title)

    for col in range(dim2):

        column_data = data[:, :, col]

        print(
            f"Column {col:2d}: "
            f"mean={np.mean(column_data):.6f}, "
            f"std={np.std(column_data):.6f}, "
            f"min={np.min(column_data):.6f}, "
            f"max={np.max(column_data):.6f}"
        )


# ============================================================
# 9. SAMPLE INSPECTION
# ============================================================

def inspect_samples(
    data,
    labels,
    subjects
):

    section(
        "9. FIRST SAMPLE INSPECTION"
    )

    number_to_show = min(
        5,
        len(data)
    )

    np.set_printoptions(
        precision=4,
        suppress=True,
        linewidth=120
    )

    for i in range(number_to_show):

        print(
            f"\nSample {i}"
        )

        print(
            f"Subject : "
            f"{subjects[i]}"
        )

        print(
            f"Label   : "
            f"{labels[i]}"
        )

        print(
            f"Shape   : "
            f"{data[i].shape}"
        )

        print("\nData:")

        print(data[i])


# ============================================================
# 10. CLASS STATISTICS
# ============================================================

def analyze_class_statistics(
    data,
    labels
):

    section(
        "10. CLASS-WISE DATA STATISTICS"
    )

    unique_labels = np.unique(labels)

    for label in unique_labels:

        class_data = data[
            labels == label
        ]

        print(
            f"\nLabel {label}"
        )

        print(
            f"Samples : "
            f"{len(class_data)}"
        )

        print(
            f"Mean    : "
            f"{np.mean(class_data):.6f}"
        )

        print(
            f"Std     : "
            f"{np.std(class_data):.6f}"
        )

        print(
            f"Min     : "
            f"{np.min(class_data):.6f}"
        )

        print(
            f"Max     : "
            f"{np.max(class_data):.6f}"
        )


# ============================================================
# 11. SUBJECT STATISTICS
# ============================================================

def analyze_subject_statistics(
    data,
    subjects
):

    section(
        "11. SUBJECT-WISE DATA STATISTICS"
    )

    for subject in np.unique(subjects):

        subject_data = data[
            subjects == subject
        ]

        print(
            f"Subject {subject:2d}: "
            f"samples={len(subject_data):4d}, "
            f"mean={np.mean(subject_data):.4f}, "
            f"std={np.std(subject_data):.4f}"
        )


# ============================================================
# 12. SAMPLES PER SUBJECT
# ============================================================

def analyze_samples_per_subject(
    subjects
):

    section(
        "12. SAMPLES PER SUBJECT"
    )

    _, counts = np.unique(
        subjects,
        return_counts=True
    )

    print(
        f"Minimum samples per subject: "
        f"{counts.min()}"
    )

    print(
        f"Maximum samples per subject: "
        f"{counts.max()}"
    )

    print(
        f"Mean samples per subject: "
        f"{counts.mean():.2f}"
    )

    if np.all(counts == counts[0]):

        print(
            f"\n✓ Every subject has exactly "
            f"{counts[0]} samples."
        )

    else:

        print(
            "\n⚠ Sample count differs "
            "between subjects."
        )


# ============================================================
# 13. SUBJECT × LABEL
# ============================================================

def analyze_subject_label_consistency(
    subjects,
    labels
):

    section(
        "13. SUBJECT × LABEL CONSISTENCY"
    )

    unique_subjects = np.unique(subjects)

    unique_labels = np.unique(labels)

    print("\nDistribution table:\n")

    header = "Subject"

    for label in unique_labels:

        header += f"\tLabel {label}"

    print(header)

    print("-" * 60)

    matrix = []

    for subject in unique_subjects:

        row = [subject]

        for label in unique_labels:

            count = np.sum(
                (subjects == subject)
                &
                (labels == label)
            )

            row.append(count)

        matrix.append(row)

        print(
            f"{subject:7d}\t"
            +
            "\t".join(
                str(value)
                for value in row[1:]
            )
        )

    matrix = np.array(matrix)

    print(
        "\nChecking whether every subject "
        "has every label..."
    )

    all_present = True

    for i, subject in enumerate(
        unique_subjects
    ):

        counts = matrix[i, 1:]

        if np.any(counts == 0):

            all_present = False

            print(
                f"⚠ Subject {subject} "
                f"does not contain all labels."
            )

    if all_present:

        print(
            "✓ Every subject contains "
            "every emotion class."
        )


# ============================================================
# 14. SAMPLE COUNT STRUCTURE
# ============================================================

def analyze_sample_count_structure(
    subjects
):

    section(
        "14. SAMPLE COUNT STRUCTURE"
    )

    _, counts = np.unique(
        subjects,
        return_counts=True
    )

    if np.all(counts == counts[0]):

        n = int(counts[0])

        print(
            f"Each subject contains "
            f"{n} samples."
        )

        print(
            "\nInteger factors:"
        )

        for i in range(
            1,
            int(np.sqrt(n)) + 1
        ):

            if n % i == 0:

                print(
                    f"{i} × {n // i} = {n}"
                )

        print(
            "\nNOTE:"
        )

        print(
            "These are only mathematical factors."
        )

        print(
            "They do NOT prove the meaning of "
            "the dimensions."
        )

    else:

        print(
            "Sample counts differ "
            "between subjects."
        )


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

def print_final_summary(
    data,
    labels,
    subjects
):

    section(
        "FINAL SEED DATASET SUMMARY"
    )

    print(
        f"Dataset shape      : {data.shape}"
    )

    print(
        f"Number of samples  : {len(data)}"
    )

    print(
        f"Number of subjects : "
        f"{len(np.unique(subjects))}"
    )

    print(
        f"Number of classes  : "
        f"{len(np.unique(labels))}"
    )

    print(
        f"Labels             : "
        f"{np.unique(labels).tolist()}"
    )

    print(
        f"Subjects           : "
        f"{np.unique(subjects).tolist()}"
    )

    print(
        f"NaN count          : "
        f"{np.isnan(data).sum()}"
    )

    print(
        f"Inf count          : "
        f"{np.isinf(data).sum()}"
    )

    print(
        "\nIMPORTANT NEXT DECISION:"
    )

    print(
        "Before SEED preprocessing/modeling, "
        "we need to establish what the 5 × 62 "
        "representation actually represents."
    )

    print(
        "Do NOT apply DE/PSD or raw-EEG filtering "
        "to this representation until that is confirmed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print("SEED DATASET ANALYSIS")

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    data, labels, subjects = load_seed_arrays()

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    analyze_basic_structure(
        data,
        labels,
        subjects
    )

    analyze_labels(
        labels
    )

    analyze_subjects(
        subjects,
        labels
    )

    analyze_subject_order(
        subjects
    )

    analyze_label_order(
        labels,
        subjects
    )

    analyze_missing_values(
        data
    )

    analyze_global_statistics(
        data
    )

    analyze_representation(
        data
    )

    inspect_samples(
        data,
        labels,
        subjects
    )

    analyze_class_statistics(
        data,
        labels
    )

    analyze_subject_statistics(
        data,
        subjects
    )

    analyze_samples_per_subject(
        subjects
    )

    analyze_subject_label_consistency(
        subjects,
        labels
    )

    analyze_sample_count_structure(
        subjects
    )

    print_final_summary(
        data,
        labels,
        subjects
    )

    print("\n")

    print("=" * 70)

    print("SEED ANALYSIS COMPLETE")

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()