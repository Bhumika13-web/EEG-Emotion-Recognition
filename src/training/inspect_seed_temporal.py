"""
Inspect SEED temporal structure.

Purpose:
    Determine whether the Kaggle SEED representation preserves a
    meaningful temporal ordering that can be used to construct
    GCN + GRU sequences.

This script DOES NOT modify any dataset files.
"""

from pathlib import Path
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "SEED"

DATA_FILE = DATA_DIR / "DatasetCaricatoNoImage.npz"
LABEL_FILE = DATA_DIR / "LabelsNoImage.npz"
SUBJECT_FILE = DATA_DIR / "SubjectsNoImage.npz"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_array(npz_file):
    """
    Return the first array stored inside an NPZ file.
    """
    data = np.load(npz_file)

    print(f"\nFile: {npz_file.name}")
    print(f"Keys: {data.files}")

    if len(data.files) == 0:
        raise ValueError(f"No arrays found in {npz_file}")

    array = data[data.files[0]]

    print(f"Selected key: {data.files[0]}")
    print(f"Shape: {array.shape}")
    print(f"Dtype: {array.dtype}")

    return array


def get_runs(values):
    """
    Find consecutive runs of identical values.

    Example:
        [2,2,2,1,1,0,0]
    becomes:
        [(2, 0, 3), (1, 3, 5), (0, 5, 7)]
    """

    values = np.asarray(values)

    if len(values) == 0:
        return []

    runs = []

    start = 0
    current = values[0]

    for i in range(1, len(values)):
        if values[i] != current:
            runs.append((int(current), start, i))
            start = i
            current = values[i]

    runs.append((int(current), start, len(values)))

    return runs


def print_separator():
    print("\n" + "=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

print_separator()
print("SEED TEMPORAL STRUCTURE INSPECTION")
print_separator()

print("\nProject root:")
print(PROJECT_ROOT)

print("\nData directory:")
print(DATA_DIR)


# Check files
for file_path in [DATA_FILE, LABEL_FILE, SUBJECT_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n{file_path}\n"
            f"Please check your SEED data directory."
        )

print("\n All SEED files found.")


# ============================================================
# LOAD ARRAYS
# ============================================================

print_separator()
print("LOADING DATA")
print_separator()

data = find_array(DATA_FILE)
labels = find_array(LABEL_FILE)
subjects = find_array(SUBJECT_FILE)


# ============================================================
# BASIC VALIDATION
# ============================================================

print_separator()
print("BASIC VALIDATION")
print_separator()

print(f"\nData shape     : {data.shape}")
print(f"Labels shape   : {labels.shape}")
print(f"Subjects shape : {subjects.shape}")

if len(data) != len(labels) or len(data) != len(subjects):
    raise ValueError("Data, labels and subjects do not have the same number of samples.")

print("\n Number of samples matches across all files.")


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

print_separator()
print("DATA INFORMATION")
print_separator()

print(f"\nTotal samples: {len(data)}")

unique_subjects = np.unique(subjects)
unique_labels = np.unique(labels)

print(f"Number of subjects: {len(unique_subjects)}")
print(f"Subjects: {unique_subjects}")

print(f"\nNumber of emotion classes: {len(unique_labels)}")
print(f"Labels: {unique_labels}")

print("\nLabel interpretation used in the current project:")
print("0 = Negative")
print("1 = Neutral")
print("2 = Positive")


# ============================================================
# SUBJECT COUNTS
# ============================================================

print_separator()
print("SAMPLES PER SUBJECT")
print_separator()

for subject in unique_subjects:
    indices = np.where(subjects == subject)[0]

    print(
        f"Subject {subject:2d}: "
        f"{len(indices):5d} samples | "
        f"indices {indices[0]:5d} - {indices[-1]:5d}"
    )


# ============================================================
# CHECK SUBJECT CONTIGUITY
# ============================================================

print_separator()
print("SUBJECT CONTIGUITY CHECK")
print_separator()

subject_changes = np.where(subjects[1:] != subjects[:-1])[0] + 1

print(f"\nNumber of subject boundaries: {len(subject_changes)}")

print("\nSubject boundaries:")

for boundary in subject_changes:
    previous_subject = subjects[boundary - 1]
    next_subject = subjects[boundary]

    print(
        f"Index {boundary:5d}: "
        f"Subject {previous_subject} -> Subject {next_subject}"
    )

# Check if each subject appears in exactly one contiguous block
contiguous = True

for subject in unique_subjects:

    indices = np.where(subjects == subject)[0]

    expected = np.arange(indices[0], indices[-1] + 1)

    if not np.array_equal(indices, expected):
        contiguous = False
        print(f" Subject {subject} is NOT contiguous.")

if contiguous:
    print("\n Every subject occupies one contiguous block.")


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print_separator()
print("GLOBAL LABEL DISTRIBUTION")
print_separator()

for label in unique_labels:

    count = np.sum(labels == label)
    percentage = 100 * count / len(labels)

    print(
        f"Label {label}: "
        f"{count:5d} samples "
        f"({percentage:.2f}%)"
    )


# ============================================================
# LABEL RUN ANALYSIS
# ============================================================

print_separator()
print("LABEL RUN ANALYSIS")
print_separator()

print(
    "\nA label run means consecutive samples belonging "
    "to the same emotion label."
)

all_run_lengths = []

for subject in unique_subjects:

    subject_indices = np.where(subjects == subject)[0]

    subject_labels = labels[subject_indices]

    runs = get_runs(subject_labels)

    print_separator()
    print(f"SUBJECT {subject}")

    print(f"Number of label runs: {len(runs)}")

    print("\nRuns:")

    for run_number, (label, start, end) in enumerate(runs, start=1):

        length = end - start

        all_run_lengths.append(length)

        global_start = subject_indices[start]
        global_end = subject_indices[end - 1]

        print(
            f"Run {run_number:2d}: "
            f"Label={label} | "
            f"Length={length:4d} | "
            f"Global indices={global_start:5d}-{global_end:5d}"
        )


# ============================================================
# RUN LENGTH STATISTICS
# ============================================================

print_separator()
print("LABEL RUN LENGTH STATISTICS")
print_separator()

all_run_lengths = np.array(all_run_lengths)

print(f"\nTotal runs: {len(all_run_lengths)}")

print(f"Minimum run length : {all_run_lengths.min()}")
print(f"Maximum run length : {all_run_lengths.max()}")
print(f"Mean run length    : {all_run_lengths.mean():.2f}")
print(f"Median run length  : {np.median(all_run_lengths):.2f}")

print("\nRun length distribution:")

unique_lengths, counts = np.unique(
    all_run_lengths,
    return_counts=True
)

for length, count in zip(unique_lengths, counts):

    percentage = 100 * count / len(all_run_lengths)

    print(
        f"Length {length:4d}: "
        f"{count:4d} runs "
        f"({percentage:.2f}%)"
    )


# ============================================================
# LABEL TRANSITIONS
# ============================================================

print_separator()
print("LABEL TRANSITIONS")
print_separator()

total_transitions = 0

for subject in unique_subjects:

    indices = np.where(subjects == subject)[0]
    subject_labels = labels[indices]

    transitions = np.where(
        subject_labels[1:] != subject_labels[:-1]
    )[0] + 1

    total_transitions += len(transitions)

    print(
        f"\nSubject {subject:2d}: "
        f"{len(transitions)} label transitions"
    )

    if len(transitions) > 0:

        print("Transition positions:")

        for position in transitions:

            global_index = indices[position]

            old_label = subject_labels[position - 1]
            new_label = subject_labels[position]

            print(
                f"  Global index {global_index:5d}: "
                f"{old_label} -> {new_label}"
            )

print(f"\nTotal label transitions: {total_transitions}")


# ============================================================
# CHECK CONSECUTIVE SAMPLE INDICES
# ============================================================

print_separator()
print("SAMPLE INDEX CONTINUITY")
print_separator()

index_array = np.arange(len(data))

# Because the arrays themselves are stored sequentially,
# verify that each subject's samples occupy consecutive indices.

continuity_ok = True

for subject in unique_subjects:

    indices = np.where(subjects == subject)[0]

    differences = np.diff(indices)

    if len(differences) > 0:

        if not np.all(differences == 1):

            continuity_ok = False

            print(
                f" Subject {subject}: "
                "indices are not consecutive."
            )

if continuity_ok:
    print(
        "\n Samples belonging to each subject are "
        "stored consecutively."
    )


# ============================================================
# FIRST SAMPLES INSPECTION
# ============================================================

print_separator()
print("FIRST 100 SAMPLES")
print_separator()

print("\nIndex | Subject | Label")

for i in range(min(100, len(data))):

    print(
        f"{i:5d} | "
        f"{subjects[i]:7d} | "
        f"{labels[i]}"
    )


# ============================================================
# LAST SAMPLES OF EACH SUBJECT
# ============================================================

print_separator()
print("SUBJECT END CHECK")
print_separator()

for subject in unique_subjects:

    indices = np.where(subjects == subject)[0]

    print(f"\nSubject {subject}:")

    start = indices[0]
    end = indices[-1]

    print(
        f"First index: {start}, "
        f"Last index: {end}"
    )

    print("Last 10 labels:")

    print(labels[indices[-10:]])


# ============================================================
# CHECK DATA VALIDITY
# ============================================================

print_separator()
print("DATA VALIDITY")
print_separator()

print(f"\nNaN values in data: {np.isnan(data).sum()}")
print(f"Inf values in data: {np.isinf(data).sum()}")

print(f"NaN values in labels: {np.isnan(labels).sum()}")
print(f"NaN values in subjects: {np.isnan(subjects).sum()}")

if not np.isnan(data).any() and not np.isinf(data).any():
    print("\n No NaN or Inf values found in EEG features.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print_separator()
print("FINAL TEMPORAL STRUCTURE SUMMARY")
print_separator()

print(f"""
Total samples       : {len(data)}
Total subjects      : {len(unique_subjects)}
Total emotion class : {len(unique_labels)}

Data shape          : {data.shape}

Subject contiguous  : {contiguous}
Index continuity    : {continuity_ok}

Total label runs    : {len(all_run_lengths)}
Total transitions   : {total_transitions}

Minimum run length  : {all_run_lengths.min()}
Maximum run length  : {all_run_lengths.max()}
Mean run length     : {all_run_lengths.mean():.2f}
Median run length   : {np.median(all_run_lengths):.2f}
""")


print_separator()
print("IMPORTANT")
print_separator()

print("""
This script does NOT prove that every consecutive sample is a
time-contiguous EEG window.

It only establishes the ordering and label structure present in
the Kaggle representation.

Before implementing GCN + GRU, use these results to determine
whether the samples can safely be grouped into temporal sequences.
""")

print("\n Temporal inspection completed successfully.")