import pickle
from pathlib import Path

import numpy as np

from scipy.signal import (
    butter,
    sosfiltfilt,
    welch,
)


# ============================================================
# CONFIGURATION
# ============================================================

FS = 128

WINDOW_SECONDS = 4
STEP_SECONDS = 2

WINDOW_SIZE = FS * WINDOW_SECONDS
STEP_SIZE = FS * STEP_SECONDS


# ============================================================
# EEG FREQUENCY BANDS
# ============================================================

BANDS = {
    "delta": (4, 8),
    "theta": (8, 13),
    "alpha": (13, 30),
    "beta": (30, 45),
    "gamma": (45, 50),
}


# ============================================================
# SUBJECT SPLIT
# ============================================================

# IMPORTANT:
# These are ZERO-BASED project subject IDs: 0-31.
#
# DEAP filenames are ONE-BASED:
# subject 0 -> s01.dat
# subject 1 -> s02.dat
# ...
# subject 31 -> s32.dat

TRAIN_SUBJECTS = [
    31, 19, 7, 27, 26, 18,
    5, 22, 28, 10, 24, 23,
    20, 9, 6, 16, 3, 0,
    15, 17, 25, 12, 21, 11,
]

VAL_SUBJECTS = [
    14, 30, 2, 4,
]

TEST_SUBJECTS = [
    29, 1, 13, 8,
]


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "DEAP"
    / "deap-dataset"
    / "data_preprocessed_python"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DEAP SUBJECT
# ============================================================

def load_subject(subject_id):

    # --------------------------------------------------------
    # Convert our 0-based subject ID to DEAP's 1-based
    # filename.
    #
    # Example:
    # 0  -> s01.dat
    # 1  -> s02.dat
    # 31 -> s32.dat
    # --------------------------------------------------------

    deap_subject_number = subject_id + 1

    file_path = (
        DATA_DIRECTORY
        / f"s{deap_subject_number:02d}.dat"
    )

    print(
        f"Loading subject {subject_id}: "
        f"{file_path.name}"
    )

    # --------------------------------------------------------
    # Check that the file actually exists
    # --------------------------------------------------------

    if not file_path.exists():

        raise FileNotFoundError(
            f"DEAP file not found: {file_path}"
        )

    # --------------------------------------------------------
    # Load .dat file
    # --------------------------------------------------------

    with open(
        file_path,
        "rb",
    ) as file:

        subject = pickle.load(
            file,
            encoding="latin1",
        )

    data = subject["data"]
    labels = subject["labels"]

    # --------------------------------------------------------
    # Keep only the first 32 EEG channels.
    #
    # DEAP contains EEG + peripheral channels.
    # --------------------------------------------------------

    data = data[:, :32, :]

    return data, labels


# ============================================================
# SEGMENT ONE EEG TRIAL
# ============================================================

def segment_trial(trial):

    windows = []

    for start in range(
        0,
        trial.shape[-1] - WINDOW_SIZE + 1,
        STEP_SIZE,
    ):

        end = start + WINDOW_SIZE

        window = trial[
            :,
            start:end,
        ]

        windows.append(
            window
        )

    return np.array(
        windows,
        dtype=np.float32,
    )


# ============================================================
# DIFFERENTIAL ENTROPY
# ============================================================

def calculate_de(window):

    de_features = []

    # Calculate DE separately for each frequency band.
    for low, high in BANDS.values():

        # ----------------------------------------------------
        # Band-pass filter
        # ----------------------------------------------------

        sos = butter(
            4,
            [low, high],
            btype="bandpass",
            fs=FS,
            output="sos",
        )

        filtered = sosfiltfilt(
            sos,
            window,
            axis=-1,
        )

        # ----------------------------------------------------
        # Variance for each EEG channel
        # ----------------------------------------------------

        variance = np.var(
            filtered,
            axis=-1,
        )

        # Prevent log(0)
        variance = np.maximum(
            variance,
            1e-10,
        )

        # ----------------------------------------------------
        # Differential Entropy
        # ----------------------------------------------------

        de = 0.5 * np.log(
            2 * np.pi * np.e * variance
        )

        # Shape:
        # (32,)
        de_features.append(
            de
        )

    # Five bands  32 channels
    #
    # Result:
    # (32, 5)

    return np.stack(
        de_features,
        axis=-1,
    )


# ============================================================
# PSD BAND POWER
# ============================================================

def calculate_psd(window):

    frequencies, power = welch(
        window,
        fs=FS,
        axis=-1,
        nperseg=256,
    )

    psd_features = []

    for low, high in BANDS.values():

        # ----------------------------------------------------
        # Select frequencies belonging to this band
        # ----------------------------------------------------

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        # ----------------------------------------------------
        # Integrate power inside frequency band
        # ----------------------------------------------------

        band_power = np.trapezoid(
            power[..., mask],
            frequencies[mask],
            axis=-1,
        )

        # Shape:
        # (32,)

        psd_features.append(
            band_power
        )

    # Result:
    # (32, 5)

    return np.stack(
        psd_features,
        axis=-1,
    )


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(trial):

    # --------------------------------------------------------
    # Segment trial
    # --------------------------------------------------------

    windows = segment_trial(
        trial
    )

    de_features = []
    psd_features = []

    # --------------------------------------------------------
    # Extract DE + PSD from every window
    # --------------------------------------------------------

    for window in windows:

        de = calculate_de(
            window
        )

        psd = calculate_psd(
            window
        )

        de_features.append(
            de
        )

        psd_features.append(
            psd
        )

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    de_features = np.array(
        de_features,
        dtype=np.float32,
    )

    psd_features = np.array(
        psd_features,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Combine DE + PSD
    #
    # DE:
    # (30, 32, 5)
    #
    # PSD:
    # (30, 32, 5)
    #
    # Combined:
    # (30, 32, 10)
    # --------------------------------------------------------

    features = np.concatenate(
        [
            de_features,
            psd_features,
        ],
        axis=-1,
    )

    return features


# ============================================================
# BUILD ONE DATASET SPLIT
# ============================================================

def build_split(
    subjects,
    split_name,
):

    all_features = []

    valence_labels = []
    arousal_labels = []

    subject_ids = []
    trial_ids = []

    print()
    print("=" * 60)

    print(
        f"BUILDING {split_name.upper()} DATASET"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Process every subject
    # --------------------------------------------------------

    for subject_id in subjects:

        data, labels = load_subject(
            subject_id
        )

        print(
            f"  Data shape: {data.shape}"
        )

        # ----------------------------------------------------
        # Process every trial
        # ----------------------------------------------------

        for trial_index in range(
            len(data)
        ):

            features = process_trial(
                data[trial_index]
            )

            # ------------------------------------------------
            # Save features
            # ------------------------------------------------

            all_features.append(
                features
            )

            # ------------------------------------------------
            # Binary Valence
            #
            # < 5 -> 0
            # >= 5 -> 1
            # ------------------------------------------------

            valence = int(
                labels[
                    trial_index,
                    0
                ] >= 5
            )

            # ------------------------------------------------
            # Binary Arousal
            #
            # < 5 -> 0
            # >= 5 -> 1
            # ------------------------------------------------

            arousal = int(
                labels[
                    trial_index,
                    1
                ] >= 5
            )

            valence_labels.append(
                valence
            )

            arousal_labels.append(
                arousal
            )

            # ------------------------------------------------
            # Keep subject/trial information
            # ------------------------------------------------

            subject_ids.append(
                subject_id
            )

            trial_ids.append(
                trial_index
            )

    # ========================================================
    # CONVERT EVERYTHING TO NUMPY
    # ========================================================

    all_features = np.array(
        all_features,
        dtype=np.float32,
    )

    valence_labels = np.array(
        valence_labels,
        dtype=np.int64,
    )

    arousal_labels = np.array(
        arousal_labels,
        dtype=np.int64,
    )

    subject_ids = np.array(
        subject_ids,
        dtype=np.int64,
    )

    trial_ids = np.array(
        trial_ids,
        dtype=np.int64,
    )

    # ========================================================
    # SAVE DATASET
    # ========================================================

    output_file = (
        OUTPUT_DIRECTORY
        / f"deap_{split_name}.npz"
    )

    np.savez_compressed(
        output_file,
        features=all_features,
        valence=valence_labels,
        arousal=arousal_labels,
        subject_ids=subject_ids,
        trial_ids=trial_ids,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print(
        f"{split_name.upper()} DATASET COMPLETE"
    )

    print(
        "Features:",
        all_features.shape,
    )

    print(
        "Valence:",
        valence_labels.shape,
    )

    print(
        "Arousal:",
        arousal_labels.shape,
    )

    print(
        "Subjects:",
        subject_ids.shape,
    )

    print(
        "Trials:",
        trial_ids.shape,
    )

    print(
        "Saved to:",
        output_file,
    )

    return all_features


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "DEAP DATASET BUILDER"
    )
    print("=" * 60)

    print()

    print(
        "Training subjects:",
        len(TRAIN_SUBJECTS),
    )

    print(
        "Validation subjects:",
        len(VAL_SUBJECTS),
    )

    print(
        "Test subjects:",
        len(TEST_SUBJECTS),
    )

    # ========================================================
    # TRAIN
    # ========================================================

    build_split(
        TRAIN_SUBJECTS,
        "train",
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    build_split(
        VAL_SUBJECTS,
        "val",
    )

    # ========================================================
    # TEST
    # ========================================================

    build_split(
        TEST_SUBJECTS,
        "test",
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 60)
    print(
        "ALL DEAP DATASETS BUILT SUCCESSFULLY!"
    )
    print("=" * 60)