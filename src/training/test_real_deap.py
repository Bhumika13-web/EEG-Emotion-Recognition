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


# Five EEG frequency bands
BANDS = {
    "delta": (4, 8),
    "theta": (8, 13),
    "alpha": (13, 30),
    "beta": (30, 45),
    "gamma": (45, 50),
}


# ============================================================
# LOAD DEAP SUBJECT
# ============================================================

def load_subject(subject_id):

    project_root = Path(
        __file__
    ).resolve().parents[2]

    data_directory = (
        project_root
        / "data"
        / "DEAP"
        / "deap-dataset"
        / "data_preprocessed_python"
    )

    file_path = (
        data_directory
        / f"s{subject_id:02d}.dat"
    )

    print(
        "Loading:",
        file_path,
    )

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

    # Keep only the first 32 EEG channels.
    # DEAP also contains peripheral channels.
    data = data[:, :32, :]

    return data, labels


# ============================================================
# SEGMENT EEG TRIAL
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
        windows
    )


# ============================================================
# DIFFERENTIAL ENTROPY
# ============================================================

def calculate_de(window):

    de_features = []

    # Calculate DE separately for each
    # of the five frequency bands.
    for low, high in BANDS.values():

        # Band-pass filter
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

        # Variance for each EEG channel
        variance = np.var(
            filtered,
            axis=-1,
        )

        # Prevent log(0)
        variance = np.maximum(
            variance,
            1e-10,
        )

        # Differential Entropy
        de = 0.5 * np.log(
            2 * np.pi * np.e * variance
        )

        # Shape:
        # (32,)
        de_features.append(
            de
        )

    # Five arrays of shape (32,)
    # become:
    #
    # (32, 5)
    #
    # 32 EEG channels
    # 5 frequency bands

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

    band_features = []

    for low, high in BANDS.values():

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        band_power = np.trapezoid(
            power[..., mask],
            frequencies[mask],
            axis=-1,
        )

        # Shape:
        # (32,)
        band_features.append(
            band_power
        )

    # Convert five band arrays into:
    #
    # (32, 5)

    return np.stack(
        band_features,
        axis=-1,
    )


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(subject_id):

    data, labels = load_subject(
        subject_id
    )

    print()
    print(
        "Original data shape:"
    )

    print(
        data.shape
    )

    print(
        "Labels shape:"
    )

    print(
        labels.shape
    )

    # --------------------------------------------------------
    # Process first trial only
    # --------------------------------------------------------

    trial = data[0]

    print()
    print(
        "First trial shape:"
    )

    print(
        trial.shape
    )

    # --------------------------------------------------------
    # Segment trial
    # --------------------------------------------------------

    windows = segment_trial(
        trial
    )

    print()
    print(
        "Windows shape:"
    )

    print(
        windows.shape
    )

    # --------------------------------------------------------
    # Calculate DE and PSD
    # --------------------------------------------------------

    de_features = []

    psd_features = []

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

    # Convert lists to NumPy arrays

    de_features = np.array(
        de_features
    )

    psd_features = np.array(
        psd_features
    )

    print()
    print(
        "DE shape:"
    )

    print(
        de_features.shape
    )

    print(
        "PSD shape:"
    )

    print(
        psd_features.shape
    )

    # --------------------------------------------------------
    # Combine DE + PSD
    # --------------------------------------------------------

    features = np.concatenate(
        [
            de_features,
            psd_features,
        ],
        axis=-1,
    )

    print()
    print(
        "Combined feature shape:"
    )

    print(
        features.shape
    )

    # --------------------------------------------------------
    # Create binary labels
    # --------------------------------------------------------

    original_valence = labels[0, 0]
    original_arousal = labels[0, 1]

    valence = int(
        original_valence >= 5
    )

    arousal = int(
        original_arousal >= 5
    )

    print()
    print(
        "Original Valence:"
    )

    print(
        original_valence
    )

    print(
        "Original Arousal:"
    )

    print(
        original_arousal
    )

    print()
    print(
        "Binary Valence:",
        valence,
    )

    print(
        "Binary Arousal:",
        arousal,
    )

    return (
        features,
        valence,
        arousal,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Test subject 1
    features, valence, arousal = process_subject(
        subject_id=1
    )

    print()
    print("=" * 60)
    print(
        "REAL DEAP PIPELINE TEST"
    )
    print("=" * 60)

    print(
        "Final feature shape:",
        features.shape,
    )

    print(
        "Valence label:",
        valence,
    )

    print(
        "Arousal label:",
        arousal,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    # 30 windows
    # 32 EEG channels
    # 10 features = 5 DE + 5 PSD

    assert features.shape == (
        30,
        32,
        10,
    )

    assert valence in [
        0,
        1,
    ]

    assert arousal in [
        0,
        1,
    ]

    print()
    print(
        "REAL DEAP PIPELINE TEST PASSED!"
    )