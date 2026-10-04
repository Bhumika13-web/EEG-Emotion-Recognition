import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch


# =========================================================
# Configuration
# =========================================================

SAMPLING_RATE = 128

WINDOW_SECONDS = 4
OVERLAP_SECONDS = 2

WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS
STEP_SAMPLES = SAMPLING_RATE * (
    WINDOW_SECONDS - OVERLAP_SECONDS
)

# Only the first 32 channels are EEG in DEAP.
N_EEG_CHANNELS = 32

FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 30),
    "gamma": (30, 45),
}


# =========================================================
# Load DEAP subject
# =========================================================

def load_subject(file_path):
    """
    Load one DEAP preprocessed subject.

    Returns
    -------
    data : np.ndarray
        Shape: (40, 32, 8064)

    labels : np.ndarray
        Shape: (40, 4)
        Columns:
        valence, arousal, dominance, liking
    """

    with open(
        file_path,
        "rb",
    ) as file:

        subject = pickle.load(
            file,
            encoding="latin1",
        )

    data = subject["data"][:, :N_EEG_CHANNELS, :]
    labels = subject["labels"]

    return data.astype(np.float32), labels.astype(
        np.float32
    )


# =========================================================
# Filtering
# =========================================================

def bandpass_filter(
    data,
    lowcut=1.0,
    highcut=45.0,
    order=4,
):
    """
    Band-pass filter EEG between 1 and 45 Hz.
    """

    nyquist = SAMPLING_RATE / 2

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band",
    )

    filtered = filtfilt(
        b,
        a,
        data,
        axis=-1,
    )

    return filtered.astype(np.float32)


# =========================================================
# Segmentation
# =========================================================

def segment_trials(data):
    """
    Convert each 63-second DEAP trial into
    4-second windows with 2-second overlap.

    Input:
        (trials, channels, samples)

    Output:
        (trials, windows, channels, samples)
    """

    n_trials = data.shape[0]

    windows_per_trial = (
        (
            data.shape[-1]
            - WINDOW_SAMPLES
        )
        // STEP_SAMPLES
    ) + 1

    segmented = np.zeros(
        (
            n_trials,
            windows_per_trial,
            data.shape[1],
            WINDOW_SAMPLES,
        ),
        dtype=np.float32,
    )

    for trial_index in range(n_trials):

        for window_index in range(
            windows_per_trial
        ):

            start = (
                window_index
                * STEP_SAMPLES
            )

            end = (
                start
                + WINDOW_SAMPLES
            )

            segmented[
                trial_index,
                window_index,
            ] = data[
                trial_index,
                :,
                start:end,
            ]

    return segmented


# =========================================================
# Differential Entropy
# =========================================================

def calculate_differential_entropy(
    signal,
):
    """
    Differential entropy assuming
    approximately Gaussian EEG amplitude.

        DE = 0.5 * log(2*pi*e*variance)
    """

    variance = np.var(
        signal,
        axis=-1,
    )

    variance = np.maximum(
        variance,
        1e-12,
    )

    return (
        0.5
        * np.log(
            2
            * np.pi
            * np.e
            * variance
        )
    )


def extract_de_features(
    segments,
):
    """
    Extract DE features for all five
    EEG frequency bands.

    Input:
        (trials, windows, channels, samples)

    Output:
        (trials, windows, channels, 5)
    """

    n_trials = segments.shape[0]
    n_windows = segments.shape[1]
    n_channels = segments.shape[2]

    features = np.zeros(
        (
            n_trials,
            n_windows,
            n_channels,
            len(FREQUENCY_BANDS),
        ),
        dtype=np.float32,
    )

    for band_index, (
        band_name,
        (low, high),
    ) in enumerate(
        FREQUENCY_BANDS.items()
    ):

        nyquist = SAMPLING_RATE / 2

        b, a = butter(
            4,
            [
                low / nyquist,
                high / nyquist,
            ],
            btype="band",
        )

        band_signal = filtfilt(
            b,
            a,
            segments,
            axis=-1,
        )

        de = calculate_differential_entropy(
            band_signal
        )

        features[
            :,
            :,
            :,
            band_index,
        ] = de

    return features


# =========================================================
# PSD
# =========================================================

def extract_psd_features(
    segments,
):
    """
    Extract PSD band-power features.

    Output:
        (trials, windows, channels, 5)
    """

    n_trials = segments.shape[0]
    n_windows = segments.shape[1]
    n_channels = segments.shape[2]

    features = np.zeros(
        (
            n_trials,
            n_windows,
            n_channels,
            len(FREQUENCY_BANDS),
        ),
        dtype=np.float32,
    )

    for trial_index in range(n_trials):

        for window_index in range(
            n_windows
        ):

            for channel_index in range(
                n_channels
            ):

                signal = segments[
                    trial_index,
                    window_index,
                    channel_index,
                ]

                frequencies, psd = welch(
                    signal,
                    fs=SAMPLING_RATE,
                    nperseg=min(
                        256,
                        len(signal),
                    ),
                )

                for band_index, (
                    band_name,
                    (low, high),
                ) in enumerate(
                    FREQUENCY_BANDS.items()
                ):

                    mask = (
                        (frequencies >= low)
                        & (frequencies < high)
                    )

                    if not np.any(mask):
                        continue

                    power = np.trapezoid(
                        psd[mask],
                        frequencies[mask],
                    )

                    features[
                        trial_index,
                        window_index,
                        channel_index,
                        band_index,
                    ] = power

    return features


# =========================================================
# Emotion labels
# =========================================================

def create_emotion_labels(
    labels,
    threshold=5.0,
):
    """
    Convert DEAP valence/arousal ratings
    into four quadrants.

    Class 0 = Low Valence + Low Arousal
    Class 1 = Low Valence + High Arousal
    Class 2 = High Valence + Low Arousal
    Class 3 = High Valence + High Arousal

    Valence = labels[:, 0]
    Arousal = labels[:, 1]
    """

    valence = labels[:, 0]
    arousal = labels[:, 1]

    low_valence = valence < threshold
    high_valence = valence >= threshold

    low_arousal = arousal < threshold
    high_arousal = arousal >= threshold

    emotion_labels = np.zeros(
        len(labels),
        dtype=np.int64,
    )

    emotion_labels[
        low_valence & low_arousal
    ] = 0

    emotion_labels[
        low_valence & high_arousal
    ] = 1

    emotion_labels[
        high_valence & low_arousal
    ] = 2

    emotion_labels[
        high_valence & high_arousal
    ] = 3

    return emotion_labels


# =========================================================
# Complete dataset builder
# =========================================================

def build_deap_dataset(
    file_path,
):
    """
    Complete DEAP preprocessing pipeline.

    Returns
    -------
    combined_features:
        (40, 30, 32, 10)

    emotion_labels:
        (40,)
    """

    print("Loading DEAP subject...")

    data, labels = load_subject(
        file_path
    )

    print(
        "Original data shape:",
        data.shape,
    )

    # ---------------------------------------------
    # Filtering
    # ---------------------------------------------

    filtered = bandpass_filter(
        data
    )

    print(
        "Filtered data shape:",
        filtered.shape,
    )

    # ---------------------------------------------
    # Segmentation
    # ---------------------------------------------

    segments = segment_trials(
        filtered
    )

    print(
        "Segmented data shape:",
        segments.shape,
    )

    # ---------------------------------------------
    # Differential Entropy
    # ---------------------------------------------

    print(
        "Extracting Differential Entropy..."
    )

    de_features = extract_de_features(
        segments
    )

    print(
        "DE feature shape:",
        de_features.shape,
    )

    # ---------------------------------------------
    # PSD
    # ---------------------------------------------

    print(
        "Extracting PSD..."
    )

    psd_features = extract_psd_features(
        segments
    )

    print(
        "PSD feature shape:",
        psd_features.shape,
    )

    # ---------------------------------------------
    # Combine DE + PSD
    # ---------------------------------------------

    combined_features = np.concatenate(
        [
            de_features,
            psd_features,
        ],
        axis=-1,
    )

    print(
        "Combined feature shape:",
        combined_features.shape,
    )

    # ---------------------------------------------
    # Emotion labels
    # ---------------------------------------------

    emotion_labels = create_emotion_labels(
        labels
    )

    print(
        "Emotion labels shape:",
        emotion_labels.shape,
    )

    return (
        combined_features,
        emotion_labels,
    )


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    project_root = Path(
        __file__
    ).resolve().parents[2]

    subject_file = (
        project_root
        / "data"
        / "DEAP"
        / "deap-dataset"
        / "data_preprocessed_python"
        / "s01.dat"
    )

    print(
        "Testing DEAP dataset builder"
    )
    print(
        "============================"
    )

    X, y = build_deap_dataset(
        subject_file
    )

    print()
    print(
        "FINAL DATASET"
    )
    print(
        "-------------"
    )

    print(
        "X shape:",
        X.shape,
    )

    print(
        "y shape:",
        y.shape,
    )

    print(
        "Unique classes:",
        np.unique(
            y,
            return_counts=True,
        ),
    )

    print(
        "Expected X shape:",
        "(40, 30, 32, 10)",
    )

    print(
        "Expected y shape:",
        "(40,)",
    )

    print()
    print(
        "DEAP dataset builder test completed successfully."
    )