import numpy as np
from scipy.signal import welch


SAMPLING_RATE = 128

# EEG frequency bands
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 30),
    "gamma": (30, 45),
}


def calculate_differential_entropy(
    eeg_segments,
    sampling_rate=SAMPLING_RATE,
):
    """
    Calculate Differential Entropy (DE) features
    for EEG segments.

    Parameters
    ----------
    eeg_segments : np.ndarray
        EEG segments with shape:

        (segments, channels, samples)

    sampling_rate : int
        EEG sampling frequency.

    Returns
    -------
    de_features : np.ndarray
        Differential Entropy features with shape:

        (segments, channels, 5)

        The five features correspond to:

        Delta
        Theta
        Alpha
        Beta
        Gamma
    """

    eeg_segments = np.asarray(
        eeg_segments,
        dtype=np.float32,
    )

    if eeg_segments.ndim != 3:
        raise ValueError(
            "Expected EEG segments with shape "
            "(segments, channels, samples)."
        )

    n_segments = eeg_segments.shape[0]
    n_channels = eeg_segments.shape[1]

    n_bands = len(FREQUENCY_BANDS)

    de_features = np.zeros(
        (n_segments, n_channels, n_bands),
        dtype=np.float32,
    )

    for segment_index in range(n_segments):

        for channel_index in range(n_channels):

            signal = eeg_segments[
                segment_index,
                channel_index,
            ]

            # Estimate power spectral density.
            frequencies, psd = welch(
                signal,
                fs=sampling_rate,
                nperseg=min(
                    256,
                    len(signal),
                ),
            )

            for band_index, (
                band_name,
                (low_freq, high_freq),
            ) in enumerate(
                FREQUENCY_BANDS.items()
            ):

                # Select frequencies inside the band.
                mask = (
                    (frequencies >= low_freq)
                    & (frequencies < high_freq)
                )

                if not np.any(mask):
                    continue

                # Estimate band power.
                band_power = np.trapezoid(
                    psd[mask],
                    frequencies[mask],
                )

                # Prevent log(0).
                band_power = max(
                    band_power,
                    1e-12,
                )

                # Differential entropy for
                # a Gaussian signal:
                #
                # DE = 0.5 * log(2*pi*e*variance)
                #
                # We use band power as the
                # variance estimate.
                de = 0.5 * np.log(
                    2 * np.pi * np.e * band_power
                )

                de_features[
                    segment_index,
                    channel_index,
                    band_index,
                ] = de

    return de_features


if __name__ == "__main__":

    from pathlib import Path
    import sys

    sys.path.append(
        str(
            Path(__file__).resolve().parents[1]
            / "preprocessing"
        )
    )

    from deap_loader import load_deap_subject
    from filtering import bandpass_filter
    from segmentation import segment_eeg

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

    # Load DEAP.
    data, labels = load_deap_subject(
        subject_file
    )

    print(
        "Original data shape:",
        data.shape,
    )

    # Filter EEG.
    filtered_data = bandpass_filter(
        data
    )

    print(
        "Filtered data shape:",
        filtered_data.shape,
    )

    # Segment EEG.
    segments, trial_indices = segment_eeg(
        filtered_data,
        window_seconds=4,
        overlap_seconds=2,
    )

    print(
        "Segmented data shape:",
        segments.shape,
    )

    # Calculate DE features.
    de_features = calculate_differential_entropy(
        segments
    )

    print(
        "DE feature shape:",
        de_features.shape,
    )

    print(
        "Expected shape:",
        "(1200, 32, 5)",
    )

    print(
        "Frequency bands:",
        list(FREQUENCY_BANDS.keys()),
    )

    print(
        "Differential Entropy extraction "
        "completed successfully."
    )