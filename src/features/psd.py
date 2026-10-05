import numpy as np
from scipy.signal import welch


SAMPLING_RATE = 128

FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 30),
    "gamma": (30, 45),
}


def calculate_psd_features(
    eeg_segments,
    sampling_rate=SAMPLING_RATE,
):
    """
    Calculate PSD band-power features.

    Input:
        (segments, channels, samples)

    Output:
        (segments, channels, 5)
    """

    eeg_segments = np.asarray(
        eeg_segments,
        dtype=np.float32,
    )

    if eeg_segments.ndim != 3:
        raise ValueError(
            "Expected EEG data with shape "
            "(segments, channels, samples)."
        )

    n_segments = eeg_segments.shape[0]
    n_channels = eeg_segments.shape[1]
    n_bands = len(FREQUENCY_BANDS)

    psd_features = np.zeros(
        (n_segments, n_channels, n_bands),
        dtype=np.float32,
    )

    for segment_index in range(n_segments):
        for channel_index in range(n_channels):

            signal = eeg_segments[
                segment_index,
                channel_index,
            ]

            frequencies, psd = welch(
                signal,
                fs=sampling_rate,
                nperseg=min(256, len(signal)),
            )

            for band_index, (
                band_name,
                (low_freq, high_freq),
            ) in enumerate(FREQUENCY_BANDS.items()):

                mask = (
                    (frequencies >= low_freq)
                    & (frequencies < high_freq)
                )

                if not np.any(mask):
                    continue

                band_power = np.trapezoid(
                    psd[mask],
                    frequencies[mask],
                )

                psd_features[
                    segment_index,
                    channel_index,
                    band_index,
                ] = band_power

    return psd_features


if __name__ == "__main__":

    from pathlib import Path
    import sys

    preprocessing_path = (
        Path(__file__).resolve().parents[1]
        / "preprocessing"
    )

    sys.path.append(str(preprocessing_path))

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

    data, labels = load_deap_subject(subject_file)

    print("Original data shape:", data.shape)

    filtered_data = bandpass_filter(data)

    print("Filtered data shape:", filtered_data.shape)

    segments, trial_indices = segment_eeg(
        filtered_data,
        window_seconds=4,
        overlap_seconds=2,
    )

    print("Segmented data shape:", segments.shape)

    psd_features = calculate_psd_features(segments)

    print("PSD feature shape:", psd_features.shape)
    print("Expected shape:", "(1200, 32, 5)")
    print(
        "Frequency bands:",
        list(FREQUENCY_BANDS.keys()),
    )
    print("PSD extraction completed successfully.")