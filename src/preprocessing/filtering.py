import numpy as np
from scipy.signal import butter, sosfiltfilt


SAMPLING_RATE = 128


def bandpass_filter(
    eeg_data,
    low_freq=4.0,
    high_freq=45.0,
    sampling_rate=SAMPLING_RATE,
    filter_order=4,
):
    """
    Apply a Butterworth band-pass filter to EEG data.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data with shape:
        (trials, channels, samples)

    low_freq : float
        Lower cutoff frequency in Hz.

    high_freq : float
        Upper cutoff frequency in Hz.

    sampling_rate : int
        EEG sampling frequency in Hz.

    filter_order : int
        Butterworth filter order.

    Returns
    -------
    filtered_data : np.ndarray
        Filtered EEG data with the same shape as input.
    """

    eeg_data = np.asarray(eeg_data, dtype=np.float32)

    if eeg_data.ndim != 3:
        raise ValueError(
            "Expected EEG data with shape "
            "(trials, channels, samples)."
        )

    nyquist = sampling_rate / 2

    if low_freq <= 0:
        raise ValueError("low_freq must be greater than 0.")

    if high_freq >= nyquist:
        raise ValueError(
            f"high_freq must be less than Nyquist frequency ({nyquist} Hz)."
        )

    if low_freq >= high_freq:
        raise ValueError(
            "low_freq must be smaller than high_freq."
        )

    # Design Butterworth band-pass filter.
    sos = butter(
        filter_order,
        [low_freq, high_freq],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )

    # Filter along the time/sample axis.
    filtered_data = sosfiltfilt(
        sos,
        eeg_data,
        axis=-1,
    )

    return filtered_data.astype(np.float32)


if __name__ == "__main__":

    from pathlib import Path
    import sys

    # Allow importing deap_loader.py from the same folder.
    sys.path.append(str(Path(__file__).resolve().parent))

    from deap_loader import load_deap_subject

    project_root = Path(__file__).resolve().parents[2]

    subject_file = (
        project_root
        / "data"
        / "DEAP"
        / "deap-dataset"
        / "data_preprocessed_python"
        / "s01.dat"
    )

    # Load DEAP data.
    data, labels = load_deap_subject(subject_file)

    print("Original data shape:", data.shape)

    # Apply filtering.
    filtered_data = bandpass_filter(data)

    print("Filtered data shape:", filtered_data.shape)
    print("Filtering completed successfully.")