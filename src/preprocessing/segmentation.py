import numpy as np


SAMPLING_RATE = 128


def segment_eeg(
    eeg_data,
    window_seconds=4,
    overlap_seconds=2,
    sampling_rate=SAMPLING_RATE,
):
    """
    Divide EEG trials into overlapping time windows.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data with shape:
        (trials, channels, samples)

    window_seconds : int
        Length of each window in seconds.

    overlap_seconds : int
        Overlap between consecutive windows in seconds.

    sampling_rate : int
        EEG sampling frequency in Hz.

    Returns
    -------
    segments : np.ndarray
        Segmented EEG data with shape:
        (segments, channels, window_samples)

    trial_indices : np.ndarray
        Original trial number for each segment.
    """

    eeg_data = np.asarray(eeg_data, dtype=np.float32)

    if eeg_data.ndim != 3:
        raise ValueError(
            "Expected EEG data with shape "
            "(trials, channels, samples)."
        )

    window_samples = int(window_seconds * sampling_rate)
    overlap_samples = int(overlap_seconds * sampling_rate)

    if window_samples <= 0:
        raise ValueError("Window length must be greater than 0.")

    if overlap_samples < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap_samples >= window_samples:
        raise ValueError(
            "Overlap must be smaller than the window length."
        )

    step_samples = window_samples - overlap_samples

    segments = []
    trial_indices = []

    for trial_index in range(eeg_data.shape[0]):

        trial = eeg_data[trial_index]

        for start in range(
            0,
            trial.shape[-1] - window_samples + 1,
            step_samples,
        ):

            end = start + window_samples

            segment = trial[:, start:end]

            segments.append(segment)
            trial_indices.append(trial_index)

    if not segments:
        raise ValueError(
            "No segments could be created. "
            "Check the window size and EEG data length."
        )

    segments = np.stack(segments)
    trial_indices = np.asarray(trial_indices, dtype=np.int32)

    return segments, trial_indices


if __name__ == "__main__":

    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

    from deap_loader import load_deap_subject
    from filtering import bandpass_filter

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

    # Filter EEG.
    filtered_data = bandpass_filter(data)

    print("Filtered data shape:", filtered_data.shape)

    # Segment EEG.
    segments, trial_indices = segment_eeg(
        filtered_data,
        window_seconds=4,
        overlap_seconds=2,
    )

    print("Segmented data shape:", segments.shape)
    print("Trial indices shape:", trial_indices.shape)
    print("Number of segments:", len(segments))
    print("Window samples:", segments.shape[-1])
    print("Segmentation completed successfully.")