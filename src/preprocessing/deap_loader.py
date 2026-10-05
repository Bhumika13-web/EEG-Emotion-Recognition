from pathlib import Path
import pickle
import numpy as np


EEG_CHANNELS = 32
SAMPLING_RATE = 128


def load_deap_subject(subject_file):
    """Load one preprocessed DEAP subject."""

    subject_file = Path(subject_file)

    if not subject_file.exists():
        raise FileNotFoundError(
            f"DEAP subject file not found: {subject_file}"
        )

    with open(subject_file, "rb") as file:
        subject = pickle.load(file, encoding="latin1")

    if "data" not in subject or "labels" not in subject:
        raise ValueError(
            "DEAP file must contain 'data' and 'labels'."
        )

    data = np.asarray(subject["data"], dtype=np.float32)
    labels = np.asarray(subject["labels"], dtype=np.float32)

    # DEAP contains 32 EEG channels + 8 peripheral channels.
    # Keep only the 32 EEG channels.
    data = data[:, :EEG_CHANNELS, :]

    return data, labels


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    subject_file = (
        project_root
        / "data"
        / "DEAP"
        / "deap-dataset"
        / "data_preprocessed_python"
        / "s01.dat"
    )

    data, labels = load_deap_subject(subject_file)

    print("DEAP loader test")
    print("----------------")
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    print("Number of trials:", data.shape[0])
    print("Number of EEG channels:", data.shape[1])
    print("Samples per trial:", data.shape[2])
    print("Sampling rate:", SAMPLING_RATE)