import pickle
import glob

import numpy as np


def load_all_labels(data_directory):
    """
    Load DEAP labels from all subjects.
    """

    files = sorted(
        glob.glob(
            str(data_directory / "s*.dat")
        )
    )

    labels = []

    for file_path in files:

        with open(
            file_path,
            "rb",
        ) as file:

            subject = pickle.load(
                file,
                encoding="latin1",
            )

        labels.append(
            subject["labels"]
        )

    return np.concatenate(
        labels,
        axis=0,
    )


def create_binary_labels(labels):
    """
    Create binary Valence and Arousal labels.

    0 = Low
    1 = High
    """

    valence = (
        labels[:, 0] >= 5
    ).astype(np.int64)

    arousal = (
        labels[:, 1] >= 5
    ).astype(np.int64)

    return valence, arousal


def calculate_class_weights(y):
    """
    Calculate inverse-frequency class weights.
    """

    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    total = len(y)

    n_classes = len(classes)

    weights = (
        total
        / (
            n_classes
            * counts
        )
    )

    return classes, counts, weights


if __name__ == "__main__":

    from pathlib import Path

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

    labels = load_all_labels(
        data_directory
    )

    # Subject split from Step 30.
    train_subjects = np.array([
        31, 19, 7, 27, 26, 18,
        5, 22, 28, 10, 24, 23,
        20, 9, 6, 16, 3, 0,
        15, 17, 25, 12, 21, 11
    ])

    # Each subject has 40 trials.
    subject_ids = np.repeat(
        np.arange(32),
        40,
    )

    train_mask = np.isin(
        subject_ids,
        train_subjects,
    )

    train_labels = labels[
        train_mask
    ]

    valence, arousal = (
        create_binary_labels(
            train_labels
        )
    )

    print(
        "Training subjects:",
        len(train_subjects),
    )

    print(
        "Training trials:",
        len(train_labels),
    )

    print()
    print("VALENCE CLASS WEIGHTS")

    classes, counts, weights = (
        calculate_class_weights(
            valence
        )
    )

    for c, count, weight in zip(
        classes,
        counts,
        weights,
    ):
        print(
            f"Class {c}: "
            f"{count} samples, "
            f"weight = {weight:.4f}"
        )

    print()
    print("AROUSAL CLASS WEIGHTS")

    classes, counts, weights = (
        calculate_class_weights(
            arousal
        )
    )

    for c, count, weight in zip(
        classes,
        counts,
        weights,
    ):
        print(
            f"Class {c}: "
            f"{count} samples, "
            f"weight = {weight:.4f}"
        )

    print()
    print(
        "Class weight calculation completed successfully."
    )