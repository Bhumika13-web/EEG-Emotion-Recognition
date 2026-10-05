import numpy as np


def create_subject_split(
    subjects,
    train_ratio=0.75,
    val_ratio=0.125,
    seed=42,
):
    """
    Create a subject-wise train/validation/test split.

    Each subject remains entirely within one split.
    This prevents subject-level data leakage.
    """

    subjects = np.asarray(
        subjects
    )

    unique_subjects = np.unique(
        subjects
    )

    rng = np.random.default_rng(seed)

    shuffled_subjects = unique_subjects.copy()

    rng.shuffle(
        shuffled_subjects
    )

    n_subjects = len(
        shuffled_subjects
    )

    n_train = int(
        n_subjects * train_ratio
    )

    n_val = int(
        n_subjects * val_ratio
    )

    train_subjects = shuffled_subjects[
        :n_train
    ]

    val_subjects = shuffled_subjects[
        n_train:n_train + n_val
    ]

    test_subjects = shuffled_subjects[
        n_train + n_val:
    ]

    return (
        train_subjects,
        val_subjects,
        test_subjects,
    )


def get_split_indices(
    subjects,
    train_subjects,
    val_subjects,
    test_subjects,
):
    """
    Convert subject IDs into sample indices.
    """

    subjects = np.asarray(
        subjects
    )

    train_indices = np.where(
        np.isin(
            subjects,
            train_subjects,
        )
    )[0]

    val_indices = np.where(
        np.isin(
            subjects,
            val_subjects,
        )
    )[0]

    test_indices = np.where(
        np.isin(
            subjects,
            test_subjects,
        )
    )[0]

    return (
        train_indices,
        val_indices,
        test_indices,
    )


if __name__ == "__main__":

    # Example: 32 DEAP subjects.
    subjects = np.arange(32)

    (
        train_subjects,
        val_subjects,
        test_subjects,
    ) = create_subject_split(
        subjects
    )

    print(
        "Train subjects:",
        train_subjects,
    )

    print(
        "Validation subjects:",
        val_subjects,
    )

    print(
        "Test subjects:",
        test_subjects,
    )

    print(
        "Number of training subjects:",
        len(train_subjects),
    )

    print(
        "Number of validation subjects:",
        len(val_subjects),
    )

    print(
        "Number of test subjects:",
        len(test_subjects),
    )

    # Verify there is no overlap.
    assert len(
        set(train_subjects)
        & set(val_subjects)
    ) == 0

    assert len(
        set(train_subjects)
        & set(test_subjects)
    ) == 0

    assert len(
        set(val_subjects)
        & set(test_subjects)
    ) == 0

    print(
        "No subject overlap detected."
    )