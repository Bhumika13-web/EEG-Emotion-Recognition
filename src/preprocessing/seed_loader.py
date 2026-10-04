from pathlib import Path

import numpy as np


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED_DIR = PROJECT_ROOT / "data" / "SEED"

DATA_FILE = SEED_DIR / "DatasetCaricatoNoImage.npz"
LABEL_FILE = SEED_DIR / "LabelsNoImage.npz"
SUBJECT_FILE = SEED_DIR / "SubjectsNoImage.npz"


# ============================================================
# SEED LOADER
# ============================================================

class SEEDLoader:
    """
    Loader for the Kaggle SEED dataset currently stored as:

        DatasetCaricatoNoImage.npz
        LabelsNoImage.npz
        SubjectsNoImage.npz

    Known representation:

        data    -> (50910, 5, 62)
        labels  -> (50910,)
        subjects -> (50910,)

    The 5 x 62 representation is preserved exactly as stored.
    We do NOT assume that the 5 dimension represents raw EEG
    time samples until the dataset structure is verified.
    """

    def __init__(
        self,
        data_file=DATA_FILE,
        label_file=LABEL_FILE,
        subject_file=SUBJECT_FILE,
    ):

        self.data_file = Path(data_file)
        self.label_file = Path(label_file)
        self.subject_file = Path(subject_file)

        self.data = None
        self.labels = None
        self.subjects = None

    # ========================================================
    # CHECK FILES
    # ========================================================

    def check_files(self):

        print("=" * 70)
        print("CHECKING SEED FILES")
        print("=" * 70)

        files = {
            "Dataset": self.data_file,
            "Labels": self.label_file,
            "Subjects": self.subject_file,
        }

        for name, path in files.items():

            print(
                f"{name:10s}: {path}"
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"Missing SEED file:\n{path}"
                )

            print(
                f"{'':10s}  ✓ Found"
            )

    # ========================================================
    # LOAD NPZ FILE
    # ========================================================

    @staticmethod
    def _load_npz_array(path, description):

        print(
            f"\nLoading {description}:"
        )

        print(path)

        archive = np.load(
            path,
            allow_pickle=True
        )

        print(
            "Keys:",
            archive.files
        )

        # ----------------------------------------------------
        # If there is only one array, use it directly.
        # ----------------------------------------------------

        if len(archive.files) == 1:

            array = archive[archive.files[0]]

        else:

            # ------------------------------------------------
            # Prefer common names if available
            # ------------------------------------------------

            preferred_names = [
                "data",
                "dataset",
                "labels",
                "subjects",
                "arr_0",
            ]

            selected_key = None

            for key in preferred_names:

                if key in archive.files:

                    selected_key = key
                    break

            if selected_key is None:

                raise ValueError(
                    f"Could not determine the correct array "
                    f"inside {path}.\n"
                    f"Available keys: {archive.files}"
                )

            array = archive[selected_key]

        print(
            "Shape:",
            array.shape
        )

        print(
            "Dtype:",
            array.dtype
        )

        return array

    # ========================================================
    # LOAD DATASET
    # ========================================================

    def load(self):

        self.check_files()

        print("\n" + "=" * 70)
        print("LOADING SEED DATASET")
        print("=" * 70)

        self.data = self._load_npz_array(
            self.data_file,
            "EEG dataset"
        )

        self.labels = self._load_npz_array(
            self.label_file,
            "emotion labels"
        )

        self.subjects = self._load_npz_array(
            self.subject_file,
            "subject IDs"
        )

        # ----------------------------------------------------
        # Convert labels / subjects to normal NumPy arrays
        # ----------------------------------------------------

        self.labels = np.asarray(
            self.labels
        ).reshape(-1)

        self.subjects = np.asarray(
            self.subjects
        ).reshape(-1)

        # ----------------------------------------------------
        # Basic consistency checks
        # ----------------------------------------------------

        if len(self.data) != len(self.labels):

            raise ValueError(
                "Number of data samples does not match "
                "number of labels."
            )

        if len(self.data) != len(self.subjects):

            raise ValueError(
                "Number of data samples does not match "
                "number of subject IDs."
            )

        print("\n" + "=" * 70)
        print("SEED DATASET LOADED SUCCESSFULLY")
        print("=" * 70)

        print(
            f"Data shape     : {self.data.shape}"
        )

        print(
            f"Labels shape   : {self.labels.shape}"
        )

        print(
            f"Subjects shape : {self.subjects.shape}"
        )

        return (
            self.data,
            self.labels,
            self.subjects
        )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    def summary(self):

        if self.data is None:

            raise RuntimeError(
                "Dataset has not been loaded yet. "
                "Call load() first."
            )

        print("\n" + "=" * 70)
        print("SEED DATASET SUMMARY")
        print("=" * 70)

        # ----------------------------------------------------
        # General information
        # ----------------------------------------------------

        print(
            f"Total samples : {len(self.data)}"
        )

        print(
            f"Data shape    : {self.data.shape}"
        )

        print(
            f"Labels shape  : {self.labels.shape}"
        )

        print(
            f"Subjects shape: {self.subjects.shape}"
        )

        # ----------------------------------------------------
        # Label information
        # ----------------------------------------------------

        unique_labels, label_counts = np.unique(
            self.labels,
            return_counts=True
        )

        print("\nEmotion labels:")

        for label, count in zip(
            unique_labels,
            label_counts
        ):

            percentage = (
                count /
                len(self.labels) *
                100
            )

            print(
                f"  Label {label}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

        # ----------------------------------------------------
        # Subject information
        # ----------------------------------------------------

        unique_subjects, subject_counts = np.unique(
            self.subjects,
            return_counts=True
        )

        print(
            f"\nNumber of subjects: "
            f"{len(unique_subjects)}"
        )

        print("\nSubject distribution:")

        for subject, count in zip(
            unique_subjects,
            subject_counts
        ):

            print(
                f"  Subject {subject}: "
                f"{count} samples"
            )

    # ========================================================
    # CHECK LABEL × SUBJECT DISTRIBUTION
    # ========================================================

    def subject_label_distribution(self):

        if self.data is None:

            raise RuntimeError(
                "Call load() before checking "
                "subject distribution."
            )

        print("\n" + "=" * 70)
        print("SUBJECT × LABEL DISTRIBUTION")
        print("=" * 70)

        unique_subjects = np.unique(
            self.subjects
        )

        unique_labels = np.unique(
            self.labels
        )

        header = (
            "Subject".ljust(12)
            + "".join(
                f"Label {label}".rjust(15)
                for label in unique_labels
            )
        )

        print(header)
        print("-" * len(header))

        for subject in unique_subjects:

            row = (
                f"{str(subject):<12}"
            )

            for label in unique_labels:

                count = np.sum(
                    (
                        self.subjects == subject
                    )
                    &
                    (
                        self.labels == label
                    )
                )

                row += (
                    f"{count:>15}"
                )

            print(row)

    # ========================================================
    # GET SUBJECTS
    # ========================================================

    def get_subject_ids(self):

        if self.subjects is None:

            raise RuntimeError(
                "Call load() first."
            )

        return np.unique(
            self.subjects
        )

    # ========================================================
    # GET DATA FOR SELECTED SUBJECTS
    # ========================================================

    def get_by_subjects(
        self,
        subject_ids
    ):

        if self.data is None:

            raise RuntimeError(
                "Call load() first."
            )

        subject_ids = np.asarray(
            subject_ids
        )

        mask = np.isin(
            self.subjects,
            subject_ids
        )

        return (
            self.data[mask],
            self.labels[mask],
            self.subjects[mask]
        )


# ============================================================
# TEST / INSPECTION
# ============================================================

def main():

    print("\n")
    print("#" * 70)
    print("# SEED DATASET LOADER TEST")
    print("#" * 70)

    loader = SEEDLoader()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data, labels, subjects = loader.load()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    loader.summary()

    # --------------------------------------------------------
    # Subject × label distribution
    # --------------------------------------------------------

    loader.subject_label_distribution()

    # --------------------------------------------------------
    # Show first sample
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST SAMPLE INSPECTION")
    print("=" * 70)

    print(
        "First sample shape:",
        data[0].shape
    )

    print(
        "First sample label:",
        labels[0]
    )

    print(
        "First sample subject:",
        subjects[0]
    )

    print(
        "\nFirst sample statistics:"
    )

    print(
        f"Min  : {data[0].min():.6f}"
    )

    print(
        f"Max  : {data[0].max():.6f}"
    )

    print(
        f"Mean : {data[0].mean():.6f}"
    )

    print(
        f"Std  : {data[0].std():.6f}"
    )

    # --------------------------------------------------------
    # Test subject selection
    # --------------------------------------------------------

    unique_subjects = loader.get_subject_ids()

    if len(unique_subjects) >= 2:

        test_subjects = unique_subjects[:2]

        selected_data, selected_labels, selected_subjects = (
            loader.get_by_subjects(
                test_subjects
            )
        )

        print("\n" + "=" * 70)
        print("SUBJECT SELECTION TEST")
        print("=" * 70)

        print(
            "Selected subjects:",
            test_subjects
        )

        print(
            "Selected data shape:",
            selected_data.shape
        )

        print(
            "Selected labels shape:",
            selected_labels.shape
        )

        print(
            "Selected subjects shape:",
            selected_subjects.shape
        )

    print("\n" + "=" * 70)
    print("SEED LOADER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()