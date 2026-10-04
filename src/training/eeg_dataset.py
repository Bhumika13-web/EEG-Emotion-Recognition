import torch
from torch.utils.data import Dataset


class EEGSequenceDataset(Dataset):

    def __init__(
        self,
        features,
        labels,
    ):
        """
        Dataset for pre-built EEG temporal sequences.

        Expected features shape:

            (samples, 30, 32, 10)

        Where:

            samples = EEG trials
            30      = temporal windows
            32      = EEG electrodes
            10      = DE + PSD features

        Labels shape:

            (samples,)
        """

        self.features = torch.tensor(
            features,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if self.features.ndim != 4:

            raise ValueError(
                "Expected features with 4 dimensions: "
                "(samples, sequence, nodes, features). "
                f"Got shape: {self.features.shape}"
            )

        if self.features.shape[1] != 30:

            raise ValueError(
                "Expected sequence length = 30. "
                f"Got: {self.features.shape[1]}"
            )

        if self.features.shape[2] != 32:

            raise ValueError(
                "Expected 32 EEG channels. "
                f"Got: {self.features.shape[2]}"
            )

        if self.features.shape[3] != 10:

            raise ValueError(
                "Expected 10 features "
                "(5 DE + 5 PSD). "
                f"Got: {self.features.shape[3]}"
            )

        if len(self.features) != len(self.labels):

            raise ValueError(
                "Number of features and labels "
                "must be equal."
            )

    def __len__(self):

        return len(
            self.features
        )

    def __getitem__(self, index):

        x = self.features[index]

        y = self.labels[index]

        return x, y