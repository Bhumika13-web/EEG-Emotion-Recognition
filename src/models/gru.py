import torch
import torch.nn as nn


class EEGGRUEncoder(nn.Module):
    """
    GRU encoder for temporal EEG representations.

    Input:
        (batch, sequence_length, input_features)

    Output:
        (batch, hidden_features)
    """

    def __init__(
        self,
        input_features=32,
        hidden_features=64,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_features,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:
            (batch, sequence_length, input_features)

        Returns
        -------
        torch.Tensor
            Final temporal representation:
            (batch, hidden_features)
        """

        _, hidden = self.gru(x)

        # hidden shape:
        # (num_layers, batch, hidden_features)

        final_hidden = hidden[-1]

        return final_hidden


class EEGGRUClassifier(nn.Module):
    """
    GRU classifier for temporal EEG representations.

    Note:
    num_classes=3 is currently only a model test
    setting. Final DEAP emotion-label definition
    will be decided during dataset preparation.
    """

    def __init__(
        self,
        input_features=32,
        hidden_features=64,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = EEGGRUEncoder(
            input_features=input_features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.classifier = nn.Linear(
            hidden_features,
            num_classes,
        )

    def forward(self, x):

        temporal_embedding = self.encoder(x)

        output = self.classifier(
            temporal_embedding
        )

        return output


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -------------------------------------------------
    # Simulated GCN output sequence
    # -------------------------------------------------
    #
    # 8 trials in a batch
    # 30 windows per trial
    # 32-dimensional GCN embedding
    #

    x = torch.randn(
        8,
        30,
        32,
        dtype=torch.float32,
        device=device,
    )

    model = EEGGRUEncoder(
        input_features=32,
        hidden_features=64,
        num_layers=2,
    ).to(device)

    embedding = model(x)

    print(
        "Input shape:",
        x.shape,
    )

    print(
        "Temporal embedding shape:",
        embedding.shape,
    )

    # -------------------------------------------------
    # Test classifier
    # -------------------------------------------------

    classifier = EEGGRUClassifier(
        input_features=32,
        hidden_features=64,
        num_layers=2,
        num_classes=3,
    ).to(device)

    output = classifier(x)

    print(
        "Classifier output shape:",
        output.shape,
    )

    print(
        "GRU model test completed successfully."
    )