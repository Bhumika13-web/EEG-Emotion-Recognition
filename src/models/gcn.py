import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class EEGGCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder for EEG.

    Input:
        x          -> (nodes, 10)
        edge_index -> (2, edges)

    Output:
        graph_embedding -> (1, embedding_dim)
    """

    def __init__(
        self,
        input_features=10,
        hidden_features=64,
        embedding_dim=32,
        dropout=0.3,
    ):
        super().__init__()

        self.gcn1 = GCNConv(
            input_features,
            hidden_features,
        )

        self.gcn2 = GCNConv(
            hidden_features,
            embedding_dim,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GCN.

        Parameters
        ----------
        x : torch.Tensor
            Node features.

        edge_index : torch.Tensor
            Graph connectivity.

        batch : torch.Tensor, optional
            Graph assignment for each node.

        Returns
        -------
        torch.Tensor
            Graph-level embedding.
        """

        x = self.gcn1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.gcn2(x, edge_index)
        x = self.relu(x)

        # If no batch vector is provided,
        # treat all 32 nodes as one graph.
        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        x = global_mean_pool(x, batch)

        return x


class EEGGCNClassifier(nn.Module):
    """
    Standalone GCN classifier.

    Useful as a spatial baseline before
    combining GCN with GRU.
    """

    def __init__(
        self,
        input_features=10,
        hidden_features=64,
        embedding_dim=32,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = EEGGCNEncoder(
            input_features=input_features,
            hidden_features=hidden_features,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

    def forward(
        self,
        x,
        edge_index,
        batch=None,
    ):
        embedding = self.encoder(
            x,
            edge_index,
            batch,
        )

        output = self.classifier(embedding)

        return output


if __name__ == "__main__":

    from pathlib import Path
    import sys
    import numpy as np

    # Import graph utilities.
    graph_path = (
        Path(__file__).resolve().parents[1]
        / "graph"
    )

    sys.path.append(str(graph_path))

    from electrode_graph import (
        create_electrode_graph,
        graph_to_adjacency,
        add_self_loops,
    )

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -------------------------------------------------
    # Create electrode graph
    # -------------------------------------------------

    graph = create_electrode_graph()

    adjacency = graph_to_adjacency(graph)
    adjacency = add_self_loops(adjacency)

    # Convert adjacency matrix to edge_index.
    edge_index = torch.tensor(
        np.array(np.nonzero(adjacency)),
        dtype=torch.long,
    )

    # -------------------------------------------------
    # Example fused EEG features
    # -------------------------------------------------

    # One EEG window:
    # 32 electrodes  10 features
    x = torch.randn(
        32,
        10,
        dtype=torch.float32,
    )

    x = x.to(device)
    edge_index = edge_index.to(device)

    # -------------------------------------------------
    # Test encoder
    # -------------------------------------------------

    encoder = EEGGCNEncoder(
        input_features=10,
        hidden_features=64,
        embedding_dim=32,
    ).to(device)

    embedding = encoder(
        x,
        edge_index,
    )

    print("Input shape:", x.shape)
    print(
        "Edge index shape:",
        edge_index.shape,
    )
    print(
        "GCN embedding shape:",
        embedding.shape,
    )

    # -------------------------------------------------
    # Test classifier
    # -------------------------------------------------

    classifier = EEGGCNClassifier(
        input_features=10,
        hidden_features=64,
        embedding_dim=32,
        num_classes=3,
    ).to(device)

    output = classifier(
        x,
        edge_index,
    )

    print(
        "Classifier output shape:",
        output.shape,
    )

    print("GCN model test completed successfully.")