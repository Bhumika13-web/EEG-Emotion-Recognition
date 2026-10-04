import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, global_mean_pool


# ============================================================
# GRAPH IMPORT
# ============================================================

GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "graph"
)

sys.path.append(str(GRAPH_PATH))

from electrode_graph import (
    create_electrode_graph,
    graph_to_adjacency,
    add_self_loops,
)


# ============================================================
# CREATE EDGE INDEX
# ============================================================

def create_edge_index(device):

    graph = create_electrode_graph()

    adjacency = graph_to_adjacency(graph)

    adjacency = add_self_loops(adjacency)

    edge_index = torch.tensor(
        np.array(np.nonzero(adjacency)),
        dtype=torch.long,
        device=device,
    )

    return edge_index, adjacency


# ============================================================
# SPATIAL GCN
# ============================================================

class SpatialGCN(nn.Module):

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

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        batch_size, num_nodes, num_features = x.shape

        # ----------------------------------------------------
        # Flatten nodes from all samples in the batch
        # ----------------------------------------------------

        x = x.reshape(
            batch_size * num_nodes,
            num_features,
        )

        # ----------------------------------------------------
        # Create batched graph
        # ----------------------------------------------------

        edge_indices = []

        for batch_index in range(batch_size):

            offset = batch_index * num_nodes

            shifted_edges = (
                edge_index + offset
            )

            edge_indices.append(
                shifted_edges
            )

        batched_edge_index = torch.cat(
            edge_indices,
            dim=1,
        )

        # ----------------------------------------------------
        # First GCN layer
        # ----------------------------------------------------

        x = self.gcn1(
            x,
            batched_edge_index,
        )

        x = self.relu(x)

        x = self.dropout(x)

        # ----------------------------------------------------
        # Second GCN layer
        # ----------------------------------------------------

        x = self.gcn2(
            x,
            batched_edge_index,
        )

        x = self.relu(x)

        # ----------------------------------------------------
        # Global mean pooling
        # ----------------------------------------------------

        batch_vector = torch.arange(
            batch_size,
            device=x.device,
        ).repeat_interleave(
            num_nodes
        )

        x = global_mean_pool(
            x,
            batch_vector,
        )

        return x


# ============================================================
# GCN + GRU MODEL
# ============================================================

class EEGGCNGRU(nn.Module):

    def __init__(
        self,
        input_features=10,
        gcn_hidden=64,
        gcn_embedding=32,
        gru_hidden=64,
        gru_layers=2,
        num_classes=2,
        dropout=0.3,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Spatial GCN
        # ----------------------------------------------------

        self.spatial_gcn = SpatialGCN(
            input_features=input_features,
            hidden_features=gcn_hidden,
            embedding_dim=gcn_embedding,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Temporal GRU
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=gcn_embedding,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=(
                dropout
                if gru_layers > 1
                else 0.0
            ),
        )

        # ----------------------------------------------------
        # Classification layer
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            gru_hidden,
            num_classes,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        batch_size, sequence_length, nodes, features = x.shape

        spatial_embeddings = []

        # ----------------------------------------------------
        # Apply GCN to every temporal window
        # ----------------------------------------------------

        for time_index in range(
            sequence_length
        ):

            window = x[
                :,
                time_index,
                :,
                :,
            ]

            embedding = self.spatial_gcn(
                window,
                edge_index,
            )

            spatial_embeddings.append(
                embedding
            )

        # ----------------------------------------------------
        # Convert spatial embeddings
        # into temporal sequence
        # ----------------------------------------------------

        spatial_sequence = torch.stack(
            spatial_embeddings,
            dim=1,
        )

        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        _, hidden = self.gru(
            spatial_sequence
        )

        temporal_embedding = hidden[-1]

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        output = self.classifier(
            temporal_embedding
        )

        return output


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("GCN + GRU MODEL TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    edge_index, adjacency = create_edge_index(
        device
    )

    print()
    print(
        "Number of EEG nodes:",
        adjacency.shape[0],
    )

    print(
        "Adjacency shape:",
        adjacency.shape,
    )

    print(
        "Edge index shape:",
        edge_index.shape,
    )

    # --------------------------------------------------------
    # Dummy EEG input
    # --------------------------------------------------------

    x = torch.randn(
        4,
        30,
        32,
        10,
        dtype=torch.float32,
        device=device,
    )

    print()
    print(
        "Input shape:",
        x.shape,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = EEGGCNGRU(
        input_features=10,
        gcn_hidden=64,
        gcn_embedding=32,
        gru_hidden=64,
        gru_layers=2,
        num_classes=2,
        dropout=0.3,
    ).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(
        x,
        edge_index,
    )

    print(
        "Output shape:",
        output.shape,
    )

    print(
        "Expected output shape:",
        "(4, 2)",
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    assert output.shape == (
        4,
        2,
    )

    print()
    print(
        "GCN + GRU model test completed successfully."
    )

    print()
    print("=" * 60)
    print("MODEL TEST PASSED!")
    print("=" * 60)