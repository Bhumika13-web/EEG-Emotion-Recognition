import numpy as np
import networkx as nx


# Standard DEAP EEG channel order
EEG_CHANNELS = [
    "FP1", "AF3", "F3", "F7",
    "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7",
    "PO3", "O1", "OZ", "O2",
    "PO4", "P8", "P4", "CP6",
    "CP2", "C4", "T8", "FC6",
    "FC2", "F4", "F8", "AF4",
    "FP2", "FZ", "CZ", "PZ",
]


# Neighbor relationships between electrodes.
# Each pair represents a spatial connection.
EDGES = [
    # Frontal region
    ("FP1", "AF3"),
    ("FP1", "F7"),
    ("FP1", "F3"),
    ("FP2", "AF4"),
    ("FP2", "F8"),
    ("FP2", "F4"),

    ("AF3", "F3"),
    ("AF3", "F7"),
    ("AF4", "F4"),
    ("AF4", "F8"),

    ("F7", "F3"),
    ("F7", "FC5"),
    ("F3", "FC1"),
    ("F3", "F4"),
    ("F4", "FC2"),
    ("F4", "F8"),
    ("F8", "FC6"),

    # Frontal-central
    ("FC5", "FC1"),
    ("FC5", "C3"),
    ("FC1", "C3"),
    ("FC1", "CZ"),
    ("FC1", "FZ"),
    ("FC2", "C4"),
    ("FC2", "CZ"),
    ("FC2", "FZ"),
    ("FC6", "C4"),

    # Central
    ("C3", "T7"),
    ("C3", "C4"),
    ("C3", "CP1"),
    ("C3", "CP5"),
    ("CZ", "C3"),
    ("CZ", "C4"),
    ("CZ", "PZ"),
    ("C4", "T8"),
    ("C4", "CP2"),
    ("C4", "CP6"),

    # Temporal
    ("T7", "CP5"),
    ("T8", "CP6"),

    # Central-parietal
    ("CP5", "CP1"),
    ("CP5", "P7"),
    ("CP5", "P3"),
    ("CP1", "P3"),
    ("CP1", "PZ"),
    ("CP2", "P4"),
    ("CP2", "PZ"),
    ("CP6", "P4"),
    ("CP6", "P8"),

    # Parietal
    ("P7", "P3"),
    ("P7", "PO3"),
    ("P3", "PO3"),
    ("P3", "P4"),
    ("P3", "PZ"),
    ("PZ", "P4"),
    ("PZ", "PO4"),
    ("P4", "PO4"),
    ("P4", "P8"),

    # Occipital
    ("PO3", "O1"),
    ("PO3", "OZ"),
    ("O1", "OZ"),
    ("OZ", "O2"),
    ("O2", "PO4"),
    ("PO4", "P8"),

    # Midline
    ("FZ", "CZ"),
    ("CZ", "PZ"),
    ("PZ", "OZ"),
]


def create_electrode_graph():
    """
    Create a NetworkX graph representing
    spatial relationships between the 32 DEAP EEG electrodes.
    """

    graph = nx.Graph()

    # Add all 32 electrodes as nodes.
    graph.add_nodes_from(EEG_CHANNELS)

    # Add spatial edges.
    graph.add_edges_from(EDGES)

    return graph


def graph_to_adjacency(graph):
    """
    Convert the electrode graph to an adjacency matrix.

    Returns
    -------
    adjacency : np.ndarray
        Shape: (32, 32)
    """

    adjacency = nx.to_numpy_array(
        graph,
        nodelist=EEG_CHANNELS,
        dtype=np.float32,
    )

    return adjacency


def add_self_loops(adjacency):
    """
    Add self-connections to every electrode.
    """

    adjacency = adjacency.copy()

    np.fill_diagonal(adjacency, 1.0)

    return adjacency


def normalize_adjacency(adjacency):
    """
    Symmetric normalization:

        D^(-1/2) A D^(-1/2)

    """

    degree = adjacency.sum(axis=1)

    degree_inv_sqrt = np.zeros_like(degree)

    non_zero = degree > 0

    degree_inv_sqrt[non_zero] = (
        degree[non_zero] ** -0.5
    )

    degree_matrix = np.diag(degree_inv_sqrt)

    normalized = (
        degree_matrix
        @ adjacency
        @ degree_matrix
    )

    return normalized.astype(np.float32)


if __name__ == "__main__":

    graph = create_electrode_graph()

    adjacency = graph_to_adjacency(graph)

    adjacency = add_self_loops(adjacency)

    normalized_adjacency = normalize_adjacency(
        adjacency
    )

    print("EEG electrode graph")
    print("-------------------")
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("Adjacency shape:", adjacency.shape)
    print(
        "Normalized adjacency shape:",
        normalized_adjacency.shape,
    )
    print(
        "Number of EEG channels:",
        len(EEG_CHANNELS),
    )
    print("Graph created successfully.")