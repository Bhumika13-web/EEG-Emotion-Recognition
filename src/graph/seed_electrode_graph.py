"""
SEED 62-Electrode Graph

SEED input:
    (samples, 5, 62)

For GCN:
    5 features per node
    62 electrode nodes
"""

from pathlib import Path

import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

GRAPH_FILE = OUTPUT_DIR / "seed_electrode_graph.npz"
EDGE_INDEX_FILE = OUTPUT_DIR / "seed_edge_index.pt"


# ============================================================
# EXACTLY 62 CHANNELS
# ============================================================

SEED_CHANNELS = [
    "FP1", "FPZ", "FP2",
    "AF3", "AF4",

    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",

    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",

    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",

    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",

    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",

    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",

    "O1", "OZ", "O2",

    "CB1", "CB2",
]


# ============================================================
# CHECK CHANNEL COUNT
# ============================================================

print()
print("=" * 70)
print("BUILDING SEED 62-ELECTRODE GRAPH")
print("=" * 70)

print()
print("Number of channels in code:", len(SEED_CHANNELS))

for i, channel in enumerate(SEED_CHANNELS):
    print(f"{i:02d}: {channel}")

assert len(SEED_CHANNELS) == 62, (
    f"ERROR: Channel list contains {len(SEED_CHANNELS)} channels, "
    f"but 62 are required."
)

print()
print(" Channel list contains exactly 62 electrodes.")


# ============================================================
# APPROXIMATE 2D POSITIONS
# ============================================================

SEED_POSITIONS = {

    "FP1": (-2.0, 4.0),
    "FPZ": (0.0, 4.2),
    "FP2": (2.0, 4.0),

    "AF3": (-1.5, 3.2),
    "AF4": (1.5, 3.2),

    "F7": (-4.0, 2.8),
    "F5": (-3.0, 2.8),
    "F3": (-2.0, 2.8),
    "F1": (-1.0, 2.8),
    "FZ": (0.0, 2.9),
    "F2": (1.0, 2.8),
    "F4": (2.0, 2.8),
    "F6": (3.0, 2.8),
    "F8": (4.0, 2.8),

    "FT7": (-4.2, 1.8),
    "FC5": (-3.0, 1.8),
    "FC3": (-2.0, 1.8),
    "FC1": (-1.0, 1.8),
    "FCZ": (0.0, 1.9),
    "FC2": (1.0, 1.8),
    "FC4": (2.0, 1.8),
    "FC6": (3.0, 1.8),
    "FT8": (4.2, 1.8),

    "T7": (-4.5, 0.5),
    "C5": (-3.5, 0.5),
    "C3": (-2.5, 0.5),
    "C1": (-1.2, 0.5),
    "CZ": (0.0, 0.5),
    "C2": (1.2, 0.5),
    "C4": (2.5, 0.5),
    "C6": (3.5, 0.5),
    "T8": (4.5, 0.5),

    "TP7": (-4.2, -0.6),
    "CP5": (-3.2, -0.6),
    "CP3": (-2.1, -0.6),
    "CP1": (-1.0, -0.6),
    "CPZ": (0.0, -0.6),
    "CP2": (1.0, -0.6),
    "CP4": (2.1, -0.6),
    "CP6": (3.2, -0.6),
    "TP8": (4.2, -0.6),

    "P7": (-4.0, -1.8),
    "P5": (-3.0, -1.8),
    "P3": (-2.0, -1.8),
    "P1": (-1.0, -1.8),
    "PZ": (0.0, -1.8),
    "P2": (1.0, -1.8),
    "P4": (2.0, -1.8),
    "P6": (3.0, -1.8),
    "P8": (4.0, -1.8),

    "PO7": (-3.5, -2.8),
    "PO5": (-2.5, -2.8),
    "PO3": (-1.5, -2.8),
    "POZ": (0.0, -2.9),
    "PO4": (1.5, -2.8),
    "PO6": (2.5, -2.8),
    "PO8": (3.5, -2.8),

    "O1": (-1.8, -3.8),
    "OZ": (0.0, -4.0),
    "O2": (1.8, -3.8),

    "CB1": (-2.0, -4.8),
    "CB2": (2.0, -4.8),
}


# ============================================================
# CHECK POSITIONS
# ============================================================

print()
print("Checking electrode positions...")

missing = [
    ch for ch in SEED_CHANNELS
    if ch not in SEED_POSITIONS
]

if missing:
    print("Missing positions:", missing)
    raise ValueError("Some electrodes do not have coordinates.")

assert len(SEED_POSITIONS) == 62

print(" All 62 electrodes have positions.")


# ============================================================
# BUILD GRAPH
# ============================================================

NUM_NODES = 62

DISTANCE_THRESHOLD = 2.1

coordinates = np.array(
    [
        SEED_POSITIONS[ch]
        for ch in SEED_CHANNELS
    ],
    dtype=np.float32,
)


# ------------------------------------------------------------
# Raw adjacency
# ------------------------------------------------------------

adjacency = np.zeros(
    (NUM_NODES, NUM_NODES),
    dtype=np.float32,
)


for i in range(NUM_NODES):

    for j in range(i + 1, NUM_NODES):

        distance = np.linalg.norm(
            coordinates[i] - coordinates[j]
        )

        if distance <= DISTANCE_THRESHOLD:

            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0


# ------------------------------------------------------------
# Add self-loops
# ------------------------------------------------------------

adjacency_with_self_loops = (
    adjacency
    + np.eye(
        NUM_NODES,
        dtype=np.float32
    )
)


# ============================================================
# NORMALIZED ADJACENCY
# ============================================================

degree = adjacency_with_self_loops.sum(axis=1)

degree_inv_sqrt = np.zeros_like(degree)

valid = degree > 0

degree_inv_sqrt[valid] = (
    degree[valid] ** -0.5
)

normalized_adjacency = (
    degree_inv_sqrt[:, None]
    * adjacency_with_self_loops
    * degree_inv_sqrt[None, :]
)


# ============================================================
# EDGE INDEX
# ============================================================

rows, cols = np.nonzero(
    adjacency_with_self_loops
)

edge_index = torch.tensor(
    np.vstack([rows, cols]),
    dtype=torch.long,
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("GRAPH VALIDATION")
print("=" * 70)

assert adjacency.shape == (62, 62)

assert normalized_adjacency.shape == (62, 62)

assert coordinates.shape == (62, 2)

assert edge_index.shape[0] == 2

assert np.allclose(
    adjacency,
    adjacency.T
)

assert np.all(
    np.diag(adjacency_with_self_loops) == 1
)


print(" Nodes:              62")
print(
    f" Adjacency shape:    {adjacency.shape}"
)
print(
    f" Normalized shape:   {normalized_adjacency.shape}"
)
print(
    f" Coordinates:        {coordinates.shape}"
)
print(
    f" Edge index shape:   {tuple(edge_index.shape)}"
)

print(
    f" Undirected edges:   "
    f"{int(adjacency.sum() / 2)}"
)

print(
    f" Total graph edges:  "
    f"{edge_index.shape[1]}"
)

print(" Graph is symmetric")
print(" Self-loops present")


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


np.savez(
    GRAPH_FILE,

    adjacency=adjacency,

    normalized_adjacency=normalized_adjacency,

    coordinates=coordinates,

    channel_names=np.array(
        SEED_CHANNELS
    ),
)


torch.save(
    edge_index,
    EDGE_INDEX_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("GRAPH SAVED SUCCESSFULLY")
print("=" * 70)

print()
print("Saved files:")

print(
    f"1. {GRAPH_FILE}"
)

print(
    f"2. {EDGE_INDEX_FILE}"
)

print()
print("SEED graph is ready for GCN.")
print("=" * 70)