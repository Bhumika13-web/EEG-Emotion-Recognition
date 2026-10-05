from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_dataset import EEGSequenceDataset


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split):

    file_path = (
        PROCESSED_DIRECTORY
        / f"deap_{split}.npz"
    )

    print()
    print(
        f"Loading {split}:"
    )

    print(
        file_path
    )

    data = np.load(
        file_path
    )

    features = data["features"]

    valence = data["valence"]

    arousal = data["arousal"]

    subject_ids = data["subject_ids"]

    trial_ids = data["trial_ids"]

    return (
        features,
        valence,
        arousal,
        subject_ids,
        trial_ids,
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

(
    features,
    valence,
    arousal,
    subject_ids,
    trial_ids,
) = load_split(
    "train"
)


# ============================================================
# PRINT DATA INFORMATION
# ============================================================

print()
print("=" * 60)
print("TRAIN DATA")
print("=" * 60)

print(
    "Features:",
    features.shape,
)

print(
    "Valence:",
    valence.shape,
)

print(
    "Arousal:",
    arousal.shape,
)

print(
    "Subjects:",
    subject_ids.shape,
)

print(
    "Trials:",
    trial_ids.shape,
)


# ============================================================
# VERIFY SUBJECTS
# ============================================================

unique_subjects = np.unique(
    subject_ids
)

print()
print(
    "Unique training subjects:"
)

print(
    unique_subjects
)

print(
    "Number of subjects:",
    len(unique_subjects),
)


# ============================================================
# CREATE DATASET
# ============================================================

# We start with Valence classification.

dataset = EEGSequenceDataset(
    features=features,
    labels=valence,
)


print()
print(
    "Number of sequences:",
    len(dataset),
)


# ============================================================
# CREATE DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)


# ============================================================
# GET ONE BATCH
# ============================================================

x, y = next(
    iter(loader)
)


print()
print(
    "Batch X shape:",
    x.shape,
)

print(
    "Batch Y shape:",
    y.shape,
)

print(
    "X dtype:",
    x.dtype,
)

print(
    "Y dtype:",
    y.dtype,
)


# ============================================================
# CHECK GPU
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print(
    "Device:",
    device,
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# MOVE BATCH TO GPU
# ============================================================

x = x.to(device)

y = y.to(device)


print()
print(
    "GPU X shape:",
    x.shape,
)

print(
    "GPU Y shape:",
    y.shape,
)


# ============================================================
# VALIDATION
# ============================================================

assert features.shape == (
    960,
    30,
    32,
    10,
)

assert valence.shape == (
    960,
)

assert arousal.shape == (
    960,
)

assert len(
    unique_subjects
) == 24

assert len(dataset) == 960

assert x.shape == (
    4,
    30,
    32,
    10,
)

assert y.shape == (
    4,
)

assert x.dtype == torch.float32

assert y.dtype == torch.int64


# ============================================================
# SUCCESS
# ============================================================

print()
print("=" * 60)
print(
    "REAL DATASET + DATALOADER TEST PASSED!"
)
print("=" * 60)