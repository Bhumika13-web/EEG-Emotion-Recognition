import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_dataset import EEGSequenceDataset


# --------------------------------------------------
# Create dummy EEG feature data
# --------------------------------------------------

# 1200 windows
# 32 EEG channels
# 10 features (5 DE + 5 PSD)

features = np.random.randn(
    1200,
    32,
    10,
).astype(np.float32)


# --------------------------------------------------
# Create trial IDs
# --------------------------------------------------

# 40 trials
# 30 windows per trial

trial_ids = np.repeat(
    np.arange(40),
    30,
)


# --------------------------------------------------
# Create binary labels
# --------------------------------------------------

labels = np.random.randint(
    0,
    2,
    size=1200,
)


# --------------------------------------------------
# Create Dataset
# --------------------------------------------------

dataset = EEGSequenceDataset(
    features=features,
    labels=labels,
    trial_ids=trial_ids,
    sequence_length=30,
)


print(
    "Number of sequences:",
    len(dataset),
)


# --------------------------------------------------
# Create DataLoader
# --------------------------------------------------

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)


# --------------------------------------------------
# Test one batch
# --------------------------------------------------

x, y = next(iter(loader))


print(
    "X shape:",
    x.shape,
)

print(
    "Y shape:",
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


# --------------------------------------------------
# Expected output
# --------------------------------------------------

print()
print("Expected:")
print("X shape: torch.Size([4, 30, 32, 10])")
print("Y shape: torch.Size([4])")


# --------------------------------------------------
# Basic validation
# --------------------------------------------------

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


print()
print(
    "Dataset and DataLoader test PASSED!"
)