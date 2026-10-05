"""
SEED GCN Training
=================

Dataset:
    SEED Differential Entropy features

Input:
    (samples, 5, 62)

Where:
    5  = DE frequency bands
    62 = EEG electrodes

For GCN:
    62 graph nodes
    5 features per node

Classes:
    0 = Negative
    1 = Neutral
    2 = Positive

Subject-independent split:
    Train      = subjects 0-9
    Validation = subjects 10-11
    Test       = subjects 12-14

Required files:
    data/processed/seed_train.npz
    data/processed/seed_val.npz
    data/processed/seed_test.npz
    data/processed/seed_edge_index.pt

Output:
    models/seed_gcn_best.pth
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models"


TRAIN_FILE = (
    PROCESSED_DIR /
    "seed_train.npz"
)

VAL_FILE = (
    PROCESSED_DIR /
    "seed_val.npz"
)

TEST_FILE = (
    PROCESSED_DIR /
    "seed_test.npz"
)

EDGE_FILE = (
    PROCESSED_DIR /
    "seed_edge_index.pt"
)

BEST_MODEL_FILE = (
    MODEL_DIR /
    "seed_gcn_best.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_NODES = 62

NUM_FEATURES = 5

NUM_CLASSES = 3

HIDDEN_CHANNELS = 32

HIDDEN_CHANNELS_2 = 64

CLASSIFIER_HIDDEN = 32

DROPOUT = 0.30

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

EPOCHS = 30

BATCH_SIZE = 256

PATIENCE = 7

RANDOM_SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class SEEDDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        npz_file
    ):

        data = np.load(
            npz_file,
            allow_pickle=True
        )

        self.features = data[
            "features"
        ].astype(
            np.float32
        )

        self.labels = data[
            "labels"
        ].astype(
            np.int64
        )

        # Optional subject IDs
        if "subject_ids" in data.files:

            self.subject_ids = data[
                "subject_ids"
            ].astype(
                np.int64
            )

        else:

            self.subject_ids = None

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if self.features.ndim != 3:

            raise ValueError(
                "Expected feature shape "
                "(N, 5, 62), "
                f"got {self.features.shape}"
            )

        if self.features.shape[1] != NUM_FEATURES:

            raise ValueError(
                f"Expected {NUM_FEATURES} "
                f"features, got "
                f"{self.features.shape[1]}"
            )

        if self.features.shape[2] != NUM_NODES:

            raise ValueError(
                f"Expected {NUM_NODES} "
                f"electrodes, got "
                f"{self.features.shape[2]}"
            )

        if len(self.features) != len(
            self.labels
        ):

            raise ValueError(
                "Features and labels have "
                "different numbers of samples."
            )

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index
    ):

        x = torch.from_numpy(
            self.features[index]
        )

        y = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return x, y


# ============================================================
# GCN MODEL
# ============================================================

class BatchedSEEDGCN(
    nn.Module
):

    """
    SEED Graph Convolutional Network.

    Input:
        B × 62 × 5

    Output:
        B × 3
    """

    def __init__(
        self,
        num_features=5,
        hidden_channels=32,
        hidden_channels_2=64,
        num_classes=3,
        dropout=0.30
    ):

        super().__init__()

        # ----------------------------------------------------
        # GCN layer 1
        # ----------------------------------------------------

        self.conv1 = GCNConv(
            num_features,
            hidden_channels
        )

        # ----------------------------------------------------
        # GCN layer 2
        # ----------------------------------------------------

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels_2
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            hidden_channels_2,
            CLASSIFIER_HIDDEN
        )

        self.fc2 = nn.Linear(
            CLASSIFIER_HIDDEN,
            num_classes
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
        batch_size
    ):

        """
        x:
            (B * 62, 5)

        edge_index:
            batched graph

        batch_size:
            actual number of EEG samples
            in the current batch
        """

        # ----------------------------------------------------
        # GCN Layer 1
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(
            x
        )

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # ----------------------------------------------------
        # GCN Layer 2
        # ----------------------------------------------------

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(
            x
        )

        # ----------------------------------------------------
        # Validate number of graph nodes
        # ----------------------------------------------------

        expected_nodes = (
            batch_size *
            NUM_NODES
        )

        if x.size(0) != expected_nodes:

            raise RuntimeError(
                "\n"
                "GCN batching error\n"
                "------------------\n"
                f"Expected nodes: "
                f"{expected_nodes}\n"
                f"Received nodes: "
                f"{x.size(0)}\n"
                f"Batch size: "
                f"{batch_size}\n"
                f"Nodes per graph: "
                f"{NUM_NODES}\n"
            )

        # ----------------------------------------------------
        # Reshape
        #
        # B * 62 × 64
        #
        # →
        #
        # B × 62 × 64
        # ----------------------------------------------------

        x = x.reshape(
            batch_size,
            NUM_NODES,
            HIDDEN_CHANNELS_2
        )

        # ----------------------------------------------------
        # Global mean pooling
        #
        # B × 62 × 64
        #
        # →
        #
        # B × 64
        # ----------------------------------------------------

        x = x.mean(
            dim=1
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        x = self.fc1(
            x
        )

        x = F.relu(
            x
        )

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.fc2(
            x
        )

        return x


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    print()
    print("=" * 70)
    print("LOADING SEED DATA")
    print("=" * 70)

    print()
    print(
        f"Train file: {TRAIN_FILE}"
    )

    print(
        f"Val file:   {VAL_FILE}"
    )

    print(
        f"Test file:  {TEST_FILE}"
    )

    train_dataset = SEEDDataset(
        TRAIN_FILE
    )

    val_dataset = SEEDDataset(
        VAL_FILE
    )

    test_dataset = SEEDDataset(
        TEST_FILE
    )

    print()
    print(
        "Dataset shapes"
    )

    print(
        "-" * 70
    )

    print(
        f"Train: "
        f"{train_dataset.features.shape}"
    )

    print(
        f"Val:   "
        f"{val_dataset.features.shape}"
    )

    print(
        f"Test:  "
        f"{test_dataset.features.shape}"
    )

    # --------------------------------------------------------
    # Label distributions
    # --------------------------------------------------------

    print()

    print(
        "Train labels: "
        f"{np.bincount(train_dataset.labels, minlength=3)}"
    )

    print(
        "Val labels:   "
        f"{np.bincount(val_dataset.labels, minlength=3)}"
    )

    print(
        "Test labels:  "
        f"{np.bincount(test_dataset.labels, minlength=3)}"
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    print()
    print("=" * 70)
    print("LOADING SEED ELECTRODE GRAPH")
    print("=" * 70)

    if not EDGE_FILE.exists():

        raise FileNotFoundError(
            "\nGraph file not found:\n"
            f"{EDGE_FILE}\n\n"
            "Run first:\n"
            "python "
            "src\\graph\\seed_electrode_graph.py"
        )

    edge_index = torch.load(
        EDGE_FILE,
        map_location="cpu",
        weights_only=True
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if edge_index.ndim != 2:

        raise ValueError(
            "edge_index must have "
            "2 dimensions."
        )

    if edge_index.shape[0] != 2:

        raise ValueError(
            "edge_index must have "
            "shape (2, E)."
        )

    print()

    print(
        f"Edge index shape: "
        f"{tuple(edge_index.shape)}"
    )

    print(
        f"Number of graph edges: "
        f"{edge_index.shape[1]}"
    )

    print(
        "✓ Graph loaded successfully."
    )

    return edge_index


# ============================================================
# CREATE BATCHED GRAPH
# ============================================================

def create_batched_edge_index(
    edge_index,
    batch_size,
    num_nodes,
    device
):

    """
    Replicate the same 62-node EEG graph
    for every sample in a batch.

    Example:

        One graph:
            62 nodes

        Batch of 256:
            256 × 62 = 15,872 nodes
    """

    edge_index = edge_index.to(
        device
    )

    num_edges = (
        edge_index.shape[1]
    )

    # --------------------------------------------------------
    # Offsets for every graph
    #
    # Graph 0 → +0
    # Graph 1 → +62
    # Graph 2 → +124
    # ...
    # --------------------------------------------------------

    offsets = (
        torch.arange(
            batch_size,
            device=device
        )
        * num_nodes
    )

    offsets = offsets.view(
        batch_size,
        1,
        1
    )

    # --------------------------------------------------------
    # Duplicate edges
    # --------------------------------------------------------

    batched_edges = (
        edge_index.unsqueeze(0)
        + offsets
    )

    # Shape:
    #
    # batch × 2 × edges
    #
    # Convert to:
    #
    # 2 × (batch × edges)
    # --------------------------------------------------------

    batched_edges = (
        batched_edges
        .permute(
            1,
            0,
            2
        )
        .contiguous()
        .reshape(
            2,
            batch_size * num_edges
        )
    )

    return batched_edges


# ============================================================
# FORWARD BATCH
# ============================================================

def forward_batch(
    model,
    x,
    edge_index
):

    """
    Process one mini-batch.

    Input:
        x = (B, 5, 62)

    Convert:
        (B, 5, 62)
        ↓
        (B, 62, 5)
        ↓
        (B*62, 5)

    Output:
        (B, 3)
    """

    # --------------------------------------------------------
    # IMPORTANT:
    # Get actual batch size BEFORE reshaping.
    # --------------------------------------------------------

    batch_size = x.size(0)

    # --------------------------------------------------------
    # Convert:
    #
    # B × 5 × 62
    #
    # →
    #
    # B × 62 × 5
    # --------------------------------------------------------

    x = x.transpose(
        1,
        2
    ).contiguous()

    # --------------------------------------------------------
    # Flatten nodes
    #
    # B × 62 × 5
    #
    # →
    #
    # (B*62) × 5
    # --------------------------------------------------------

    x = x.reshape(
        batch_size * NUM_NODES,
        NUM_FEATURES
    )

    # --------------------------------------------------------
    # Create batched graph
    # --------------------------------------------------------

    batched_edge_index = (
        create_batched_edge_index(
            edge_index=edge_index,
            batch_size=batch_size,
            num_nodes=NUM_NODES,
            device=x.device
        )
    )

    # --------------------------------------------------------
    # Model
    #
    # IMPORTANT:
    # Explicitly send batch_size.
    # --------------------------------------------------------

    logits = model(
        x,
        batched_edge_index,
        batch_size
    )

    return logits


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    labels
):

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    ).astype(
        np.float32
    )

    print()
    print(
        "Class distribution"
    )

    print(
        "-" * 70
    )

    class_names = [
        "Negative",
        "Neutral",
        "Positive"
    ]

    for i in range(
        NUM_CLASSES
    ):

        print(
            f"{i} "
            f"({class_names[i]}): "
            f"{int(counts[i])}"
        )

    # --------------------------------------------------------
    # Balanced inverse-frequency weights
    # --------------------------------------------------------

    weights = (
        len(labels)
        /
        (
            NUM_CLASSES *
            counts
        )
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )

    print()

    print(
        "Class weights:",
        weights.detach()
        .cpu()
        .numpy()
    )

    return weights


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    edge_index
):

    model.train()

    total_loss = 0.0

    all_predictions = []

    all_labels = []

    for x, y in loader:

        # ----------------------------------------------------
        # Move to GPU
        # ----------------------------------------------------

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = forward_batch(
            model,
            x,
            edge_index
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            y
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Accumulate loss
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            * x.size(0)
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            y.detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    avg_loss = (
        total_loss
        /
        len(loader.dataset)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        np.array(
            all_labels
        ),
        np.array(
            all_predictions
        )
    )

    return (
        avg_loss,
        metrics
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    edge_index
):

    model.eval()

    total_loss = 0.0

    all_predictions = []

    all_labels = []

    for x, y in loader:

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = forward_batch(
            model,
            x,
            edge_index
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            y
        )

        total_loss += (
            loss.item()
            * x.size(0)
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
        )

        all_labels.extend(
            y.cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    avg_loss = (
        total_loss
        /
        len(loader.dataset)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    return (
        avg_loss,
        metrics,
        y_true,
        y_pred
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    y_true,
    y_pred
):

    from sklearn.metrics import (
        confusion_matrix
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2
        ]
    )

    print()
    print(
        "Confusion Matrix"
    )

    print(
        "-" * 70
    )

    print(
        "              Predicted"
    )

    print(
        "              Neg   Neu   Pos"
    )

    names = [
        "Actual Neg",
        "Actual Neu",
        "Actual Pos"
    ]

    for name, row in zip(
        names,
        cm
    ):

        print(
            f"{name:12s}"
            f"{row[0]:6d}"
            f"{row[1]:6d}"
            f"{row[2]:6d}"
        )

    return cm


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def print_classification_report(
    y_true,
    y_pred
):

    from sklearn.metrics import (
        classification_report
    )

    print()
    print(
        "Per-Class Performance"
    )

    print(
        "-" * 70
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2
        ],
        target_names=[
            "Negative",
            "Neutral",
            "Positive"
        ],
        digits=4,
        zero_division=0
    )

    print(
        report
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_info(
    model
):

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Model"
    )

    print(
        "-" * 70
    )

    print(
        model
    )

    print()

    print(
        f"Total parameters:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "SEED GCN TRAINING"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "CUDA: "
            f"{torch.version.cuda}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = load_datasets()

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    )

    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    )

    test_loader = (
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    )

    # ========================================================
    # LOAD GRAPH
    # ========================================================

    edge_index = load_graph()

    edge_index = edge_index.to(
        DEVICE
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            train_dataset.labels
        )
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = BatchedSEEDGCN(
        num_features=NUM_FEATURES,
        hidden_channels=HIDDEN_CHANNELS,
        hidden_channels_2=HIDDEN_CHANNELS_2,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(
        DEVICE
    )

    print_model_info(
        model
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = (
        nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )

    # ========================================================
    # MODEL DIRECTORY
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # BEST MODEL TRACKING
    # ========================================================

    best_val_f1 = -np.inf

    best_epoch = 0

    epochs_without_improvement = 0

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            train_loss,
            train_metrics
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            edge_index=edge_index
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_metrics,
            _,
            _
        ) = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            edge_index=edge_index
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_metrics[
                "macro_f1"
            ]
        )

        current_lr = (
            optimizer
            .param_groups[0]
            ["lr"]
        )

        # ----------------------------------------------------
        # Epoch output
        # ----------------------------------------------------

        print()

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS}"
        )

        print(
            f"LR: "
            f"{current_lr:.6f}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Acc: "
            f"{train_metrics['accuracy']:.4f} | "
            f"Train Bal Acc: "
            f"{train_metrics['balanced_accuracy']:.4f} | "
            f"Train Macro F1: "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Loss:   "
            f"{val_loss:.4f} | "
            f"Val Acc: "
            f"{val_metrics['accuracy']:.4f} | "
            f"Val Bal Acc: "
            f"{val_metrics['balanced_accuracy']:.4f} | "
            f"Val Macro F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        current_val_f1 = (
            val_metrics[
                "macro_f1"
            ]
        )

        if (
            current_val_f1
            > best_val_f1
        ):

            best_val_f1 = (
                current_val_f1
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_macro_f1":
                        best_val_f1,

                    "val_accuracy":
                        val_metrics[
                            "accuracy"
                        ],

                    "val_balanced_accuracy":
                        val_metrics[
                            "balanced_accuracy"
                        ],

                    "config": {

                        "num_nodes":
                            NUM_NODES,

                        "num_features":
                            NUM_FEATURES,

                        "num_classes":
                            NUM_CLASSES,

                        "hidden_channels":
                            HIDDEN_CHANNELS,

                        "hidden_channels_2":
                            HIDDEN_CHANNELS_2,

                        "dropout":
                            DROPOUT
                    }
                },

                BEST_MODEL_FILE
            )

            print(
                f"✓ Best model saved "
                f"(Val Macro F1: "
                f"{best_val_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()

            print(
                f"Early stopping at "
                f"epoch {epoch}."
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "LOADING BEST MODEL"
    )

    print(
        "=" * 70
    )

    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation "
        f"Macro F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    (
        val_loss,
        val_metrics,
        val_true,
        val_pred
    ) = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        edge_index=edge_index
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL VALIDATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Loss:               "
        f"{val_loss:.4f}"
    )

    print(
        f"Accuracy:           "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{val_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{val_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:        "
        f"{val_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Macro Precision:    "
        f"{val_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:       "
        f"{val_metrics['macro_recall']:.4f}"
    )

    print_classification_report(
        val_true,
        val_pred
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    (
        test_loss,
        test_metrics,
        test_true,
        test_pred
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        edge_index=edge_index
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Loss:               "
        f"{test_loss:.4f}"
    )

    print(
        f"Accuracy:           "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:        "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Macro Precision:    "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:       "
        f"{test_metrics['macro_recall']:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print_confusion_matrix(
        test_true,
        test_pred
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print_classification_report(
        test_true,
        test_pred
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "SEED GCN TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Input:"
    )

    print(
        "  62 EEG electrodes"
    )

    print(
        "  5 DE frequency-band features"
    )

    print()

    print(
        "Classes:"
    )

    print(
        "  0 = Negative"
    )

    print(
        "  1 = Neutral"
    )

    print(
        "  2 = Positive"
    )

    print()

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print()

    print(
        "Best model:"
    )

    print(
        f"  {BEST_MODEL_FILE}"
    )

    print()

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()