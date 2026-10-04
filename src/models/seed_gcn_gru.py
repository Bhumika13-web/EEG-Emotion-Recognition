"""
SEED GCN + GRU
==============

Spatial-Temporal EEG Emotion Recognition

Input:
    (batch, sequence_length, 5, 62)

    5  = frequency bands
    62 = EEG electrodes

Architecture:
    EEG features
        ↓
    GCN
        ↓
    Spatial representation
        ↓
    GRU
        ↓
    Emotion classifier

Classes:
    0 = Negative
    1 = Neutral
    2 = Positive

Subject split:
    Train      : Subjects 0-9
    Validation : Subjects 10-11
    Test       : Subjects 12-14
"""

from pathlib import Path
import copy
import random

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import GCNConv

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_FILE = DATA_DIR / "seed_train.npz"
VAL_FILE = DATA_DIR / "seed_val.npz"
TEST_FILE = DATA_DIR / "seed_test.npz"

GRAPH_FILE = (
    GRAPH_DIR /
    "seed_edge_index.pt"
)

CHECKPOINT_FILE = (
    MODEL_DIR /
    "seed_gcn_gru_best.pth"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SEQ_LEN = 10

STRIDE = 10

BATCH_SIZE = 64

NUM_EPOCHS = 30

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

EARLY_STOPPING_PATIENCE = 7

NUM_CLASSES = 3

NUM_BANDS = 5

NUM_NODES = 62

GCN_HIDDEN = 32

GCN_OUTPUT = 64

GRU_HIDDEN = 64

GRU_LAYERS = 1


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)
print("SEED GCN + GRU")
print("=" * 80)

print(
    f"\nDevice: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# DATASET
# ============================================================

class SEEDSequenceDataset(Dataset):

    def __init__(
        self,
        data_file,
        seq_len=20,
        stride=20,
    ):

        loaded = np.load(
            data_file
        )

        # ----------------------------------------------------
        # Load only fields that actually exist
        # ----------------------------------------------------

        self.features = loaded[
            "features"
        ].astype(
            np.float32
        )

        self.labels = loaded[
            "labels"
        ].astype(
            np.int64
        )

        self.subject_ids = loaded[
            "subject_ids"
        ].astype(
            np.int64
        )

        print(
            f"\nLoaded: {data_file.name}"
        )

        print(
            f"Features: {self.features.shape}"
        )

        print(
            f"Labels: {self.labels.shape}"
        )

        print(
            f"Subjects: {self.subject_ids.shape}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if len(self.features) != len(
            self.labels
        ):

            raise ValueError(
                "Features and labels have "
                "different lengths."
            )

        if len(self.features) != len(
            self.subject_ids
        ):

            raise ValueError(
                "Features and subject_ids "
                "have different lengths."
            )

        # ----------------------------------------------------
        # Build temporal sequences
        # ----------------------------------------------------

        self.sequences = []

        self._build_sequences(
            seq_len,
            stride
        )

    # ========================================================
    # BUILD SEQUENCES
    # ========================================================

    def _build_sequences(
        self,
        seq_len,
        stride,
    ):

        """
        Construct sequences ONLY inside:

            1. one subject
            2. one emotion run

        Therefore a sequence never crosses:

            Subject A → Subject B

        or:

            Negative → Neutral
            Neutral → Positive
            etc.

        This is important because the GRU should receive
        a coherent temporal sequence belonging to one
        emotion segment.
        """

        total_samples = len(
            self.features
        )

        global_start = 0

        # ----------------------------------------------------
        # Process subject by subject
        # ----------------------------------------------------

        while global_start < total_samples:

            current_subject = (
                self.subject_ids[
                    global_start
                ]
            )

            # ------------------------------------------------
            # Find end of current subject
            # ------------------------------------------------

            subject_end = global_start

            while (
                subject_end < total_samples
                and
                self.subject_ids[
                    subject_end
                ]
                ==
                current_subject
            ):

                subject_end += 1

            # ------------------------------------------------
            # Process emotion runs inside subject
            # ------------------------------------------------

            run_start = global_start

            while run_start < subject_end:

                current_label = (
                    self.labels[
                        run_start
                    ]
                )

                # --------------------------------------------
                # Find end of emotion run
                # --------------------------------------------

                run_end = run_start

                while (
                    run_end < subject_end
                    and
                    self.labels[
                        run_end
                    ]
                    ==
                    current_label
                ):

                    run_end += 1

                run_length = (
                    run_end -
                    run_start
                )

                # --------------------------------------------
                # Create sequences inside this run
                # --------------------------------------------

                if run_length >= seq_len:

                    sequence_start = run_start

                    while (
                        sequence_start
                        + seq_len
                        <= run_end
                    ):

                        sequence_end = (
                            sequence_start
                            + seq_len
                        )

                        self.sequences.append(
                            (
                                sequence_start,
                                sequence_end,
                                int(
                                    current_label
                                ),
                                int(
                                    current_subject
                                ),
                            )
                        )

                        sequence_start += stride

                # --------------------------------------------
                # Move to next emotion run
                # --------------------------------------------

                run_start = run_end

            # ------------------------------------------------
            # Move to next subject
            # ------------------------------------------------

            global_start = subject_end

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.sequences
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        (
            start,
            end,
            label,
            subject
        ) = self.sequences[
            index
        ]

        x = self.features[
            start:end
        ]

        y = label

        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),

            torch.tensor(
                y,
                dtype=torch.long
            ),
        )


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n" + "=" * 80)
print("LOADING SEQUENCE DATA")
print("=" * 80)


train_dataset = SEEDSequenceDataset(
    TRAIN_FILE,
    SEQ_LEN,
    STRIDE,
)


val_dataset = SEEDSequenceDataset(
    VAL_FILE,
    SEQ_LEN,
    STRIDE,
)


test_dataset = SEEDSequenceDataset(
    TEST_FILE,
    SEQ_LEN,
    STRIDE,
)


print("\n" + "=" * 80)
print("SEQUENCE COUNTS")
print("=" * 80)

print(
    f"\nTrain sequences: "
    f"{len(train_dataset)}"
)

print(
    f"Validation sequences: "
    f"{len(val_dataset)}"
)

print(
    f"Test sequences: "
    f"{len(test_dataset)}"
)


if len(train_dataset) == 0:

    raise RuntimeError(
        "No training sequences were created."
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# BATCH CHECK
# ============================================================

sample_x, sample_y = next(
    iter(train_loader)
)


print("\n" + "=" * 80)
print("BATCH SHAPE CHECK")
print("=" * 80)

print(
    f"\nInput batch shape: "
    f"{sample_x.shape}"
)

print(
    f"Label batch shape: "
    f"{sample_y.shape}"
)

print(
    "\nExpected input:"
)

print(
    "(batch, sequence_length, 5, 62)"
)


# ============================================================
# LOAD ELECTRODE GRAPH
# ============================================================

print("\n" + "=" * 80)
print("LOADING EEG GRAPH")
print("=" * 80)


edge_index = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=True,
)

edge_index = edge_index.long()


print(
    f"\nEdge index shape: "
    f"{edge_index.shape}"
)

print(
    f"Number of graph edges: "
    f"{edge_index.shape[1]}"
)


# ============================================================
# CREATE BATCHED GRAPH
# ============================================================

def create_batched_edge_index(
    edge_index,
    batch_size,
    num_nodes,
):

    """
    Replicate the same 62-node electrode graph
    for every graph in the batch.
    """

    edge_index = edge_index.to(
        DEVICE
    )

    num_edges = (
        edge_index.shape[1]
    )

    offsets = (
        torch.arange(
            batch_size,
            device=DEVICE,
        )
        * num_nodes
    )

    offsets = offsets.view(
        batch_size,
        1,
        1,
    )

    edges = (
        edge_index
        .unsqueeze(0)
        .repeat(
            batch_size,
            1,
            1,
        )
    )

    edges = (
        edges
        + offsets
    )

    edges = edges.permute(
        1,
        0,
        2,
    )

    edges = edges.reshape(
        2,
        batch_size * num_edges,
    )

    return edges


# ============================================================
# MODEL
# ============================================================

class SEEDGCNGRU(
    nn.Module
):

    def __init__(
        self,
        num_bands=5,
        gcn_hidden=32,
        gcn_output=64,
        gru_hidden=64,
        gru_layers=1,
        num_classes=3,
        dropout=0.30,
    ):

        super().__init__()

        # ----------------------------------------------------
        # GCN
        # ----------------------------------------------------

        self.gcn1 = GCNConv(
            num_bands,
            gcn_hidden,
        )

        self.gcn2 = GCNConv(
            gcn_hidden,
            gcn_output,
        )

        self.gcn_dropout = (
            nn.Dropout(dropout)
        )

        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=gcn_output,
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
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                gru_hidden,
                32,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                num_classes,
            ),
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        edge_index,
    ):

        """
        Input:

            B × T × 5 × 62

        B = batch size
        T = sequence length
        5 = frequency bands
        62 = electrodes
        """

        B, T, F, N = x.shape

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if F != NUM_BANDS:

            raise ValueError(
                f"Expected {NUM_BANDS} "
                f"frequency bands, "
                f"received {F}."
            )

        if N != NUM_NODES:

            raise ValueError(
                f"Expected {NUM_NODES} "
                f"electrodes, "
                f"received {N}."
            )

        # ----------------------------------------------------
        # Rearrange:
        #
        # B × T × F × N
        #
        # →
        #
        # B × T × N × F
        # ----------------------------------------------------

        x = x.permute(
            0,
            1,
            3,
            2,
        )

        # ----------------------------------------------------
        # Flatten B and T
        #
        # B × T × N × F
        #
        # →
        #
        # B*T*N × F
        # ----------------------------------------------------

        x = x.reshape(
            B * T * N,
            F,
        )

        # ----------------------------------------------------
        # Create graph for every temporal snapshot
        # ----------------------------------------------------

        batched_edges = (
            create_batched_edge_index(
                edge_index,
                B * T,
                N,
            )
        )

        # ----------------------------------------------------
        # GCN Layer 1
        # ----------------------------------------------------

        x = self.gcn1(
            x,
            batched_edges,
        )

        x = torch.relu(x)

        x = self.gcn_dropout(x)

        # ----------------------------------------------------
        # GCN Layer 2
        # ----------------------------------------------------

        x = self.gcn2(
            x,
            batched_edges,
        )

        x = torch.relu(x)

        # ----------------------------------------------------
        # Restore node dimension
        #
        # B*T*N × 64
        #
        # →
        #
        # B*T × N × 64
        # ----------------------------------------------------

        x = x.reshape(
            B * T,
            N,
            GCN_OUTPUT,
        )

        # ----------------------------------------------------
        # Spatial mean pooling
        #
        # B*T × N × 64
        #
        # →
        #
        # B*T × 64
        # ----------------------------------------------------

        x = x.mean(
            dim=1
        )

        # ----------------------------------------------------
        # Restore temporal dimension
        #
        # B*T × 64
        #
        # →
        #
        # B × T × 64
        # ----------------------------------------------------

        x = x.reshape(
            B,
            T,
            GCN_OUTPUT,
        )

        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        gru_output, _ = self.gru(
            x
        )

        # ----------------------------------------------------
        # Take final temporal state
        # ----------------------------------------------------

        temporal_embedding = (
            gru_output[
                :,
                -1,
                :
            ]
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            temporal_embedding
        )

        return logits


# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 80)
print("MODEL")
print("=" * 80)


model = SEEDGCNGRU(
    num_bands=NUM_BANDS,
    gcn_hidden=GCN_HIDDEN,
    gcn_output=GCN_OUTPUT,
    gru_hidden=GRU_HIDDEN,
    gru_layers=GRU_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(
    DEVICE
)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(model)

print(
    f"\nTotal parameters: "
    f"{total_params:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_params:,}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 80)
print("CLASS WEIGHTS")
print("=" * 80)


train_labels = np.array(
    [
        sequence[2]
        for sequence
        in train_dataset.sequences
    ]
)


class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES,
)


print(
    f"\nTraining sequence labels: "
    f"{class_counts}"
)


class_weights = (
    len(train_labels)
    /
    (
        NUM_CLASSES
        *
        class_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)


print(
    "Class weights: "
    f"{class_weights.cpu().numpy()}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision = (
        precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
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

def train_one_epoch():

    model.train()

    total_loss = 0.0

    all_true = []

    all_pred = []

    for x, y in train_loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            x,
            edge_index,
        )

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * x.size(0)
        )

        predictions = (
            torch.argmax(
                logits,
                dim=1,
            )
        )

        all_true.extend(
            y.detach()
            .cpu()
            .numpy()
        )

        all_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        total_loss
        /
        len(train_dataset)
    )

    metrics = calculate_metrics(
        all_true,
        all_pred,
    )

    return (
        epoch_loss,
        metrics,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    loader,
    dataset,
):

    model.eval()

    total_loss = 0.0

    all_true = []

    all_pred = []

    for x, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            x,
            edge_index,
        )

        loss = criterion(
            logits,
            y,
        )

        total_loss += (
            loss.item()
            * x.size(0)
        )

        predictions = (
            torch.argmax(
                logits,
                dim=1,
            )
        )

        all_true.extend(
            y.cpu().numpy()
        )

        all_pred.extend(
            predictions.cpu().numpy()
        )

    epoch_loss = (
        total_loss
        /
        len(dataset)
    )

    metrics = calculate_metrics(
        all_true,
        all_pred,
    )

    return (
        epoch_loss,
        metrics,
        np.array(all_true),
        np.array(all_pred),
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)


best_val_f1 = -float("inf")

best_epoch = 0

patience_counter = 0

best_state = None


for epoch in range(
    1,
    NUM_EPOCHS + 1,
):

    train_loss, train_metrics = (
        train_one_epoch()
    )

    (
        val_loss,
        val_metrics,
        _,
        _,
    ) = evaluate(
        val_loader,
        val_dataset,
    )

    scheduler.step(
        val_metrics[
            "macro_f1"
        ]
    )

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    print(
        f"\nEpoch "
        f"{epoch:02d}/"
        f"{NUM_EPOCHS}"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.6f}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Acc: "
        f"{train_metrics['accuracy']:.4f} | "
        f"Bal Acc: "
        f"{train_metrics['balanced_accuracy']:.4f} | "
        f"Macro F1: "
        f"{train_metrics['macro_f1']:.4f}"
    )

    print(
        f"Val Loss:   "
        f"{val_loss:.4f} | "
        f"Acc: "
        f"{val_metrics['accuracy']:.4f} | "
        f"Bal Acc: "
        f"{val_metrics['balanced_accuracy']:.4f} | "
        f"Macro F1: "
        f"{val_metrics['macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if (
        val_metrics["macro_f1"]
        >
        best_val_f1
    ):

        best_val_f1 = (
            val_metrics[
                "macro_f1"
            ]
        )

        best_epoch = epoch

        patience_counter = 0

        best_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_macro_f1":
                    best_val_f1,

                "config": {
                    "seq_len":
                        SEQ_LEN,

                    "stride":
                        STRIDE,

                    "gcn_hidden":
                        GCN_HIDDEN,

                    "gcn_output":
                        GCN_OUTPUT,

                    "gru_hidden":
                        GRU_HIDDEN,

                    "gru_layers":
                        GRU_LAYERS,
                },
            },
            CHECKPOINT_FILE,
        )

        print(
            "✓ Best model saved."
        )

    else:

        patience_counter += 1

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        patience_counter
        >=
        EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

print("\n" + "=" * 80)
print("RESTORING BEST MODEL")
print("=" * 80)


if best_state is not None:

    model.load_state_dict(
        best_state
    )

else:

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


print(
    f"\nBest epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation Macro F1: "
    f"{best_val_f1:.4f}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION RESULTS")
print("=" * 80)


(
    val_loss,
    val_metrics,
    val_true,
    val_pred,
) = evaluate(
    val_loader,
    val_dataset,
)


print(
    f"\nLoss: "
    f"{val_loss:.4f}"
)

print(
    f"Accuracy: "
    f"{val_metrics['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{val_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Macro F1: "
    f"{val_metrics['macro_f1']:.4f}"
)

print(
    f"Weighted F1: "
    f"{val_metrics['weighted_f1']:.4f}"
)

print(
    f"Macro Precision: "
    f"{val_metrics['macro_precision']:.4f}"
)

print(
    f"Macro Recall: "
    f"{val_metrics['macro_recall']:.4f}"
)


print(
    "\nValidation Confusion Matrix:"
)

print(
    confusion_matrix(
        val_true,
        val_pred,
    )
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)


(
    test_loss,
    test_metrics,
    test_true,
    test_pred,
) = evaluate(
    test_loader,
    test_dataset,
)


print(
    f"\nLoss: "
    f"{test_loss:.4f}"
)

print(
    f"Accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Macro F1: "
    f"{test_metrics['macro_f1']:.4f}"
)

print(
    f"Weighted F1: "
    f"{test_metrics['weighted_f1']:.4f}"
)

print(
    f"Macro Precision: "
    f"{test_metrics['macro_precision']:.4f}"
)

print(
    f"Macro Recall: "
    f"{test_metrics['macro_recall']:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

test_cm = confusion_matrix(
    test_true,
    test_pred,
)


print(
    "\nTest Confusion Matrix:"
)

print(
    "              Predicted"
)

print(
    "              Neg   Neu   Pos"
)

print(
    f"Actual Neg    "
    f"{test_cm[0, 0]:4d} "
    f"{test_cm[0, 1]:5d} "
    f"{test_cm[0, 2]:5d}"
)

print(
    f"Actual Neu    "
    f"{test_cm[1, 0]:4d} "
    f"{test_cm[1, 1]:5d} "
    f"{test_cm[1, 2]:5d}"
)

print(
    f"Actual Pos    "
    f"{test_cm[2, 0]:4d} "
    f"{test_cm[2, 1]:5d} "
    f"{test_cm[2, 2]:5d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        test_true,
        test_pred,

        target_names=[
            "Negative",
            "Neutral",
            "Positive",
        ],

        digits=4,

        zero_division=0,
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SEED GCN + GRU COMPLETE")
print("=" * 80)


print(
    f"\nBest epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation Macro F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"\nTest Accuracy: "
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

print(
    "\nCheckpoint:"
)

print(
    CHECKPOINT_FILE
)

print(
    "\n✓ Training and evaluation finished."
)