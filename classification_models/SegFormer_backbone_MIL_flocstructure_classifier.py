"""
SegFormer-encoder MIL (multiple-instance learning) classifier
----------------------------------------------------------------
Instead of training per-image and aggregating predictions afterward, this
groups every experiment's images into a "bag", encodes each image with the
(frozen) SegFormer MiT encoder, mean-pools the per-image embeddings into one
experiment-level feature vector, and classifies that directly.

Only mean pooling is implemented (cfg.classifier_mil_arch) -- other pooling
types (max/attention/etc.) can be added as alternative branches in
ExperimentMILClassifier.aggregate() later without touching the rest of the
model or training loop.

Train/test assignment comes from the authoritative fixed split workbook
(cfg.split_workbook_path), grouped by cfg.split_group_col (experiment_id).

"""

from config.config import cfg  # must be imported first -- sets CPU thread env vars

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# CONFIG -- edit these
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    seg_checkpoint_path: str = cfg.seg_checkpoint_path  
    split_workbook_path: str = cfg.split_workbook_path
    match_table_path : str = cfg.match_table_path
    split_train_sheet: str = cfg.split_train_sheet
    split_test_sheet: str = cfg.split_test_sheet
    split_merge_col: str = cfg.split_merge_col
    split_group_col: str = cfg.split_group_col
    norm_mean: int = cfg.norm_mean
    norm_std: int = cfg.norm_std
    image_size: int = cfg.tile_size         # resize target
    seed: int = cfg.seed
    gpu_id: int = cfg.gpu_id   
    num_threads: int = cfg.num_threads

    freeze_encoder: bool = True    # False = fine-tune everything end-to-end
    target_col : str = "Structuur (STRU_VMF_2)"
    classifier_class_names = ['Diffuus', 'Compact']
    classifier_mil_arch: str = "mean"        # pooling method; mean, max, attention
    classifier_mil_attention_hidden_dim: int = 128
    classifier_mil_chunk_size: int = 4       # images encoded per forward-pass chunk (memory management)
    classifier_mil_batch_size: int = 4       # experiments per gradient-accumulation step (bag sizes vary, so this isn't a normal tensor batch)
    classifier_mil_epochs: int = 30
    classifier_num_workers: int = 2
    classifier_mil_lr: float = 1e-4
    classifier_mil_weight_decay: float = 1e-5
    classifier_mil_hidden_dim: int = 256
    classifier_mil_dropout: float = 0.2
    segformer_mil_classifier_output_dir: str =  cfg.output_dir / "classification_models" / "segformer_backbone_flocstructure_mil_classifier_MEANpool_LR1e4"
            


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# MIL pooling modules
# --------------------------------------------------------------------------- #

class MeanPooling(nn.Module):
    """Simple mean pooling across instances."""

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.mean(dim=0)

class MaxPooling(nn.Module):
    """Element-wise max pooling across instances."""

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.max(dim=0).values

class AttentionPooling(nn.Module):
    """
    Gated attention MIL pooling.Learns one attention weight per image in the bag and computes
    a weighted sum of the image embeddings.

    """

    def __init__(self, input_dim: int, attention_hidden_dim: int):
        super().__init__()

        # Gated attention: a_i = softmax(w^T [tanh(V h_i) * sigmoid(U h_i)])
        # where h_i is the embedding of image i.

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Sigmoid(),
        )

        self.attention_weights = nn.Linear(
            attention_hidden_dim,
            1,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:

        # (N, attention_hidden_dim)
        v = self.attention_V(embeddings)

        # (N, attention_hidden_dim)
        u = self.attention_U(embeddings)

        # Gated attention (N, attention_hidden_dim)
        gated = v * u

        # One attention score per image (N, 1)
        scores = self.attention_weights(gated)

        # Normalize across images in this experiment (N, 1)
        attention = torch.softmax(scores, dim=0)

        # Weighted sum of image embeddings (D,)
        pooled = torch.sum(attention * embeddings, dim=0)

        return pooled
    
class ExperimentMILClassifier(nn.Module):
    """
    Multiple-instance learning classifier: encodes every image in an
    experiment's bag, mean-pools the per-image embeddings into one
    experiment-level feature vector, then classifies.
    """

    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int,
                 dropout: float, freeze_encoder: bool, chunk_size: int, 
                 mil_arch: str = "mean",  attention_hidden_dim: int = 128,):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.chunk_size = chunk_size
        self.mil_arch = mil_arch.lower()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        feat_channels = self.encoder.out_channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)

        if self.mil_arch == "mean":
            self.aggregate = MeanPooling()
        elif self.mil_arch == "max":
            self.aggregate = MaxPooling()
        elif self.mil_arch == "attention":
            self.aggregate = AttentionPooling(input_dim=feat_channels, attention_hidden_dim=attention_hidden_dim)
        else:
            raise ValueError( f"Unknown MIL pooling architecture: '{mil_arch}'. "
                f"Choose from: 'mean', 'max', 'attention'." )
        
        self.head = nn.Sequential(
            nn.LayerNorm(feat_channels),
            nn.Dropout(dropout),
            nn.Linear(feat_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _encode_chunks(self, images: torch.Tensor) -> torch.Tensor:
        """images: (n_images, C, H, W) -> (n_images, feat_channels)."""
        embeddings = []
        for chunk in images.split(self.chunk_size):
            feats = self.encoder(chunk)
            last = feats[-1]
            pooled = self.pool(last).flatten(1)
            embeddings.append(pooled)
        return torch.cat(embeddings, dim=0)

    def encode_bag(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                return self._encode_chunks(images)
        return self._encode_chunks(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.encode_bag(images)
        pooled = self.aggregate(embeddings)
        return self.head(pooled)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self


def load_pretrained_encoder(cfg, device: torch.device) -> nn.Module:
    seg_model = torch.load(cfg.seg_checkpoint_path, map_location=device)

    if not isinstance(seg_model, nn.Module):
        raise TypeError(
            f"Expected a whole nn.Module at {cfg.seg_checkpoint_path}, but got a "
            f"{type(seg_model)}. If this checkpoint is actually a state_dict, load it "
            f"into an smp.Segformer(...) instance first, then pass model.encoder here."
        )
    if not hasattr(seg_model, "encoder"):
        raise AttributeError(
            "Loaded model has no `.encoder` attribute -- check that this checkpoint "
            "is a segmentation_models_pytorch model (e.g. smp.Segformer(...))."
        )

    encoder = seg_model.encoder
    del seg_model
    return encoder


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def _resolve_image_path(row, cfg) -> Path:
    """Prefer resolved_image_path (server-local); fall back to image_path."""
    resolved = Path(row["resolved_image_path" ])
    if resolved.is_file():
        return resolved
    return Path(row["image_path"])

def build_experiment_groups(df: pd.DataFrame, cfg) -> list:
    """
    Groups a (label-encoded) split dataframe by experiment_id into bags:
    [{"experiment_id": ..., "label": int, "paths": [Path, ...]}, ...]

    Drops individual missing images (keeping the rest of the bag), and drops
    an entire experiment only if NONE of its images are found on disk.
    """
    groups = []
    n_dropped_images = 0

    for exp_id, group_df in df.groupby("experiment_id", sort=True):
        labels = group_df["target_encoded"].unique()
        if len(labels) != 1:
            raise ValueError(f"Inconsistent label within experiment {exp_id}: {labels}")

        resolved_paths = group_df.apply(lambda row: _resolve_image_path(row, cfg), axis=1)
        exists_mask = resolved_paths.apply(lambda p: p.is_file())
        n_dropped_images += int((~exists_mask).sum())
        valid_paths = resolved_paths[exists_mask].tolist()

        if not valid_paths:
            print(f"Warning: experiment {exp_id} has no images on disk, skipping entirely")
            continue

        groups.append({
            "experiment_id": str(exp_id),
            "label": int(labels[0]),
            "paths": valid_paths,
        })

    if n_dropped_images > 0:
        print(f"Warning: dropped {n_dropped_images} individual missing image(s) across all experiments")

    return groups

def normalize_image_path(path: str) -> str:
    """Normalize a path so it can be used as a merge key across sheets."""
    path = str(path).replace("\\", "/")
    marker = "Aquafin_data_cleaned/"  
    if marker in path:
        return path[path.index(marker):]
    return path

def load_fixed_split_groups(cfg):
    match_table_df = pd.read_excel(cfg.match_table_path)
    train_split = pd.read_excel(cfg.split_workbook_path, sheet_name=cfg.split_train_sheet)
    test_split = pd.read_excel(cfg.split_workbook_path, sheet_name=cfg.split_test_sheet)

    assert set(train_split[cfg.split_group_col]).isdisjoint(test_split[cfg.split_group_col]), \
        "train/test experiment_id sets overlap -- split workbook is not group-disjoint"

    # Normalize paths before merging
    match_table_df["merge_image_path"] = match_table_df[cfg.split_merge_col].apply(normalize_image_path)
    train_split["merge_image_path"] = train_split[cfg.split_merge_col].apply(normalize_image_path)
    test_split["merge_image_path"] = test_split[cfg.split_merge_col].apply(normalize_image_path)

    join_cols = ["merge_image_path", cfg.target_col]

    train_split = train_split.merge(match_table_df[join_cols], on="merge_image_path", how="inner")
    test_split = test_split.merge(match_table_df[join_cols], on="merge_image_path", how="inner")

    print(f"Train: matched {len(train_split)} rows to match_table_df "
          f"({train_split[cfg.split_group_col].nunique()} experiments)")
    print(f"Test:  matched {len(test_split)} rows to match_table_df "
          f"({test_split[cfg.split_group_col].nunique()} experiments)")

    label_to_id = {label: index for index, label in enumerate(cfg.classifier_class_names)}
    for df in (train_split, test_split):
        df["target_encoded"] = df[cfg.target_col].map(label_to_id)
        n_unmapped = df["target_encoded"].isna().sum()
        if n_unmapped > 0:
            print(f"Warning: {n_unmapped} rows have a label not in {cfg.classifier_class_names}, dropping")
        df.dropna(subset=["target_encoded"], inplace=True)
        df["target_encoded"] = df["target_encoded"].astype(int)

    train_groups = build_experiment_groups(train_split, cfg)
    test_groups = build_experiment_groups(test_split, cfg)

    train_bag_sizes = [len(g["paths"]) for g in train_groups]
    test_bag_sizes = [len(g["paths"]) for g in test_groups]
    print(f"Train: {len(train_groups)} experiments, bag size min/mean/max = "
          f"{min(train_bag_sizes)}/{np.mean(train_bag_sizes):.1f}/{max(train_bag_sizes)}")
    print(f"Test:  {len(test_groups)} experiments, bag size min/mean/max = "
          f"{min(test_bag_sizes)}/{np.mean(test_bag_sizes):.1f}/{max(test_bag_sizes)}")
    from collections import Counter

    print("\nLabel distribution — TRAIN:")
    train_label_counts = Counter(g["label"] for g in train_groups)
    for label_id, count in sorted(train_label_counts.items()):
        print(f"  {cfg.classifier_class_names[label_id]}: {count} experiments")

    print("\nLabel distribution — TEST:")
    test_label_counts = Counter(g["label"] for g in test_groups)
    for label_id, count in sorted(test_label_counts.items()):
        print(f"  {cfg.classifier_class_names[label_id]}: {count} experiments")
    return train_groups, test_groups


class ExperimentDataset(Dataset):
    def __init__(self, groups: list, transform: A.Compose):
        self.groups = groups
        self.transform = transform

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        images = []
        for path in group["paths"]:
            image_np = np.array(Image.open(path).convert("RGB"))
            augmented = self.transform(image=image_np)
            images.append(augmented["image"])
        images = torch.stack(images)  # (n_images, C, H, W)
        return images, group["label"], group["experiment_id"]


def collate_experiments(batch):
    """Bag sizes vary per experiment, so images stay a list of tensors
    (one per experiment) rather than being stacked into one batch tensor."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    experiment_ids = [item[2] for item in batch]
    return images, labels, experiment_ids


def build_transforms(cfg):
    """Same crop-based transform as the rest of the pipeline (PadIfNeeded +
    Random/CenterCrop) -- no fiber_stack preprocessing, no alternate resize
    strategies, per your call to keep this ablation isolated to pooling."""
    train_tf = A.Compose([
        A.PadIfNeeded(min_height=cfg.image_size, min_width=cfg.image_size),
        A.RandomCrop(height=cfg.image_size, width=cfg.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.PadIfNeeded(min_height=cfg.image_size, min_width=cfg.image_size),
        A.CenterCrop(height=cfg.image_size, width=cfg.image_size),
        A.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
        ToTensorV2(),
    ])
    return train_tf, val_tf


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images_list, labels, _ in loader:
            for images, label in zip(images_list, labels):
                logits = model(images.to(device))
                all_preds.append(int(logits.argmax().cpu()))
                all_labels.append(int(label))

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }
    return metrics, all_preds, all_labels
    
def final_evaluation_and_report(model, loader, device, cfg) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images_list, labels, _ in loader:
            for images, label in zip(images_list, labels):
                logits = model(images.to(device))
                all_preds.append(int(logits.argmax().cpu()))
                all_labels.append(int(label))
    

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n--------------------- final report ----------------------------")
    print(f"Accuracy: {accuracy:.3f} | Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"Macro F1: {f1_macro:.3f} | Weighted F1: {f1_weighted:.3f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=cfg.classifier_class_names))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg.classifier_class_names)
    disp.plot()
    plt.tight_layout()
    plt.savefig(cfg.segformer_mil_classifier_output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    final_metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    summary_df = pd.DataFrame([final_metrics])
    summary_path = cfg.segformer_mil_classifier_output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to {summary_path}")


def main(cfg: Config):
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    output_dir = cfg.segformer_mil_classifier_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_groups, test_groups = load_fixed_split_groups(cfg)

    train_tf, val_tf = build_transforms(cfg)
    train_ds = ExperimentDataset(train_groups, train_tf)
    test_ds = ExperimentDataset(test_groups, val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.classifier_mil_batch_size, shuffle=True,
                               num_workers=cfg.classifier_num_workers, collate_fn=collate_experiments)
    test_loader = DataLoader(test_ds, batch_size=cfg.classifier_mil_batch_size, shuffle=False,
                              num_workers=cfg.classifier_num_workers, collate_fn=collate_experiments)

    encoder = load_pretrained_encoder(cfg, device)
    model = ExperimentMILClassifier(
        encoder=encoder,
        num_classes=len(cfg.classifier_class_names),
        hidden_dim=cfg.classifier_mil_hidden_dim,
        dropout=cfg.classifier_mil_dropout,
        freeze_encoder=cfg.freeze_encoder,
        chunk_size=cfg.classifier_mil_chunk_size,
        mil_arch=cfg.classifier_mil_arch,
        attention_hidden_dim=cfg.classifier_mil_attention_hidden_dim,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=cfg.classifier_mil_lr, weight_decay=cfg.classifier_mil_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.classifier_mil_epochs)

    best_f1 = -1.0
    history = []

    for epoch in range(cfg.classifier_mil_epochs):
        model.train()
        total_loss = 0.0

        for images_list, labels, _ in train_loader:
            optimizer.zero_grad()
            for images, label in zip(images_list, labels):
                logits = model(images.to(device))
                loss = criterion(logits.unsqueeze(0), label.to(device).unsqueeze(0))
                (loss / len(images_list)).backward()
                total_loss += loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        val_metrics, val_preds, val_labels = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1:03d}/{cfg.classifier_mil_epochs} | train loss {train_loss:.4f} | "
              f"val acc {val_metrics['accuracy']:.4f} f1 {val_metrics['f1_macro']:.4f}")

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val": val_metrics})

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            cm = confusion_matrix(val_labels, val_preds)
            with open(output_dir / "best_confusion_matrix.json", "w") as f:
                json.dump({"classes": cfg.classifier_class_names, "confusion_matrix": cm.tolist()}, f, indent=2)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val macro-F1: {best_f1:.4f}")
    print(f"Artifacts saved to: {output_dir}")

    best_model = model
    best_model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    final_evaluation_and_report(best_model, test_loader, device, cfg)

if __name__ == "__main__":
    cfg = Config()
    main(cfg)
