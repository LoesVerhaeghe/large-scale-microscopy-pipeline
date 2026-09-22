"""
SegFormer-encoder MIL (multiple-instance learning) classifier
----------------------------------------------------------------
Instead of training per-image and aggregating predictions afterward, this
groups every experiment's images into a "bag", encodes each image with the
(frozen) SegFormer MiT encoder, mean-pools the per-image embeddings into one
experiment-level feature vector, and classifies that directly.

Train/test assignment comes from the authoritative fixed split workbook.

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
from dataclasses import dataclass, field
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# CONFIG -- edit these
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    seg_checkpoint_path: str = cfg.seg_checkpoint_path  
    split_workbook_path: str = "/data/nvme3/loesv/analysis/vlokgrotte/vlokgroote_GroupSplit_Seed0_Phase100um.xlsx"
    match_table_path : str = cfg.match_table_path
    split_train_sheet: str = "train"
    split_test_sheet: str = "validation"
    split_merge_col: str = cfg.split_merge_col
    split_group_col: str = cfg.split_group_col
    norm_mean: int = cfg.norm_mean
    norm_std: int = cfg.norm_std
    image_size: int = cfg.tile_size         # resize target
    seed: int = cfg.seed
    gpu_id: int = cfg.gpu_id   
    num_threads: int = cfg.num_threads

    freeze_encoder: bool = True    # False = fine-tune everything end-to-end
    target_cols: list = field(default_factory=lambda: [
        "Gedispergeerd (GEDI_VGR_3) [%]",
        "Klein (KLEI_VGR_3) [%]",
        "Middelgroot (MIDG_VGR_3) [%]",
        "Groot (GROO_VGR_3) [%]",
    ])
    target_sum_tolerance: float = 1.0   # allowed deviation from 100 before warning

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
    segformer_mil_classifier_output_dir: str =  cfg.output_dir / "classification_models" / "segformer_backbone_flocsize_mil_MEANpool_LR1e4"
            


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
    
class ExperimentMILRegressor(nn.Module):
    """
    Multiple-instance learning regressor: encodes every image in an
    experiment's bag, pools the per-image embeddings into one
    experiment-level feature vector, then regresses onto 4 floc-size
    fractions. A softmax + scale-by-100 output layer guarantees predictions
    are non-negative and sum to 100, matching the target compositional
    structure.
    """

    def __init__(self, encoder: nn.Module, num_targets: int, hidden_dim: int,
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
            nn.Linear(hidden_dim, num_targets),
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
        logits = self.head(pooled)
        return torch.softmax(logits, dim=-1) * 100.0  # scale to sum to 100

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
        target_values = group_df[cfg.target_cols].to_numpy(dtype=float)
        target = target_values[0]

        resolved_paths = group_df.apply(lambda row: _resolve_image_path(row, cfg), axis=1)
        exists_mask = resolved_paths.apply(lambda p: p.is_file())
        n_dropped_images += int((~exists_mask).sum())
        valid_paths = resolved_paths[exists_mask].tolist()

        if not valid_paths:
            print(f"Warning: experiment {exp_id} has no images on disk, skipping entirely")
            continue

        groups.append({
            "experiment_id": str(exp_id),
            "target": target,
            "paths": valid_paths,
        })

    if n_dropped_images > 0:
        print(f"Warning: dropped {n_dropped_images} individual missing image(s) across all experiments")

    return groups

def load_fixed_split_groups(cfg):
    train_split = pd.read_excel(cfg.split_workbook_path, sheet_name=cfg.split_train_sheet)
    test_split = pd.read_excel(cfg.split_workbook_path, sheet_name=cfg.split_test_sheet)

    assert set(train_split[cfg.split_group_col]).isdisjoint(test_split[cfg.split_group_col]), \
        "train/test experiment_id sets overlap -- split workbook is not group-disjoint"

    for name, df in (("train", train_split), ("test", test_split)):
        n_before = len(df)
        df.dropna(subset=cfg.target_cols, inplace=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"Warning: dropped {n_dropped} {name} rows with missing target values")
 
        row_sums = df[cfg.target_cols].sum(axis=1)
        bad_sum_mask = (row_sums - 100.0).abs() > cfg.target_sum_tolerance
        if bad_sum_mask.any():
            print(f"Warning: {bad_sum_mask.sum()} {name} rows have target columns summing to "
                  f"more than {cfg.target_sum_tolerance} away from 100 "
                  f"(min={row_sums.min():.2f}, max={row_sums.max():.2f})")

 
    train_groups = build_experiment_groups(train_split, cfg)
    test_groups = build_experiment_groups(test_split, cfg)

    train_bag_sizes = [len(g["paths"]) for g in train_groups]
    test_bag_sizes = [len(g["paths"]) for g in test_groups]
    print(f"Train: {len(train_groups)} experiments, bag size min/mean/max = "
          f"{min(train_bag_sizes)}/{np.mean(train_bag_sizes):.1f}/{max(train_bag_sizes)}")
    print(f"Test:  {len(test_groups)} experiments, bag size min/mean/max = "
          f"{min(test_bag_sizes)}/{np.mean(test_bag_sizes):.1f}/{max(test_bag_sizes)}")
    
    print("\nTarget distribution — TRAIN (mean ± std, %):")
    train_targets = np.stack([g["target"] for g in train_groups])
    for i, name in enumerate(cfg.target_cols):
        print(f"  {name}: {train_targets[:, i].mean():.2f} ± {train_targets[:, i].std():.2f}")
 
    print("\nTarget distribution — TEST (mean ± std, %):")
    test_targets = np.stack([g["target"] for g in test_groups])
    for i, name in enumerate(cfg.target_cols):
        print(f"  {name}: {test_targets[:, i].mean():.2f} ± {test_targets[:, i].std():.2f}")
 
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
        target = torch.tensor(group["target"], dtype=torch.float32)  # (4,)
        return images, target, group["experiment_id"]


def collate_experiments(batch):
    """Bag sizes vary per experiment, so images stay a list of tensors
    (one per experiment) rather than being stacked into one batch tensor."""
    images = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    experiment_ids = [item[2] for item in batch]
    return images, targets, experiment_ids


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
def _compute_regression_metrics(all_preds: np.ndarray, all_targets: np.ndarray, class_names: list) -> dict:
    """all_preds, all_targets: (n_experiments, num_targets)."""
    metrics = {
        "mae_overall": mean_absolute_error(all_targets, all_preds),
        "rmse_overall": root_mean_squared_error(all_targets, all_preds),
        "r2_overall": r2_score(all_targets, all_preds),
    }
    for i, name in enumerate(class_names):
        metrics[f"mae_{name}"] = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        metrics[f"rmse_{name}"] = root_mean_squared_error(all_targets[:, i], all_preds[:, i])
        metrics[f"r2_{name}"] = r2_score(all_targets[:, i], all_preds[:, i])
    return metrics

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images_list, targets, _ in loader:
            for images, target in zip(images_list, targets):
                pred = model(images.to(device))
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.numpy())
 
    all_preds = np.stack(all_preds)
    all_targets = np.stack(all_targets)
    return all_preds, all_targets

def final_evaluation_and_report(model, loader, device, cfg) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images_list, targets, _ in loader:
            for images, target in zip(images_list, targets):
                pred = model(images.to(device))
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.numpy())
 
    all_preds = np.stack(all_preds)
    all_targets = np.stack(all_targets)
    final_metrics = _compute_regression_metrics(all_preds, all_targets, cfg.target_cols)
 
    print(f"\n--------------------- final report ----------------------------")
    print(f"Overall MAE: {final_metrics['mae_overall']:.3f} | "
          f"RMSE: {final_metrics['rmse_overall']:.3f} | R2: {final_metrics['r2_overall']:.3f}")
    print("\nPer-class metrics:")
    for name in cfg.target_cols:
        print(f"  {name}: MAE={final_metrics[f'mae_{name}']:.3f}  "
              f"RMSE={final_metrics[f'rmse_{name}']:.3f}  R2={final_metrics[f'r2_{name}']:.3f}")
 
    # Parity plots (predicted vs. actual) replace the confusion matrix.
    n_classes = len(cfg.target_cols)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, cfg.target_cols)):
        ax.scatter(all_targets[:, i], all_preds[:, i], alpha=0.6)
        lims = [0, 100]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual (%)")
        ax.set_ylabel("Predicted (%)")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(cfg.segformer_mil_classifier_output_dir / "parity_plots.png", dpi=200)
    plt.close()
 
    summary_df = pd.DataFrame([final_metrics])
    summary_path = cfg.segformer_mil_classifier_output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to {summary_path}")
 
    return final_metrics


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
    model = ExperimentMILRegressor(
        encoder=encoder,
        num_targets=len(cfg.target_cols),
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(trainable, lr=cfg.classifier_mil_lr, weight_decay=cfg.classifier_mil_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.classifier_mil_epochs)

    best_mae = float("inf")
    history = []

    for epoch in range(cfg.classifier_mil_epochs):
        model.train()
        total_loss = 0.0

        for images_list, targets, _ in train_loader:
            optimizer.zero_grad()
            for images, target in zip(images_list, targets):
                pred  = model(images.to(device))
                loss = criterion(pred.unsqueeze(0), target.to(device).unsqueeze(0))
                (loss / len(images_list)).backward()
                total_loss += loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        val_preds, val_targets = evaluate(model, test_loader, device)
        val_metrics = _compute_regression_metrics(val_preds, val_targets, cfg.target_cols)
        print(f"Epoch {epoch+1:03d}/{cfg.classifier_mil_epochs} | train MSE {train_loss:.4f} | "
              f"val MAE {val_metrics['mae_overall']:.4f} | val R2 {val_metrics['r2_overall']:.4f}")
 
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val": val_metrics})
 
        if val_metrics["mae_overall"] < best_mae:
            best_mae = val_metrics["mae_overall"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            np.savez(output_dir / "best_val_predictions.npz",
                     preds=val_preds, targets=val_targets, class_names=cfg.target_cols)
 
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
 
    print(f"\nBest val MAE: {best_mae:.4f}")
    print(f"Artifacts saved to: {output_dir}")
 
    best_model = model
    best_model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    final_evaluation_and_report(best_model, test_loader, device, cfg)

if __name__ == "__main__":
    cfg = Config()
    main(cfg)
