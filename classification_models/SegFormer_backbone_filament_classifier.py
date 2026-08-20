"""
SegFormer-encoder-as-classifier
--------------------------------
Reuses the MiT encoder from an already-trained 
SegFormer as a FROZEN feature extractor, adds a small classification
head on top, and trains only that head on your labels.

"""
from config.config import cfg  # must be imported first -- sets CPU thread env vars

import json
import os
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# --------------------------------------------------------------------------- #
# CONFIG -- edit these
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    seg_checkpoint_path: str = cfg.seg_checkpoint_path  
    metrics_path: str = cfg.metrics_path
    split_workbook_path: str = cfg.split_workbook_path
    split_train_sheet: str = cfg.split_train_sheet
    split_test_sheet: str = cfg.split_test_sheet
    split_merge_col: str = cfg.split_merge_col
    split_group_col: str = cfg.split_group_col
    image_size: int = cfg.tile_size         # resize target
    seed: int = cfg.seed
    gpu_id: int = cfg.gpu_id   
    
    image_path_col: str = "image_path"
    target_col: str = "Semikwantitatieve beoordeling (SSCO_FIG_2)"

    classifier_class_names = ["Matig", "Veel", "Zeer veel"]
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    freeze_encoder: bool = True    # False = fine-tune everything end-to-end
    hidden_dim: int = 256          # size of the MLP head's hidden layer (0 = linear probe only)
    dropout: float = 0.2
    output_dir: str =  cfg.output_dir / "classification_models" / "segformer_backbone_classifier"
            


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class SegformerEncoderClassifier(nn.Module):
    """
    Wraps an smp SegFormer's encoder (MiT backbone) with global average pooling
    and a classification head. The encoder can be frozen or fine-tuned.
    """

    def __init__(self, encoder: nn.Module, num_classes: int,
                 hidden_dim: int = 256, dropout: float = 0.2, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()  # keep BN/dropout in eval mode when frozen

        # smp encoders return a list of multi-scale feature maps; the last one
        # is the deepest / lowest-resolution stage -- what we want to pool for
        # a global classification feature.
        feat_channels = self.encoder.out_channels[-1]

        self.pool = nn.AdaptiveAvgPool2d(1)

        if hidden_dim and hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feat_channels, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # plain linear probe
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feat_channels, num_classes),
            )

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)

        last = feats[-1]           # (B, C, H', W')
        pooled = self.pool(last)   # (B, C, 1, 1)
        return self.head(pooled)

    def train(self, mode: bool = True):
        # override so encoder stays in eval() when frozen, even if someone
        # calls model.train() on the whole wrapper
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self


def load_pretrained_encoder(cfg: Config, device: torch.device) -> nn.Module:
    """
    Loads your trained segmentation model exactly the way your inference script
    does (torch.load of the whole object, not a state_dict), then pulls out
    just the encoder (MiT backbone).
    """
    seg_model = torch.load(cfg.seg_checkpoint_path, map_location=device)

    if not isinstance(seg_model, nn.Module):
        raise TypeError(
            f"Expected a whole nn.Module at {cfg.seg_checkpoint_path} (as in your "
            f"inference script), but got a {type(seg_model)}. If this checkpoint "
            f"is actually a state_dict, load it into an smp.Segformer(...) instance "
            f"first, then pass model.encoder here."
        )

    if not hasattr(seg_model, "encoder"):
        raise AttributeError(
            "Loaded model has no `.encoder` attribute -- check that this checkpoint "
            "is an segmentation_models_pytorch model (e.g. smp.Segformer(...))."
        )

    encoder = seg_model.encoder
    del seg_model
    return encoder


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def normalize_image_path(path):
    path = str(path).replace("\\", "/")
    marker = "Aquafin_data_cleaned/"

    if marker in path:
        return path[path.index(marker):]

    return path


def load_fixed_split(cfg):
    """
    Loads the authoritative train/test split.

    IMPORTANT:
    - Train/test assignment comes ONLY from the split workbook.
    - Shape features and the target label come from the metrics Excel.
    - The two files are joined on a normalized image path.
    """

    train_df = pd.read_excel(
        cfg.split_workbook_path,
        sheet_name=cfg.split_train_sheet
    )
    test_df = pd.read_excel(
        cfg.split_workbook_path,
        sheet_name=cfg.split_test_sheet
    )

    # Check that train/test are group-disjoint
    assert set(train_df[cfg.split_group_col]).isdisjoint(
        test_df[cfg.split_group_col]
    ), (
        "train/test experiment_id sets overlap -- "
        "split workbook is not group-disjoint"
    )

    print(
        f"Train: {len(train_df)} rows in split workbook -> "
        f"{len(train_df)} matched to metrics "
        f"({train_df[cfg.split_group_col].nunique()} experiments)"
    )
    print(
        f"Test:  {len(test_df)} rows in split workbook -> "
        f"{len(test_df)} matched to metrics "
        f"({test_df[cfg.split_group_col].nunique()} experiments)"
    )

    label_to_id = {
        label: index
        for index, label in enumerate(cfg.classifier_class_names)
    }

    train_df["target_encoded"] = train_df["label"].map(label_to_id)
    test_df["target_encoded"] = test_df["label"].map(label_to_id)

    n_unmapped = (
        train_df["target_encoded"].isna().sum()
        + test_df["target_encoded"].isna().sum()
    )

    if n_unmapped > 0:
        print(
            f"Warning: {n_unmapped} rows have a label not in "
            f"{cfg.classifier_class_names}, dropping them"
        )
        train_df = train_df.dropna(subset=["target_encoded"])
        test_df = test_df.dropna(subset=["target_encoded"])

    train_df["target_encoded"] = train_df["target_encoded"].astype(int)
    test_df["target_encoded"] = test_df["target_encoded"].astype(int)

    print("\nTarget distribution:")
    print(
        pd.concat([train_df, test_df])["label"]
        .value_counts()
        .reindex(cfg.classifier_class_names, fill_value=0)
    )

    return train_df, test_df


class QIAImageDataset(Dataset):
    """
    Reads image/label pairs from a dataframe produced by load_fixed_split().
    """

    def __init__(self, dataframe: pd.DataFrame, image_path_col: str,
                 transform: A.Compose):
        self.df = dataframe.reset_index(drop=True)
        self.image_path_col = image_path_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row[self.image_path_col]
        label = int(row["target_encoded"])

        image_np = np.array(Image.open(image_path).convert("RGB"))
        augmented = self.transform(image=image_np)
        image_tensor = augmented["image"]

        return image_tensor, label


def build_transforms(cfg: Config):
    # Same normalization as the SegFormer encoder training/inference.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = A.Compose([
        A.PadIfNeeded(min_height=cfg.image_size, min_width=cfg.image_size),
        A.RandomCrop(height=cfg.image_size, width=cfg.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.PadIfNeeded(min_height=cfg.image_size, min_width=cfg.image_size),
        A.CenterCrop(height=cfg.image_size, width=cfg.image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return train_tf, val_tf


def build_dataloaders(cfg: Config):
    # IMPORTANT: use the fixed train/test split defined above.
    train_df, test_df = load_fixed_split(cfg)

    # Check that all paths used by the dataset exist.
    for name, df in [("train", train_df), ("test", test_df)]:
        exists_mask = df[cfg.image_path_col].apply(os.path.isfile)
        n_missing = (~exists_mask).sum()

        if n_missing > 0:
            print(
                f"Warning: {n_missing} {name} image(s) not found on disk. "
                f"First few missing paths: "
                f"{df.loc[~exists_mask, cfg.image_path_col].head().tolist()}"
            )

        if not exists_mask.all():
            df.drop(
                index=df.index[~exists_mask],
                inplace=True
            )

    print("\nTraining samples:", len(train_df))
    print("Test samples:", len(test_df))

    train_tf, test_tf = build_transforms(cfg)

    train_ds = QIAImageDataset(
        train_df,
        cfg.image_path_col,
        train_tf
    )
    test_ds = QIAImageDataset(
        test_df,
        cfg.image_path_col,
        test_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, cfg.classifier_class_names


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_labels.append(y.detach().cpu())

    torch.set_grad_enabled(True)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }
    return metrics, all_preds, all_labels


def main(cfg: Config):
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, class_names = build_dataloaders(cfg)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    encoder = load_pretrained_encoder(cfg, device)
    model = SegformerEncoderClassifier(
        encoder=encoder,
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        freeze_encoder=cfg.freeze_encoder,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_f1 = -1.0
    history = []

    for epoch in range(cfg.epochs):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_metrics, test_preds, test_labels = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | "
              f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
              f"val loss {test_metrics['loss']:.4f} acc {test_metrics['accuracy']:.4f} f1 {test_metrics['f1_macro']:.4f}")

        history.append({"epoch": epoch + 1, "train": train_metrics, "test": test_metrics})

        if test_metrics["f1_macro"] > best_f1:
            best_f1 = test_metrics["f1_macro"]
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
            cm = confusion_matrix(test_labels, test_preds)
            with open(os.path.join(cfg.output_dir, "best_confusion_matrix.json"), "w") as f:
                json.dump({"classes": class_names, "confusion_matrix": cm.tolist()}, f, indent=2)

    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest test macro-F1: {best_f1:.4f}")
    print(f"Artifacts saved to: {cfg.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_checkpoint_path", type=str, default=Config.seg_checkpoint_path)
    parser.add_argument("--metrics_path", type=str, default=Config.metrics_path)
    parser.add_argument("--image_path_col", type=str, default=Config.image_path_col)
    parser.add_argument("--target_col", type=str, default=Config.target_col)
    parser.add_argument("--image_size", type=int, default=Config.image_size)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--freeze_encoder", action="store_true", default=Config.freeze_encoder)
    parser.add_argument("--fine_tune", dest="freeze_encoder", action="store_false",
                         help="fine-tune the whole encoder instead of freezing it")
    parser.add_argument("--hidden_dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--output_dir", type=str, default=Config.output_dir)
    args = parser.parse_args()

    cfg = Config(**vars(args))
    main(cfg)