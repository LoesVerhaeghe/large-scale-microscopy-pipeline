"""
SegFormer-encoder-as-classifier
--------------------------------
Reuses the MiT encoder from an already-trained SegFormer checkpoint as a FROZEN feature extractor,
adds a small classification head on top, and trains only that head on the labels.

Adjust the CONFIG block below and run:
    python segformer_encoder_classifier.py
"""

import os
import copy
import json
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import segmentation_models_pytorch as smp


# --------------------------------------------------------------------------- #
# CONFIG 
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    seg_checkpoint_path: str = 'models/trained_SegFormer_noVal.pt'   # your trained segmentation checkpoint
    encoder_name: str = "mit_b1"                     # must match what you trained segmentation with
    data_root: str = "data/classification"           # expects train/ and val/ subfolders (ImageFolder)
    image_size: int = 1024                             # match whatever you used for segmentation, or your ViT input size
    batch_size: int = 16
    num_workers: int = 2
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    freeze_encoder: bool = True                       # False = fine-tune everything end-to-end
    hidden_dim: int = 256                             # size of the MLP head's hidden layer (0 = linear probe only)
    dropout: float = 0.2
    output_dir: str = "classification_models/SegFormer_classifier"
    seed: int = 42


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class SegformerEncoderClassifier(nn.Module):
    """
    Wraps the SegFormer's encoder (MiT backbone) with global average pooling
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


def load_pretrained_encoder(cfg: Config) -> nn.Module:
    """
    Loads your fine-tuned segmentation checkpoint and returns just the encoder.
    Handles both `torch.save(model.state_dict(), ...)` and `torch.save(model, ...)`.
    """
    # Build a fresh smp model with the same architecture you trained with.
    # `classes` here is your *segmentation* class count -- it doesn't matter
    # for what we're doing since we only keep model.encoder, but it must match
    # the checkpoint's shapes if you're loading a state_dict.
    seg_model = smp.Segformer(
        encoder_name=cfg.encoder_name,
        encoder_weights=None,      # we're about to load your own weights instead
        in_channels=3,
        classes=1,                  # placeholder; irrelevant once we discard the decoder/head
    )

    checkpoint = torch.load(cfg.seg_checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        # whole model was saved
        seg_model = checkpoint
    else:
        # state_dict was saved
        try:
            seg_model.load_state_dict(checkpoint, strict=True)
        except RuntimeError as e:
            print("Strict load failed, retrying with strict=False. "
                  "Check this doesn't silently drop encoder weights:\n", e)
            seg_model.load_state_dict(checkpoint, strict=False)

    encoder = copy.deepcopy(seg_model.encoder)
    del seg_model
    return encoder


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def build_dataloaders(cfg: Config):
    # Keep normalization consistent with what you used for the segmentation
    # training and/or your ViT run -- mismatched normalization is a common
    # silent bug when reusing a pretrained encoder.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),      # microscopy images are usually orientation-invariant
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(os.path.join(cfg.data_root, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(cfg.data_root, "val"), transform=val_tf)

    assert train_ds.classes == val_ds.classes, "train/val class folders don't match"

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.classes


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
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = build_dataloaders(cfg)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    encoder = load_pretrained_encoder(cfg)
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
        val_metrics, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | "
              f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} f1 {val_metrics['f1_macro']:.4f}")

        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
            cm = confusion_matrix(val_labels, val_preds)
            with open(os.path.join(cfg.output_dir, "best_confusion_matrix.json"), "w") as f:
                json.dump({"classes": class_names, "confusion_matrix": cm.tolist()}, f, indent=2)

    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val macro-F1: {best_f1:.4f}")
    print(f"Artifacts saved to: {cfg.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_checkpoint_path", type=str, default=Config.seg_checkpoint_path)
    parser.add_argument("--encoder_name", type=str, default=Config.encoder_name)
    parser.add_argument("--data_root", type=str, default=Config.data_root)
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