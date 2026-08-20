
"""
Shared segmentation inference logic: loading the model, tiled prediction,
and mask decoding for visualization. Used by both
01_sample_image_segmentation.py (QC on handpicked images) and
02_run_inference.py (full dataset) so the tiling/thresholding logic exists
in exactly one place.
"""
 
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
 
from config.config import PipelineConfig
 
 
def build_val_transform(cfg: PipelineConfig) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})
 
 
def load_segmentation_model(cfg: PipelineConfig, device: torch.device) -> nn.Module:
    model = torch.load(cfg.seg_checkpoint_path, map_location=device)
    model.eval()
    return model
 
 
def predict_full_image(model: nn.Module, image_np: np.ndarray, device: torch.device,
                        val_transform: A.Compose, cfg: PipelineConfig):
    """
    Tiled inference over a full-resolution image, with overlap-averaging and
    per-class thresholding to produce a final class mask.
 
    Currently hardcoded to the 3-class (background/floc/filament) case via
    cfg.floc_threshold / cfg.filament_threshold. If you ever change
    num_seg_classes, this thresholding block needs to be generalized.
    """
    tile_size = cfg.tile_size
    overlap = cfg.overlap
    num_classes = cfg.num_seg_classes
 
    stride = tile_size - overlap
    H, W, _ = image_np.shape
 
    prob_map = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
 
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                tile = image_np[y:y + tile_size, x:x + tile_size]
                h_tile, w_tile = tile.shape[:2]
 
                if h_tile < tile_size or w_tile < tile_size:
                    pad_img = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                    pad_img[:h_tile, :w_tile] = tile
                    tile = pad_img
 
                augmented = val_transform(image=tile)
                tile_tensor = augmented["image"].unsqueeze(0).to(device)
 
                output = model(tile_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                probs = probs[:, :h_tile, :w_tile]
 
                prob_map[:, y:y + h_tile, x:x + w_tile] += probs
                count_map[y:y + h_tile, x:x + w_tile] += 1
 
    prob_map /= count_map  # (num_classes, H, W)
 
    # ---- threshold-based classification (floc/filament specific) ----
    floc_prob = prob_map[1]
    filament_prob = prob_map[2]
 
    final_mask = np.zeros((H, W), dtype=np.uint8)
 
    floc_pixels = floc_prob >= cfg.floc_threshold
    filament_pixels = filament_prob >= cfg.filament_threshold
 
    final_mask[floc_pixels] = 1
    final_mask[filament_pixels] = 2
 
    both = floc_pixels & filament_pixels
    final_mask[both & (floc_prob >= filament_prob)] = 1
    final_mask[both & (filament_prob > floc_prob)] = 2
 
    return prob_map, final_mask
 
 
def decode_mask(mask: np.ndarray, class_colors: dict) -> np.ndarray:
    """Convert [H, W] class mask -> RGB image using cfg.class_colors."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in class_colors.items():
        rgb[mask == cls] = color
    return rgb
 