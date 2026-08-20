"""
Runs the segmentation model on a fixed, handpicked set of QC images
(data/sample_images_qc/) and saves image / predicted mask / overlay figures
to output/<run_name>/qc_visualizations/, so you can compare segmentation
quality across finetuning iterations by just looking at the same filenames
under different run_name folders.
"""

from config.config import cfg

from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from QIA.segmentation_utils import build_val_transform, load_segmentation_model, predict_full_image, decode_mask


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_segmentation_model(cfg, device)
    val_transform = build_val_transform(cfg)

    cfg.qc_visualizations_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in cfg.qc_images_dir.rglob("**/*") if p.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No images found in {cfg.qc_images_dir}")

    for image_path in image_paths:
        image_np = np.array(Image.open(image_path).convert("RGB"))

        prob_map, final_mask = predict_full_image(model, image_np, device, val_transform, cfg)
        pred_rgb = decode_mask(final_mask, cfg.class_colors)

        fig = plt.figure(figsize=(12, 4), dpi=500)

        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title("Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_rgb)
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(image_np)
        plt.imshow(pred_rgb, alpha=0.4)
        plt.title("Image+predicted mask")
        plt.axis('off')

        out_path = cfg.qc_visualizations_dir / f"{image_path.stem}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(image_paths)} QC visualizations to {cfg.qc_visualizations_dir}")


if __name__ == "__main__":
    main()