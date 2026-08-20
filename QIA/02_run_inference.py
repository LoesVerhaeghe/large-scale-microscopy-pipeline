"""
Runs the segmentation model over every phase-contrast 100x image in the
match table (skipping images already processed for this run_name), and
saves masks + an updated match table under output/<run_name>/.

Rerunning this script is safe/incremental: it skips any image whose mask
already exists on disk for the current run_name, so if it's interrupted or
you add new rows to the match table, only the missing masks get computed.
"""

from config.config import cfg  

import pandas as pd
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from .segmentation_utils import build_val_transform, load_segmentation_model, predict_full_image


def build_ph100_table(cfg) -> pd.DataFrame:
    match_table = pd.read_excel(cfg.match_table_path)
    classification_df = pd.read_csv(cfg.phase_contrast_classification_path)

    ph100_paths = classification_df.loc[
        (classification_df["classification_category"] == "Phase Contrast")
        & (classification_df["ocr_text_clean"] == "100 um"),
        "image_path"
    ].tolist()

    match_table_ph100 = match_table[match_table["image_path"].isin(ph100_paths)].copy()

    match_table_ph100["mask_path"] = match_table_ph100.index.map(
        lambda idx: str(cfg.masks_dir / f"image_{idx}.png")
    )

    return match_table_ph100


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.masks_dir.mkdir(parents=True, exist_ok=True)

    match_table_ph100 = build_ph100_table(cfg)
    print(f"{len(match_table_ph100)} phase-contrast 100x images to process")

    model = load_segmentation_model(cfg, device)
    val_transform = build_val_transform(cfg)

    n_processed, n_skipped_missing, n_skipped_existing = 0, 0, 0

    with torch.no_grad():
        for index, row in match_table_ph100.iterrows():
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])

            if not image_path.exists():
                print(f"Skipping, image not found: {image_path}")
                n_skipped_missing += 1
                continue

            if mask_path.exists():
                n_skipped_existing += 1
                continue

            image = np.array(Image.open(image_path).convert("RGB"))

            _, final_mask = predict_full_image(model, image, device, val_transform, cfg)
            Image.fromarray(final_mask.astype(np.uint8), mode="L").save(mask_path)
            n_processed += 1

    print(f"Processed: {n_processed}, skipped (missing image): {n_skipped_missing}, "
          f"skipped (mask already existed): {n_skipped_existing}")

    match_table_ph100.to_excel(cfg.match_table_with_masks_path, index=False)
    print(f"Saved updated match table to {cfg.match_table_with_masks_path}")


if __name__ == "__main__":
    main()
