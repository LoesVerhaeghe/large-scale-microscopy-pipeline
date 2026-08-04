from skimage import measure, morphology
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from pathlib import Path
from PIL import Image

def decode_mask(mask, COLORS):
    """Convert [H, W] class mask → RGB image"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask == cls] = color
    return rgb

COLORS = {
    0: [0, 0, 0],        # background
    1: [255, 0, 0],      # class 1 - red
    2: [0, 255, 0],      # class 2 - green
}

match_table_extended_ph100 = pd.read_excel("QIA/match_table_extended_ph100_masks.xlsx")

for index, row in match_table_extended_ph100[2300:2310].iterrows():
    mask_path = Path(row["mask_path"])
    image_path=Path(row["image_path"])
    if not mask_path.exists() or not image_path.exists():
        print(f"Skipping, mask doesn't exists: {mask_path}")
        continue

    image_np = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    um_per_pixel = 100/(0.114*mask.shape[1])

    mask_plt = decode_mask(mask, COLORS)

    floc_mask = (mask == 1)
    filament_mask = (mask == 2)

    # label individual flocs
    floc_mask_withoutsmallobjects = morphology.remove_small_objects(floc_mask, min_size=mask.shape[0]*mask.shape[1]*1e-4) # remove small objects smaller than 0.01% of the image area
    labeled_flocs = measure.label(floc_mask_withoutsmallobjects, connectivity=2)
    regions = measure.regionprops(labeled_flocs)

    fig, ax = plt.subplots(figsize=(12,10))
    ax.imshow(image_np, alpha=0.75)
    ax.imshow(mask_plt, alpha=0.35)

    for i, r in enumerate(regions):
        area = r.area
        perimeter = r.perimeter_crofton
        compactness = 4*np.pi*area/(perimeter**2)
        solidity = r.solidity
        eccentricity = r.eccentricity
        aspect_ratio = (
            r.major_axis_length / r.minor_axis_length
            if r.minor_axis_length > 0
            else 0
        )

        diameter_um = r.equivalent_diameter * um_per_pixel

        # centroid location
        y, x = r.centroid
        text = (
            f"C={compactness:.2f}\n"
            f"S={solidity:.2f}\n"
            f"E={eccentricity:.2f}\n"
            f"AR={aspect_ratio:.1f}\n"
            f"D={diameter_um:.0f}um"
        )
        ax.text(
            x,
            y,
            text,
            fontsize=8,
            color="blue",
            ha="center",
            va="center"
        )

    ax.axis("off")
    plt.show()
