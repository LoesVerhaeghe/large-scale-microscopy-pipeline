# scalebar 100 um is 11.4% of the width
# Kleine vlokken: < 150 μm --> 17.1%
# Middelgrote vlokken: 150 tot 500 μm --> 17.1%-57%
# Grote vlokken: > 500 μm ==> 57?

import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
from skimage import measure, morphology
import matplotlib.pyplot as plt

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
floc_results = []

for index, row in match_table_extended_ph100.iterrows():
    mask_path = Path(row["mask_path"])
    image_path=Path(row["image_path"])
    if not mask_path.exists() or not image_path.exists():
        print(f"Skipping, mask doesn't exists: {mask_path}")
        continue

    image_np = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    mask_plt = decode_mask(mask, COLORS)

    # Keep only flocs (mask == 1)
    floc_mask = (mask == 1)
    filament_mask = (mask == 2)

    # Connected component labeling
    min_size=mask.shape[0]*mask.shape[1]*1e-4
    floc_mask_withoutsmallobjects = morphology.remove_small_objects(floc_mask, min_size=min_size) 
    labeled_flocs = measure.label(floc_mask_withoutsmallobjects, connectivity=2)   #Label connected regions of an integer array
    props = measure.regionprops(labeled_flocs) #Measure properties of labeled image regions

    um_per_pixel = 100/(0.114*mask.shape[1])

    if len(props) > 0:
        areas_pixels = np.array([p.area for p in props])
        diameters_pixels = np.array([p.equivalent_diameter for p in props])
        major_axis_pixels = np.array([p.major_axis_length for p in props])
        minor_axis_pixels = np.array([p.minor_axis_length for p in props])
        diameters_um = diameters_pixels * um_per_pixel

        floc_results.append({
            "index": index,
            "n_flocs": len(props),
            # "total_floc_area_um2": areas_pixels.sum() * um_per_pixel**2,
            # "mean_floc_area_um2": areas_pixels.mean() * um_per_pixel**2,
            # "median_floc_area_um2": np.median(areas_pixels) * um_per_pixel**2,
            # "mean_floc_diameter_um": diameters_pixels.mean() * um_per_pixel,
            # "median_floc_diameter_um": np.median(diameters_pixels) * um_per_pixel,
            # "mean_major_axis_um": major_axis_pixels.mean() * um_per_pixel,
            # "mean_minor_axis_um": minor_axis_pixels.mean() * um_per_pixel,
            "n_flocs_diameter_<150um": np.sum(diameters_um < 150),
            "n_flocs_diameter_150_500um": np.sum(
                (diameters_um >= 150) & (diameters_um <= 500)
            ),
            "n_flocs_diameter_>500um": np.sum(diameters_um > 500),
        })

    else:
        floc_results.append({
            "index": index,
            "n_flocs": 0,
            # "total_floc_area_um2": 0,
            # "mean_floc_area_um2": np.nan,
            # "median_floc_area_um2": np.nan,
            # "mean_floc_diameter_um": np.nan,
            # "median_floc_diameter_um": np.nan,
            # "mean_major_axis_um": np.nan,
            # "mean_minor_axis_um": np.nan,
            "n_flocs_diameter_<150um": np.nan,
            "n_flocs_diameter_150_500um": np.nan,
            "n_flocs_diameter_>500um": np.nan,
        })

    # floc_mask_withoutsmallobjects_plt = decode_mask(floc_mask_withoutsmallobjects, COLORS)

    # # Plot original, predicted mask and overlay
    # plt.figure(figsize=(12,4), dpi=500)
    # # Overlay ground truth
    # plt.subplot(1,3,1)
    # plt.imshow(image_np)
    # plt.title("Image")
    # plt.axis('off')

    # # Overlay predicted mask
    # plt.subplot(1,3,2)
    # plt.imshow(mask_plt)
    # plt.title("Predicted Mask")
    # plt.axis('off')

    # # Overlay predicted mask
    # plt.subplot(1,3,3)
    # plt.imshow(floc_mask_withoutsmallobjects_plt)
    # plt.axis('off')
    # print(labeled_flocs)




# Convert results to dataframe
floc_results_df = pd.DataFrame(floc_results).set_index("index")

# Add columns back to match table
match_table_extended_ph100 = match_table_extended_ph100.join(
    floc_results_df
)

#####
agg_dict = {
    # sums
    "n_flocs": "sum",
    "n_flocs_diameter_<150um": "mean",
    "n_flocs_diameter_150_500um": "mean",
    "n_flocs_diameter_>500um": "mean",

    # Utility measurements
    "Klein (KLEI_VGR_3) [%]": "first",
    "Middelgroot (MIDG_VGR_3) [%]": "first",
    "Groot (GROO_VGR_3) [%]": "first",
}

sample_df = (
    match_table_extended_ph100
    .groupby("order_nr", as_index=False)
    .agg(agg_dict)
)


from scipy.stats import pearsonr, spearmanr

comparisons = [
    ("Klein (KLEI_VGR_3) [%]", "n_flocs_diameter_<150um"),
    ("Middelgroot (MIDG_VGR_3) [%]", "n_flocs_diameter_150_500um"),
    ("Groot (GROO_VGR_3) [%]", "n_flocs_diameter_>500um"),
]

for aquafin_col, my_col in comparisons:

    df = sample_df[[aquafin_col, my_col]].dropna()

    pearson_r, pearson_p = pearsonr(df[aquafin_col], df[my_col])
    spearman_rho, spearman_p = spearmanr(df[aquafin_col], df[my_col])

    print(f"\n{aquafin_col} vs {my_col}")
    print(f"Pearson  r   = {pearson_r:.3f} (p={pearson_p:.4f})")
    print(f"Spearman rho = {spearman_rho:.3f} (p={spearman_p:.4f})")


for utility_col, my_col in comparisons:

    df = sample_df[[utility_col, my_col]].dropna()

    plt.figure(figsize=(5,5))
    plt.scatter(df[utility_col], df[my_col], alpha=0.2, s=15)

    plt.xlabel(f"Aquafin")
    plt.ylabel(f"Image analysis")
    plt.title(f"{my_col}")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()