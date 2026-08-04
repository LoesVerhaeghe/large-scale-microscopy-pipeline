# scalebar 100 um is 11.4% of the width
# Kleine vlokken: < 150 μm --> 17.1%
# Middelgrote vlokken: 150 tot 500 μm --> 17.1%-57%
# Grote vlokken: > 500 μm ==> 57?

import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
from skimage import measure, morphology
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import skan

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
# floc_results = []

# for index, row in match_table_extended_ph100.iterrows():
#     mask_path = Path(row["mask_path"])
#     image_path=Path(row["image_path"])
#     if not mask_path.exists() or not image_path.exists():
#         print(f"Skipping, mask doesn't exists: {mask_path}")
#         continue

#     image_np = np.array(Image.open(image_path).convert("RGB"))
#     mask = np.array(Image.open(mask_path))
#     mask_plt = decode_mask(mask, COLORS)

#     floc_mask = (mask == 1)
#     filament_mask = (mask == 2)
#     um_per_pixel = 100/(0.114*mask.shape[1])

#     # calculate floc properties:
#     floc_mask_withoutsmallobjects = morphology.remove_small_objects(floc_mask, min_size=mask.shape[0]*mask.shape[1]*1e-4) # remove small objects smaller than 0.01% of the image area
#     labeled_flocs = measure.label(floc_mask_withoutsmallobjects, connectivity=2)   #Label connected regions of an integer array
#     floc_regions = measure.regionprops(labeled_flocs) #Measure properties of labeled image regions
    
#     areas_pixels = np.array([r.area for r in floc_regions])
#     diameters_pixels = np.array([r.equivalent_diameter for r in floc_regions])
#     major_axis_pixels = np.array([r.major_axis_length for r in floc_regions])
#     minor_axis_pixels = np.array([r.minor_axis_length for r in floc_regions])
#     crofton_perimeter = np.array([r.perimeter_crofton for r in floc_regions])
#     aspect_ratios = major_axis_pixels / minor_axis_pixels
#     compactness = 4*np.pi*areas_pixels/(crofton_perimeter**2)
#     eccentricity = [r.eccentricity for r in floc_regions]
#     eq_diameters_um = diameters_pixels * um_per_pixel
#     feret_diameters = [r.feret_diameter_max for r in floc_regions] 
#     feret_diameters_um = np.array(feret_diameters) * um_per_pixel
#     micro_area = 0
#     for r in floc_regions:
#         diameter_um = r.equivalent_diameter * um_per_pixel
#         if diameter_um < 50:
#             micro_area += r.area
#     fraction_microflocs = (micro_area / (floc_mask.sum() + filament_mask.sum()))                                       

#     # calculate filament properties:

#     skeleton = skeletonize(filament_mask)
#     if skeleton.sum() > 1:
#         graph = skan.Skeleton(skeleton)
#         total_filament_length = graph.path_lengths().sum()
#     else:
#         total_filament_length = 0

#     total_filament_area = np.sum(filament_mask) 

#     labeled_filaments = measure.label(skeleton)
#     filament_regions = measure.regionprops(labeled_filaments)
#     filament_lengths = []
#     for r in filament_regions:
#         length = r.area  # number of skeleton pixels
#         filament_lengths.append(length)

#     floc_results.append({
#         "index": index,
#         "n_flocs": len(floc_regions),
#         "total_floc_area_um2": areas_pixels.sum() * um_per_pixel**2 if areas_pixels.size > 0 else 0,
#         "mean_floc_area_um2": areas_pixels.mean() * um_per_pixel**2 if areas_pixels.size > 0 else 0,
#         "median_floc_area_um2": np.median(areas_pixels) * um_per_pixel**2 if areas_pixels.size > 0 else 0,
#         "mean_floc_diameter_um": eq_diameters_um.mean() if eq_diameters_um.size > 0 else 0,
#         "median_floc_diameter_um": np.median(eq_diameters_um) if eq_diameters_um.size > 0 else 0,
#         "mean_floc_crofton_perimeter_um": crofton_perimeter.mean() * um_per_pixel if crofton_perimeter.size > 0 else 0,
#         "median_floc_crofton_perimeter_um": np.median(crofton_perimeter) * um_per_pixel if crofton_perimeter.size > 0 else 0,
#         "mean_major_axis_um": major_axis_pixels.mean() * um_per_pixel if major_axis_pixels.size > 0 else 0,
#         "mean_minor_axis_um": minor_axis_pixels.mean() * um_per_pixel if minor_axis_pixels.size > 0 else 0,
#         "n_flocs_diameter_<150um": np.sum(eq_diameters_um < 150) if eq_diameters_um.size > 0 else 0,
#         "n_flocs_diameter_150_500um": np.sum(
#             (eq_diameters_um >= 150) & (eq_diameters_um <= 500)
#         ) if eq_diameters_um.size > 0 else 0,
#         "n_flocs_diameter_>500um": np.sum(eq_diameters_um > 500) if eq_diameters_um.size > 0 else 0,
#         "feret_n_flocs_diameter_<150um": np.sum(feret_diameters_um < 150) if feret_diameters_um.size > 0 else 0,
#         "feret_n_flocs_diameter_150_500um": np.sum(
#             (feret_diameters_um >= 150) & (feret_diameters_um <= 500)
#         ) if feret_diameters_um.size > 0 else 0,
#         "feret_n_flocs_diameter_>500um": np.sum(feret_diameters_um > 500) if feret_diameters_um.size > 0 else 0,
#         "fraction_microflocs": fraction_microflocs if fraction_microflocs else 0,
#         "eccentricity": np.mean(eccentricity) if eccentricity else 0,
#         "aspect_ratio": np.mean(aspect_ratios) if aspect_ratios.size > 0 else 0,
#         "compactness": np.mean(compactness) if compactness.size > 0 else 0,
#         "total_filament_length": total_filament_length* um_per_pixel if total_filament_length else 0,
#         "total_filament_area": total_filament_area* um_per_pixel**2 if total_filament_area else 0,
#         "mean_filament_length": np.mean(filament_lengths)* um_per_pixel if filament_lengths else 0,
#         "median_filament_length": np.median(filament_lengths)* um_per_pixel if filament_lengths else 0, 
#         "filament_to_floc_ratio": total_filament_area / areas_pixels.sum() if areas_pixels.size > 0 else 0,
    
#     })


# # Convert results to dataframe
# floc_results_df = pd.DataFrame(floc_results).set_index("index")

# # Add columns back to match table
# match_table_extended_ph100 = match_table_extended_ph100.join(
#     floc_results_df
# )


# match_table_extended_ph100.to_excel("match_table_extended_ph100_masks_with_metrics.xlsx", index=False)


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

    floc_mask = (mask == 1)
    filament_mask = (mask == 2)
    um_per_pixel = 100/(0.114*mask.shape[1])

    # calculate floc properties:
    floc_mask_withoutsmallobjects = morphology.remove_small_objects(floc_mask, min_size=mask.shape[0]*mask.shape[1]*1e-4) # remove small objects smaller than 0.01% of the image area
    labeled_flocs = measure.label(floc_mask_withoutsmallobjects, connectivity=2)   #Label connected regions of an integer array
    floc_regions = measure.regionprops(labeled_flocs) #Measure properties of labeled image regions

    # Calculate floc descriptors
    floc_sizes_um = []
    floc_classes = []

    for r in floc_regions:

        # size descriptor
        floc_size_um = r.major_axis_length  * um_per_pixel

        # circularity / compactness
        perimeter = r.perimeter_crofton
        circularity = 4 * np.pi * r.area / (perimeter ** 2) if perimeter > 0 else 0

        floc_sizes_um.append(floc_size_um)

        # First classify dispersion
        if circularity < 0.15:
            floc_class = "dispersed"

        # Otherwise classify by size
        elif floc_size_um < 150:
            floc_class = "small"

        elif floc_size_um <= 500:
            floc_class = "medium"

        else:
            floc_class = "large"

        floc_classes.append(floc_class)

    n_small = floc_classes.count("small")
    n_medium = floc_classes.count("medium")
    n_large = floc_classes.count("large")
    n_dispersed = floc_classes.count("dispersed")

    dispersed_area = sum(
        r.area for r, cls in zip(floc_regions, floc_classes)
        if cls == "dispersed"
    )

    fraction_area_dispersed = (
        dispersed_area / sum(r.area for r in floc_regions)
        if floc_regions else 0
    )

    floc_results.append({
        "index": index,
        "n_flocs": len(floc_regions),
        "n_small_flocs": n_small,
        "n_medium_flocs": n_medium,
        "n_large_flocs": n_large,
        "n_dispersed_flocs": n_dispersed,
        "dispersed_area": dispersed_area,
        "fraction_area_dispersed": fraction_area_dispersed
    })


# Convert results to dataframe
floc_results_df = pd.DataFrame(floc_results).set_index("index")

# Add columns back to match table
match_table_extended_ph100 = match_table_extended_ph100.join(
    floc_results_df
)


match_table_extended_ph100.to_excel("match_table_extended_ph100_masks_with_metrics_v2.xlsx", index=False)