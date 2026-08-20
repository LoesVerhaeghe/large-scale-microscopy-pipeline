"""
Computes QIA (quantitative image analysis) metrics from segmentation masks:
floc size/shape descriptors, filament length/area, and floc size-class counts
(small/medium/large/dispersed).

Reads masks from output/<run_name>/masks/ (via the match table written by
02_run_inference.py) and writes one merged metrics table to
output/<run_name>/metrics.xlsx.

"""

from config.config import cfg

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from skimage import measure, morphology
from skimage.morphology import skeletonize
import skan
from scipy import ndimage as ndi


def um_per_pixel(mask_width_px: int) -> float:
    return 100.0 / (0.114 * mask_width_px) # scalebar_um = 100, scalebar widt fraction = 0.114


def compute_floc_filament_metrics(mask: np.ndarray) -> dict:
    """
    All descriptors for one mask, in a single pass: floc geometry, filament
    geometry, and floc size-class counts (small/medium/large/dispersed).
    """
    floc_mask = (mask == 1)
    filament_mask = (mask == 2)
    px_to_um = um_per_pixel(mask.shape[1])

    # ---- floc region properties ----
    floc_mask_clean = morphology.remove_small_objects(
        floc_mask, min_size=mask.shape[0] * mask.shape[1] * 1e-4 # drop connected components smaller than this fraction of image area`
    )
    labeled_flocs = measure.label(floc_mask_clean, connectivity=2)
    floc_regions = measure.regionprops(labeled_flocs)

    areas_pixels = np.array([r.area for r in floc_regions])
    areas_um2 = areas_pixels * px_to_um**2
    eq_diameters_um = np.array([r.equivalent_diameter for r in floc_regions])* px_to_um
    feret_diameters_um = np.array([r.feret_diameter_max for r in floc_regions]) * px_to_um
    major_axis_pixels = np.array([r.major_axis_length for r in floc_regions])
    minor_axis_pixels = np.array([r.minor_axis_length for r in floc_regions])
    crofton_perimeter_um = np.array([r.perimeter_crofton for r in floc_regions]) * px_to_um
    aspect_ratios = 1 +  4/np.pi * (major_axis_pixels / minor_axis_pixels -1) # grijspeerdt and verstraete 1997
    form_factor = 4 * np.pi * areas_um2 / (crofton_perimeter_um ** 2) # grijspeerdt and verstraete 1997
    roundness = 4* areas_um2 / (np.pi * (major_axis_pixels*px_to_um)**2) # grijspeerdt and verstraete 1997
    compactness = np.sqrt(4/np.pi * areas_um2) / feret_diameters_um # Russ 1995
    eccentricity = np.array([r.eccentricity for r in floc_regions])
    

    micro_area = sum(
        r.area for r in floc_regions
        if r.equivalent_diameter * px_to_um < 50
    )
    total_floc_filament_px = floc_mask.sum() + filament_mask.sum()
    fraction_microflocs = micro_area / total_floc_filament_px if total_floc_filament_px > 0 else 0

    # ---- floc size classification (small/medium/large/dispersed) ----
    floc_classes = []
    for r in floc_regions:
        floc_size_um = r.major_axis_length * px_to_um
        perimeter = r.perimeter_crofton
        circularity = 4 * np.pi * r.area / (perimeter ** 2) if perimeter > 0 else 0

        if circularity < 0.15:
            floc_classes.append("dispersed")
        elif floc_size_um < 150:
            floc_classes.append("small")
        elif floc_size_um <= 500:
            floc_classes.append("medium")
        else:
            floc_classes.append("large")

    dispersed_area = sum(
        r.area for r, cls in zip(floc_regions, floc_classes) if cls == "dispersed"
    )
    fraction_area_dispersed = dispersed_area / areas_pixels.sum() if areas_pixels.size > 0 else 0

    # ---- Filament properties ----

    skeleton = skeletonize(filament_mask)

    if skeleton.sum() > 1:
        graph = skan.Skeleton(skeleton)

        # Individual filament/path lengths
        filament_lengths_um = graph.path_lengths() * px_to_um

        total_filament_length_um = filament_lengths_um.sum()
        mean_filament_length_um = filament_lengths_um.mean()
        median_filament_length_um = np.median(filament_lengths_um)
        max_filament_length_um = filament_lengths_um.max()
        n_filaments = len(filament_lengths_um)

        # Skeleton topology
        skeleton_array = skeleton.astype(int)
        kernel = np.ones((3, 3), dtype=int)
        neighbours = ndi.convolve(
            skeleton_array, kernel, mode="constant", cval=0
        ) - skeleton_array

        n_junctions = np.sum(skeleton & (neighbours >= 3)) #pixels where the filament branches

    else:
        filament_lengths_um = np.array([])
        total_filament_length_um = 0
        mean_filament_length_um = 0
        median_filament_length_um = 0
        max_filament_length_um = 0
        n_filaments = 0
        n_junctions = 0

    # Filament area
    total_filament_area_pixels = np.sum(filament_mask)
    total_filament_area_um2 = total_filament_area_pixels * px_to_um**2   
    filament_area_fraction = total_filament_area_pixels / filament_mask.size if filament_mask.size > 0 else 0 # Relative filament abundance
    filament_to_floc_ratio = total_filament_area_pixels / areas_pixels.sum() if areas_pixels.size > 0 else np.nan
    image_area_um2 = filament_mask.size * px_to_um**2 # Length density
    filament_length_density = total_filament_length_um / image_area_um2
    branching_density =  n_junctions / total_filament_length_um if total_filament_length_um > 0 else 0

    small_label = f"<150um"
    medium_label = f"150_500um"
    large_label = f">500um"
    return {
        "n_flocs": len(floc_regions),
        "total_floc_area_um2": areas_um2.sum(),
        "mean_floc_area_um2": areas_um2.mean(),
        "median_floc_area_um2": np.median(areas_um2) ,
        "mean_floc_eq_diameter_um": eq_diameters_um.mean(),
        "median_floc_eq_diameter_um": np.median(eq_diameters_um) ,
        "mean_floc_crofton_perimeter_um": crofton_perimeter_um.mean() ,
        "median_floc_crofton_perimeter_um": np.median(crofton_perimeter_um) ,
        "mean_major_axis_um": major_axis_pixels.mean() * px_to_um ,
        "mean_minor_axis_um": minor_axis_pixels.mean() * px_to_um ,
        f"n_flocs_eq_diameter_{small_label}": np.sum(eq_diameters_um < 150),
        f"n_flocs_eq_diameter_{medium_label}": np.sum((eq_diameters_um >= 150) & (eq_diameters_um <= 500)),
        f"n_flocs_eq_diameter_{large_label}": np.sum(eq_diameters_um > 500) ,
        f"n_flocs_feret_diameter_{small_label}": np.sum(feret_diameters_um < 150) ,
        f"n_flocs_feret_diameter_{medium_label}": np.sum((feret_diameters_um >= 150) & (feret_diameters_um <= 500)) ,
        f"n_flocs_feret_diameter_{large_label}": np.sum(feret_diameters_um > 500) ,
        "fraction_microflocs": fraction_microflocs,
        "eccentricity": np.mean(eccentricity),
        "aspect_ratio": np.mean(aspect_ratios) ,
        "form_factor": np.mean(form_factor) ,
        "roundness": np.mean(roundness),
        "compactness": np.mean(compactness) ,
        "n_small_flocs": floc_classes.count("small") ,
        "n_medium_flocs": floc_classes.count("medium"),
        "n_large_flocs": floc_classes.count("large") ,
        "n_dispersed_flocs": floc_classes.count("dispersed") ,
        "dispersed_area_px": dispersed_area ,
        "fraction_area_dispersed": fraction_area_dispersed,

        "n_filaments": n_filaments,
        "total_filament_length_um": total_filament_length_um,
        "total_filament_area_um2": total_filament_area_um2,
        "mean_filament_length_um": mean_filament_length_um,
        "median_filament_length_um": median_filament_length_um,
        "max_filament_length_um": max_filament_length_um,
        "filament_area_fraction": filament_area_fraction,
        "filament_length_density": filament_length_density,
        "filament_to_floc_ratio": filament_to_floc_ratio,
        "n_junctions": n_junctions,
        "branching_density": branching_density
    }

def main():
    match_table = pd.read_excel(cfg.match_table_with_masks_path)

    results = []
    n_skipped = 0
    for index, row in match_table.iterrows():
        mask_path = Path(row["mask_path"])
        image_path = Path(row["image_path"])

        if not mask_path.exists() or not image_path.exists():
            print(f"Skipping, mask or image not found: {mask_path}")
            n_skipped += 1
            continue

        mask = np.array(Image.open(mask_path))
        metrics = compute_floc_filament_metrics(mask)
        metrics["index"] = index
        results.append(metrics)

    print(f"Computed metrics for {len(results)} images, skipped {n_skipped}")

    metrics_df = pd.DataFrame(results).set_index("index")
    match_table_with_metrics = match_table.join(metrics_df)

    match_table_with_metrics.to_excel(cfg.metrics_path, index=False)
    print(f"Saved merged metrics table to {cfg.metrics_path}")


if __name__ == "__main__":
    main()