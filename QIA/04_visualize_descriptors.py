"""
Visual QC for per-floc and image-level descriptors.
Runs segmentation on every image, overlays the predicted mask, annotates each
floc with selected descriptors, and shows image-level descriptors next to it.

Saves one figure per image to output/<run_name>/metrics_eval_visualizations/.
"""

from config.config import cfg
from QIA.segmentation_utils import build_val_transform, load_segmentation_model, predict_full_image, decode_mask

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology, measure
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
import skan

def um_per_pixel(mask_width_px: int) -> float:
    return 100.0 / (0.114 * mask_width_px) # scalebar_um = 100, scalebar widt fraction = 0.114

def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_segmentation_model(cfg, device)
    val_transform = build_val_transform(cfg)

    cfg.metrics_eval_visualizations_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in cfg.metrics_eval_images_dir.glob("*") if p.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No images found in {cfg.metrics_eval_images_dir}")

    for image_path in image_paths:
        image_np = np.array(Image.open(image_path).convert("RGB"))

        _, final_mask = predict_full_image(model, image_np, device, val_transform, cfg)
        mask_plt = decode_mask(final_mask, cfg.class_colors)

        px_to_um = um_per_pixel(final_mask.shape[1])

        floc_mask = final_mask == 1
        filament_mask = final_mask == 2

        floc_mask_clean = morphology.remove_small_objects(
            floc_mask, min_size=final_mask.shape[0] * final_mask.shape[1] * 1e-4 # drop connected components smaller than this fraction of image area`
        )
        labeled_flocs = measure.label(floc_mask_clean, connectivity=2)
        floc_regions = measure.regionprops(labeled_flocs)

        # ---- Floc descriptors ----
        areas_pixels = np.array([r.area for r in floc_regions])
        areas_um2 = areas_pixels * px_to_um**2

        eq_diameters_um = np.array([
            r.equivalent_diameter * px_to_um
            for r in floc_regions
        ])

        feret_diameters_um = np.array([
            r.feret_diameter_max * px_to_um
            for r in floc_regions
        ])

        major_axis_um = np.array([
            r.major_axis_length * px_to_um
            for r in floc_regions
        ])

        minor_axis_um = np.array([
            r.minor_axis_length * px_to_um
            for r in floc_regions
        ])

        crofton_perimeter_um = np.array([
            r.perimeter_crofton * px_to_um
            for r in floc_regions
        ])

        if len(floc_regions) > 0:
            aspect_ratios = 1 + 4 / np.pi * (
                major_axis_um / minor_axis_um - 1
            )

            form_factor = (
                4 * np.pi * areas_um2 /
                crofton_perimeter_um**2
            )

            roundness = (
                4 * areas_um2 /
                (np.pi * major_axis_um**2)
            )

            compactness = (
                np.sqrt(4 / np.pi * areas_um2) /
                feret_diameters_um
            )

            eccentricity = np.array([
                r.eccentricity for r in floc_regions
            ])

        # ---- Filament descriptors ----
        skeleton = skeletonize(filament_mask)

        if skeleton.sum() > 1:
            graph = skan.Skeleton(skeleton)
            filament_lengths_um = graph.path_lengths() * px_to_um

            total_filament_length_um = filament_lengths_um.sum()
            mean_filament_length_um = filament_lengths_um.mean()
            median_filament_length_um = np.median(filament_lengths_um)
            max_filament_length_um = filament_lengths_um.max()
            n_filament_paths = len(filament_lengths_um)

            skeleton_array = skeleton.astype(int)
            kernel = np.ones((3, 3), dtype=int)
            neighbours = ndi.convolve(
                skeleton_array,
                kernel,
                mode="constant",
                cval=0
            ) - skeleton_array

            n_junction_pixels = np.sum(
                skeleton & (neighbours >= 3)
            )
        else:
            total_filament_length_um = 0
            mean_filament_length_um = 0
            median_filament_length_um = 0
            max_filament_length_um = 0
            n_filament_paths = 0
            n_junction_pixels = 0

        total_filament_area_pixels = filament_mask.sum()
        total_filament_area_um2 = total_filament_area_pixels * px_to_um**2

        filament_area_fraction = (
            total_filament_area_pixels / filament_mask.size
        )

        total_floc_area_pixels = floc_mask.sum()

        filament_to_floc_ratio = (
            total_filament_area_pixels / total_floc_area_pixels
            if total_floc_area_pixels > 0 else np.nan
        )

        image_area_um2 = filament_mask.size * px_to_um**2

        filament_length_density = (
            total_filament_length_um / image_area_um2
        )

        branching_density = (
            n_junction_pixels / total_filament_length_um
            if total_filament_length_um > 0 else np.nan
        )

        # ---- Figure ----
        fig, (ax, ax_text) = plt.subplots(
            1, 2,
            figsize=(16, 10),
            gridspec_kw={"width_ratios": [3, 1]}
        )

        ax.imshow(image_np, alpha=0.75)
        ax.imshow(mask_plt, alpha=0.35)

        # Per-floc descriptors on image
        for i, r in enumerate(floc_regions):
            y, x = r.centroid

            text = (
                f"C={compactness[i]:.2f}\n"
                f"S={r.solidity:.2f}\n"
                f"E={eccentricity[i]:.2f}\n"
                f"AR={aspect_ratios[i]:.1f}\n"
                f"D={eq_diameters_um[i]:.0f} µm"
            )

            ax.text(
                x, y, text,
                fontsize=8,
                color="blue",
                ha="center",
                va="center"
            )

        ax.axis("off")
        ax.set_title(image_path.name)

        # ---- Image-level text ----
        if len(floc_regions) > 0:
            floc_text = (
                f"FLOCS\n"
                f"n flocs: {len(floc_regions)}\n"
                f"Mean area: {areas_um2.mean():.0f} µm²\n"
                f"Median area: {np.median(areas_um2):.0f} µm²\n"
                f"Mean eq. diameter: {eq_diameters_um.mean():.0f} µm\n"
                f"Median eq. diameter: {np.median(eq_diameters_um):.0f} µm\n"
                f"Mean major axis: {major_axis_um.mean():.0f} µm\n"
                f"Mean minor axis: {minor_axis_um.mean():.0f} µm\n\n"
                f"SHAPE\n"
                f"Aspect ratio: {aspect_ratios.mean():.2f}\n"
                f"Form factor: {form_factor.mean():.2f}\n"
                f"Roundness: {roundness.mean():.2f}\n"
                f"Compactness: {compactness.mean():.2f}\n"
                f"Eccentricity: {eccentricity.mean():.2f}\n"
            )
        else:
            floc_text = "FLOCS\nNo flocs detected\n"

        filament_text = (
            f"\nFILAMENTS\n"
            f"n paths: {n_filament_paths}\n"
            f"Total length: {total_filament_length_um:.0f} µm\n"
            f"Mean length: {mean_filament_length_um:.0f} µm\n"
            f"Median length: {median_filament_length_um:.0f} µm\n"
            f"Max length: {max_filament_length_um:.0f} µm\n"
            f"Area: {total_filament_area_um2:.0f} µm²\n"
            f"Area fraction: {filament_area_fraction:.3f}\n"
            f"Filament/floc ratio: {filament_to_floc_ratio:.3f}\n"
            f"Length density: {filament_length_density:.4f} µm/µm²\n"
            f"Junction pixels: {n_junction_pixels}\n"
            f"Branching density: {branching_density:.5f}"
        )

        ax_text.text(
            0,
            1,
            floc_text + filament_text,
            transform=ax_text.transAxes,
            fontsize=10,
            verticalalignment="top"
        )

        ax_text.axis("off")

        plt.tight_layout()

        out_path = (
            cfg.metrics_eval_visualizations_dir /
            f"{image_path.stem}.png"
        )

        plt.savefig(
            out_path,
            bbox_inches="tight",
            dpi=200
        )

        plt.close(fig)

    print(
        f"Saved {len(image_paths)} descriptor visualizations "
        f"to {cfg.metrics_eval_visualizations_dir}"
    )


if __name__ == "__main__":
    main()