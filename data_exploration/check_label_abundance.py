import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

path = "data/PhaseContrast Classifier/microscopyClassificationsV2_OCR_clean.csv"
path2 = "data/microscopic_match_table_extended.xlsx"


# ============================================================
# LOAD DATA
# ============================================================

df_photos = pd.read_csv(path)
df_labels = pd.read_excel(path2)

print("Path 1 shape:", df_photos.shape)
print("Path 2 shape:", df_labels.shape)

print("\nPath 1 columns:")
print(df_photos.columns.tolist())

print("\nPath 2 columns:")
print(df_labels.columns.tolist())


# ============================================================
# LABELS WE ARE INTERESTED IN
# ============================================================

label_columns = [
    "Effect op vlokstructuur (EFVS_FIG_2)",
    "Kleur (KLEU_MIC_SCR)",
    "Semikwantitatieve beoordeling (SSCO_FIG_2)",
    "Structuur (STRU_VMF_2)",
    "Vorm (VORM_VMF_2)",
    "Stevigheid (STEV_VMF_2)",
    # "n_flocs",
    # "total_floc_area_um2",
    # "mean_floc_area_um2",
    # "median_floc_area_um2",
    # "mean_floc_eq_diameter_um",
    # "median_floc_eq_diameter_um",
    # "mean_floc_crofton_perimeter_um",
    # "median_floc_crofton_perimeter_um",
    # "mean_major_axis_um",
    # "mean_minor_axis_um",
    # "fraction_microflocs",
    # "eccentricity",
    # "aspect_ratio",
    # "form_factor",
    # "roundness",
    # "compactness",
    # "dispersed_area_px",
    # "fraction_area_dispersed"
]

# ============================================================
# IMAGE PATH NORMALIZATION
# ============================================================
# The paths in the two files are similar but may not be
# character-for-character identical.
#
# This makes them easier to match:
# - converts \ to /
# - removes leading/trailing whitespace
# - converts to lowercase
#
# If the paths differ only in their root directory, we can
# additionally match using the part after Aquafin_data_cleaned.


def normalize_path(x):
    if pd.isna(x):
        return None

    x = str(x).strip().replace("\\", "/").lower()

    # remove duplicate slashes
    while "//" in x:
        x = x.replace("//", "/")

    return x


def path_key(x):
    """
    Create a robust matching key.

    If Aquafin_data_cleaned occurs in the path,
    use everything after it. This means that paths such as

    /some/root/Aquafin_data_cleaned/foo/bar.jpg

    and

    data/Aquafin_data_cleaned/foo/bar.jpg

    will still match.
    """
    x = normalize_path(x)

    if x is None:
        return None

    marker = "aquafin_data_cleaned/"

    if marker in x:
        return x.split(marker, 1)[1]

    return x


df_photos["image_path_key"] = df_photos["image_path"].apply(path_key)
df_labels["image_path_key"] = df_labels["image_path"].apply(path_key)


# ============================================================
# CHECK HOW WELL THE PATHS MATCH
# ============================================================

photo_keys = set(df_photos["image_path_key"].dropna())
label_keys = set(df_labels["image_path_key"].dropna())

matched_keys = photo_keys & label_keys

print("\nPath matching:")
print("Unique images in path 1:", len(photo_keys))
print("Unique images in path 2:", len(label_keys))
print("Images matched:", len(matched_keys))
print("Images only in path 1:", len(photo_keys - label_keys))
print("Images only in path 2:", len(label_keys - photo_keys))


# ============================================================
# MERGE
# ============================================================

# Keep the information from path 1 and add the labels/order_nr
# from path 2.

df = df_photos.merge(
    df_labels[
        ["image_path_key", "order_nr"] + label_columns
    ],
    on="image_path_key",
    how="left",
    suffixes=("", "_labeltable")
)


print("\nMerged dataframe:")
print(df.shape)

print("Images with an order_nr:",
      df["order_nr"].notna().sum())


# ============================================================
# DEFINE PHASE CONTRAST + 100 µm
# ============================================================

print("\nclassification_category values:")
print(df["classification_category"].value_counts(dropna=False))

print("\nOCR text examples:")
print(df["ocr_text_clean"].dropna().head(20).tolist())


# ------------------------------------------------------------
# IMPORTANT:
# Adjust this depending on the actual values in
# classification_category and ocr_text_clean.
# ------------------------------------------------------------

phase_contrast_mask = (
    df["classification_category"]
    .astype(str)
    .eq("Phase Contrast")
)

um_100_mask = (
    df["ocr_text_clean"]
    .astype(str)
    .str.lower()
    .str.contains("100 um", regex=True, na=False)
)

df["phase_contrast_100um"] = (
    phase_contrast_mask & um_100_mask
)

print("\nPhase contrast photos:", phase_contrast_mask.sum())
print("100 µm photos:", um_100_mask.sum())
print(
    "Phase contrast + 100 µm photos:",
    df["phase_contrast_100um"].sum()
)


# ============================================================
# CALCULATE STATISTICS PER LABEL
# ============================================================

results = []

for label in label_columns:

    # --------------------------------------------------------
    # Photos with a value for this label
    # --------------------------------------------------------

    has_label = df[label].notna()

    # Also exclude empty strings
    has_label &= df[label].astype(str).str.strip().ne("")

    n_photos_total = has_label.sum()

    # --------------------------------------------------------
    # Photos with label AND phase contrast + 100 µm
    # --------------------------------------------------------

    has_label_and_photo = (
        has_label &
        df["phase_contrast_100um"]
    )

    n_phasecontrast_100um = has_label_and_photo.sum()

    # --------------------------------------------------------
    # Experiments with this label
    #
    # One experiment can have many images, so count unique
    # order_nr.
    # --------------------------------------------------------

    n_experiments = (
        df.loc[has_label, "order_nr"]
        .dropna()
        .nunique()
    )

    # --------------------------------------------------------
    # Experiments with this label AND at least one
    # phase contrast + 100 µm photo
    # --------------------------------------------------------

    n_experiments_phasecontrast_100um = (
        df.loc[has_label_and_photo, "order_nr"]
        .dropna()
        .nunique()
    )

    results.append({
        "label": label,
        "photos_with_label": n_photos_total,
        "photos_with_label_phasecontrast_100um":
            n_phasecontrast_100um,
        "experiments_with_label":
            n_experiments,
        "experiments_with_label_phasecontrast_100um":
            n_experiments_phasecontrast_100um,
    })


results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY
# ============================================================

print("\n================ RESULTS ================\n")
print(results_df.to_string(index=False))


# ============================================================
# ALL COLUMNS IN PATH 2 WITH > 10,000 NON-EMPTY IMAGES
# ============================================================

# Columns in path2 that are definitely metadata, not labels
exclude_columns = {
    "image_path",
    "image_path_key",
    "order_nr",
}

# Everything else in path2
all_label_columns = [
    col for col in df_labels.columns
    if col not in exclude_columns
]

results = []

for label in all_label_columns:

    # --------------------------------------------------------
    # Does this label have a value?
    # --------------------------------------------------------

    has_label = (
        df_labels[label].notna()
        & df_labels[label].astype(str).str.strip().ne("")
    )

    n_photos_total = has_label.sum()

    # --------------------------------------------------------
    # Only continue if > 10,000 images have this label
    # --------------------------------------------------------

    if n_photos_total > 5_000:

        # ----------------------------------------------------
        # Get the corresponding image paths
        # ----------------------------------------------------

        label_paths = df_labels.loc[
            has_label,
            "image_path_key"
        ]

        # ----------------------------------------------------
        # Match these paths to path1
        # ----------------------------------------------------

        matched = df[
            df["image_path_key"].isin(label_paths)
        ]

        # Phase Contrast + 100 µm
        n_phasecontrast_100um = (
            matched["phase_contrast_100um"].sum()
        )

        # ----------------------------------------------------
        # Number of unique experiments
        # ----------------------------------------------------

        n_experiments = (
            df_labels.loc[has_label, "order_nr"]
            .dropna()
            .nunique()
        )

        # ----------------------------------------------------
        # Experiments with label AND
        # Phase Contrast + 100 µm photo
        # ----------------------------------------------------

        n_experiments_phasecontrast_100um = (
            matched.loc[
                matched["phase_contrast_100um"],
                "order_nr"
            ]
            .dropna()
            .nunique()
        )

        results.append({
            "label": label,
            "photos_with_label": n_photos_total,
            "photos_with_label_phasecontrast_100um":
                n_phasecontrast_100um,
            "experiments_with_label":
                n_experiments,
            "experiments_with_label_phasecontrast_100um":
                n_experiments_phasecontrast_100um,
        })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "photos_with_label",
    ascending=False
).reset_index(drop=True)

print("\n================================================")
print("LABELS WITH MORE THAN 10,000 IMAGES")
print("================================================\n")

print(results_df.to_string(index=False))