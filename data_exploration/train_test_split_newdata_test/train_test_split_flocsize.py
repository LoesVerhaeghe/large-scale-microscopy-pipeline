#!/usr/bin/env python3

"""
Create a grouped train/test split for activated-sludge microscopy images.

Input 1:
    microscopyClassificationsV2_OCR_clean.csv
    microscopyClassifications_since_april2026_OCR_clean.csv

    Eligible images are those where:
        classification_category == "Phase Contrast"
        ocr_text_clean == "100 um"

Input 2:
    microscopic_match_table_extended.xlsx
    microscopic_match_table_extended_since_april2026.xlsx

    Images are matched using:
        image_path

    Groups are defined by:
        order_nr

    All images belonging to the same order_nr are kept entirely in
    either train or test.

Output:
    flocstructure_train_test_split.xlsx

    Sheets:
        train
        test

Target split:
    85% train
    15% test

Note:
    Since order_nr groups cannot be split, the final row percentage
    will generally not be exactly 85/15.
"""

from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# =============================================================================
# ADAPTABLE SETTLINGS
# =============================================================================


CSV_FILE1 = "data/PhaseContrast Classifier/microscopyClassificationsV2_OCR_clean.csv"
CSV_FILE2 = "data/PhaseContrast Classifier/microscopyClassifications_since_april2026_OCR_clean.csv"

EXCEL_FILE1 =  "data/microscopic_match_table_extended.xlsx"
EXCEL_FILE2 =  "data/microscopic_match_table_extended_since_april2026.xlsx"

OUTPUT_FILE =  "data/analysis/train_test_splits_newdata_test/flocsizes_train_test_split.xlsx" # to be changed!
LABEL_COLUMNS = ['Klein (KLEI_VGR_3) [%]', 'Middelgroot (MIDG_VGR_3) [%]', 'Groot (GROO_VGR_3) [%]', 'Gedispergeerd (GEDI_VGR_3) [%]' ]

TEST_FRACTION = 0.15
RANDOM_SEED = 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_path(path):
    """
    Normalize image paths so that paths from the CSV and Excel can be matched
    despite differences in slash direction, whitespace, etc.
    """
    if pd.isna(path):
        return None

    path = str(path).strip().replace("\\", "/")

    # Remove duplicate slashes
    while "//" in path:
        path = path.replace("//", "/")

    return path.lower()

# =============================================================================
# READ DATA
# =============================================================================

print("Reading CSV...")
df_csv1 = pd.read_csv(CSV_FILE1)
df_csv2 = pd.read_csv(CSV_FILE2)

print("Reading Excel...")
df_excel1 = pd.read_excel(EXCEL_FILE1)
df_excel2 = pd.read_excel(EXCEL_FILE2)

# =============================================================================
# CHECK REQUIRED COLUMNS
# =============================================================================

required_csv_columns = [
    "classification_category",
    "ocr_text_clean",
    "resolved_image_path", # maybe use image_path instead?
]

required_excel_columns = [
    "image_path",
    "order_nr",
] +LABEL_COLUMNS

for col in required_csv_columns:
    if col not in df_csv1.columns or col not in df_csv2.columns:
        raise ValueError(
            f'Column "{col}" is missing from {CSV_FILE1.name} or {CSV_FILE2.name}'
        )

for col in required_excel_columns:
    if col not in df_excel1.columns or col not in df_excel2.columns:
        raise ValueError(
            f'Column "{col}" is missing from {EXCEL_FILE1.name} or {EXCEL_FILE2.name}'
        )


# =============================================================================
# FILTER ELIGIBLE IMAGES
# =============================================================================

print("\nFiltering eligible images...")

def filter_eligible(df):
    return df[
        (df["classification_category"].astype(str).str.strip() == "Phase Contrast")
        &
        (df["ocr_text_clean"].astype(str).str.strip().str.lower() == "100 um")
    ].copy()

eligible1 = filter_eligible(df_csv1)  # TRAIN
eligible2 = filter_eligible(df_csv2)  # TEST

print(f"Eligible train CSV rows: {len(eligible1):,}")
print(f"Eligible test CSV rows: {len(eligible2):,}")


# =============================================================================
# MATCH TRAIN AND TEST DATA SEPARATELY
# =============================================================================

def match_dataset(eligible, df_excel, dataset_name):
    eligible = eligible.copy()
    df_excel = df_excel.copy()

    eligible["_match_path"] = eligible["image_path"].apply(normalize_path)
    df_excel["_match_path"] = df_excel["image_path"].apply(normalize_path)

    excel_lookup = (
        df_excel.drop_duplicates("_match_path")
        .set_index("_match_path")
    )

    matched = eligible[
        eligible["_match_path"].isin(excel_lookup.index)
    ].copy()

    unmatched = eligible[
        ~eligible["_match_path"].isin(excel_lookup.index)
    ]

    print(f"{dataset_name} matched: {len(matched):,}")
    print(f"{dataset_name} unmatched: {len(unmatched):,}")

    result = excel_lookup.loc[matched["_match_path"]].copy()
    result = result.reset_index(drop=True)

    result["resolved_image_path"] = (
        matched["resolved_image_path"].values
    )

    if "experiment_id" not in result.columns:
        result["experiment_id"] = result["order_nr"]

    if "images_in_experiment" not in result.columns:
        result["images_in_experiment"] = (
            result.groupby("order_nr")["image_path"].transform("count")
        )

    if "image_index_within_experiment" not in result.columns:
        result["image_index_within_experiment"] = (
            result.groupby("order_nr").cumcount() + 1
        )

    return result


matched_train = match_dataset(eligible1, df_excel1, "Train")
matched_test = match_dataset(eligible2, df_excel2, "Test")



# =============================================================================
# FILTER LABELS
# =============================================================================

matched_train = matched_train.drop_duplicates(
    subset=["image_path"]
).reset_index(drop=True)

matched_test = matched_test.drop_duplicates(
    subset=["image_path"]
).reset_index(drop=True)


# Keep only rows where all four labels are present
valid_rows = matched_train[LABEL_COLUMNS].notna().all(axis=1)

# Keep only rows where the four percentages add up to 100
import numpy as np
valid_rows &= np.isclose(
    matched_train[LABEL_COLUMNS].sum(axis=1),
    100,
    atol=0.01,
)

# Filter the dataset
n_removed = (~valid_rows).sum()
matched_train = matched_train.loc[valid_rows].copy()

# Keep only rows where all four labels are present
valid_rows = matched_test[LABEL_COLUMNS].notna().all(axis=1)

# Keep only rows where the four percentages add up to 100
import numpy as np
valid_rows &= np.isclose(
    matched_test[LABEL_COLUMNS].sum(axis=1),
    100,
    atol=0.01,
)

# Filter the dataset
n_removed = (~valid_rows).sum()
matched_test = matched_test.loc[valid_rows].copy()
# =============================================================================
# ASSIGN TRAIN AND TEST
# =============================================================================

matched_train["_split"] = "train"
matched_test["_split"] = "test"

# Check that no order_nr occurs in both datasets
overlap = (
    set(matched_train["order_nr"].dropna())
    & set(matched_test["order_nr"].dropna())
)

if overlap:
    raise ValueError(
        f"{len(overlap)} order_nr groups occur in both train and test."
    )

if matched_train["order_nr"].isna().any() or matched_test["order_nr"].isna().any():
    raise ValueError("Some matched images have missing order_nr values.")

print("Train/test group separation verified.")


# =============================================================================
# CREATE OUTPUT TABLES
# =============================================================================

output_columns = [
    "experiment_id",
    "images_in_experiment",
    "image_index_within_experiment",
    "resolved_image_path",
    "image_path",
    "sample_barcode",
    "order_nr",
    "installatie_naam",
    "datum_monstername",
] + LABEL_COLUMNS


train = matched_train[output_columns]
test = matched_test[output_columns]

# =============================================================================
# SORT
# =============================================================================

train = train.sort_values(
    ["order_nr", "image_index_within_experiment"]
).reset_index(drop=True)

test = test.sort_values(
    ["order_nr", "image_index_within_experiment"]
).reset_index(drop=True)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

total_images = len(train) + len(test)

print("\n" + "=" * 60)
print("FINAL SPLIT")
print("=" * 60)

print(f"Train images : {len(train):,}")
print(f"Test images  : {len(test):,}")
print(f"Total images : {total_images:,}")

if total_images > 0:
    print(f"Train fraction: {len(train) / total_images * 100:.2f}%")
    print(f"Test fraction : {len(test) / total_images * 100:.2f}%")

print(f"\nTrain groups : {train['order_nr'].nunique():,}")
print(f"Test groups  : {test['order_nr'].nunique():,}")


# =============================================================================
# CHECK LABEL DISTRIBUTION IN TRAIN VS TEST
# =============================================================================

print("\n" + "=" * 60)
print("FINAL SPLIT")
print("=" * 60)

print(f"Train images : {len(train):,}")
print(f"Test images  : {len(test):,}")
print(f"Total images : {len(train) + len(test):,}")

print(
    f"Train fraction: "
    f"{len(train) / len(train) + len(test) * 100:.2f}%"
)

print(
    f"Test fraction : "
    f"{len(test) / len(train) + len(test) * 100:.2f}%"
)

print(f"\nTrain groups : {train['order_nr'].nunique():,}")
print(f"Test groups  : {test['order_nr'].nunique():,}")

print(
    train[LABEL_COLUMNS]
    .agg(["mean", "std", "min", "median", "max"])
    .to_string()
)


print(
    test[LABEL_COLUMNS]
    .agg(["mean", "std", "min", "median", "max"])
    .to_string()
)


# =============================================================================
# WRITE EXCEL
# =============================================================================

print("\nWriting output to:")
print(OUTPUT_FILE)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    train.to_excel(
        writer,
        sheet_name="train",
        index=False
    )

    test.to_excel(
        writer,
        sheet_name="test",
        index=False
    )

print("\nDone.")
print(f"Output: {OUTPUT_FILE}")




