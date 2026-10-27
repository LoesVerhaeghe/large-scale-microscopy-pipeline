#!/usr/bin/env python3

"""
Create a grouped train/test split for activated-sludge microscopy images.

Input 1:
    microscopyClassificationV2_OCR_clean.csv

    Eligible images are those where:
        classification_category == "Phase Contrast"
        ocr_text_clean == "100 um"

Input 2:
    microscopic_match_table_extended_withsvi.xlsx

    Images are matched using:
        image_path

    Groups are defined by:
        order_nr

    All images belonging to the same order_nr are kept entirely in
    either train or test.

Output:
    microscopy_SVI_train_test_split.xlsx

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


# =============================================================================
# SETTINGS
# =============================================================================
CSV_FILE1 = "data/PhaseContrast Classifier/microscopyClassificationsV2_OCR_clean.csv"
CSV_FILE2 = "data/PhaseContrast Classifier/microscopyClassifications_since_april2026_OCR_clean.csv"

EXCEL_FILE1 =  "data/microscopic_match_table_extended_withsvi.xlsx"
EXCEL_FILE2 =  "data/microscopic_match_table_extended_since_april2026_withsvi.xlsx"

OUTPUT_FILE =  "data/analysis/train_test_splits_newdata_test/SVI_train_test_split.xlsx" # to be changed!
LABEL_COLUMN = "SVI" # to be changed!

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
    LABEL_COLUMN,
]


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



matched_train = matched_train[
    matched_train[LABEL_COLUMN].notna()
].copy()

matched_train = matched_train[
    matched_train[LABEL_COLUMN]<500
].copy()

print(f"train rows remaining after removing SVI higher than 500: {len(matched_train):,}")

matched_test = matched_test[
    matched_test[LABEL_COLUMN].notna()
].copy()

matched_test = matched_test[
    matched_test[LABEL_COLUMN]<500
].copy()

print(f"test rows remaining after removing SVI higher than 500: {len(matched_test):,}")

# =============================================================================
# ASSIGN TRAIN AND TEST
# =============================================================================

matched_train["SVI"] = matched_train[LABEL_COLUMN].astype("string")
matched_test["SVI"] = matched_test[LABEL_COLUMN].astype("string")

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
    "SVI",
    "images_in_experiment",
    "image_index_within_experiment",
    "resolved_image_path",
    "image_path",
    "sample_barcode",
    "order_nr",
    "installatie_naam",
    "datum_monstername",
]

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


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

plt.hist(
    train["SVI"].dropna(),
    bins=20,
    alpha=0.6,
    label="Train",
)

plt.hist(
    test["SVI"].dropna(),
    bins=20,
    alpha=0.6,
    label="Test",
)

plt.xlabel("SVI")
plt.ylabel("Number of images")
plt.title("SVI Distribution by Train/Test Split")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


train_svi = pd.to_numeric(train["SVI"], errors="coerce").dropna()
test_svi = pd.to_numeric(test["SVI"], errors="coerce").dropna()

results = plt.boxplot(
    [train_svi, test_svi],
    labels=["Train", "Test"],
    patch_artist=True,
    boxprops={"color": "#4C72B0", "alpha": 0.7},
    medianprops={"color": "black", "linewidth": 1.5},
    whiskerprops={"color": "black"},
    capprops={"color": "black"},
    flierprops={"marker": "o", "markersize": 3},
)

for box, label, value in zip(
    results["boxes"],
    ["Train", "Test"],
    [train_svi.mean(), test_svi.mean()],
):
    box.set_label(f"{label}\nMean: {value:.3f}")

plt.xlabel("Dataset")
plt.ylabel("SVI")
plt.title("SVI Distribution by Train/Test Split")
plt.grid(axis="y", alpha=0.3)
plt.show()

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

