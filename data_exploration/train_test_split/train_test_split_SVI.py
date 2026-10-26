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

OUTPUT_FILE =  "data/analysis/train_test_splits/SVI_train_test_split.xlsx" # to be changed!
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

eligible1 = df_csv1[
    (df_csv1["classification_category"].astype(str).str.strip() == "Phase Contrast")
    &
    (df_csv1["ocr_text_clean"].astype(str).str.strip().str.lower() == "100 um")
].copy()
eligible2 = df_csv2[
    (df_csv2["classification_category"].astype(str).str.strip() == "Phase Contrast")
    &
    (df_csv2["ocr_text_clean"].astype(str).str.strip().str.lower() == "100 um")
].copy()

eligible = pd.concat([eligible1, eligible2], ignore_index=True)

print(f"Eligible CSV rows: {len(eligible):,}")

df_excel = pd.concat([df_excel1, df_excel2])
# =============================================================================
# PREPARE PATHS FOR MATCHING
# =============================================================================

eligible["_match_path"] = eligible["image_path"].apply(normalize_path)
df_excel["_match_path"] = df_excel["image_path"].apply(normalize_path)


# =============================================================================
# MATCH CSV IMAGES TO EXCEL
# =============================================================================

print("\nMatching images using image_path...")

# Check for duplicate image paths in the Excel file
duplicate_excel_paths = (
    df_excel["_match_path"]
    .value_counts()
)

duplicate_excel_paths = duplicate_excel_paths[
    duplicate_excel_paths > 1
]

if len(duplicate_excel_paths) > 0:
    print(
        f"WARNING: {len(duplicate_excel_paths):,} image paths occur "
        "more than once in the Excel file."
    )

excel_lookup = (
    df_excel
    .drop_duplicates("_match_path")
    .set_index("_match_path")
)

matched = eligible[
    eligible["_match_path"].isin(excel_lookup.index)
].copy()

print(f"Matched images: {len(matched):,}")

unmatched = eligible[
    ~eligible["_match_path"].isin(excel_lookup.index)
].copy()

print(f"Unmatched eligible images: {len(unmatched):,}")

if len(unmatched) > 0:
    print("\nExample unmatched paths:")
    for path in unmatched["image_path"].head(10):
        print("  ", path)


# =============================================================================
# GET INFORMATION FROM SECOND EXCEL
# =============================================================================

# The columns below come from the second Excel file.
excel_columns = [
    "image_path",
    "order_nr",
    LABEL_COLUMN
]

# Add all requested columns if they already exist in the Excel
optional_columns = [
    "experiment_id",
    "group_column",
    "group_id",
    "images_in_experiment",
    "image_index_within_experiment",
    "sample_barcode",
    "labo_nummer",
    "installatie_naam",
    "datum_monstername",
]

for col in optional_columns:
    if col in df_excel.columns:
        excel_columns.append(col)


matched_excel = excel_lookup.loc[
    matched["_match_path"]
].copy()

matched_excel = matched_excel.reset_index(drop=True)

# Use the resolved path from the CSV, not the Excel file.
matched_excel["resolved_image_path"] = (
    matched["resolved_image_path"].values
)

# =============================================================================
# CREATE / COMPLETE EXPERIMENT INFORMATION
# =============================================================================

# If experiment_id is not available, use order_nr as the experiment ID.
if "experiment_id" not in matched_excel.columns:
    matched_excel["experiment_id"] = matched_excel["order_nr"]

# =============================================================================
# CALCULATE IMAGE POSITION WITHIN EXPERIMENT
# =============================================================================

if "images_in_experiment" not in matched_excel.columns:
    matched_excel["images_in_experiment"] = (
        matched_excel.groupby("order_nr")["image_path"]
        .transform("count")
    )

if "image_index_within_experiment" not in matched_excel.columns:
    matched_excel["image_index_within_experiment"] = (
        matched_excel.groupby("order_nr")
        .cumcount()
        + 1
    )

# =============================================================================
# CHECK ORDER_NR
# =============================================================================

if matched_excel["order_nr"].isna().any():
    n_missing = matched_excel["order_nr"].isna().sum()

    raise ValueError(
        f"{n_missing} matched images have no order_nr. "
        "These cannot be assigned to a group and must be resolved first."
    )

# SVI is the prediction target, so only rows with a valid SVI
# can participate in the train/test split.
matched_excel[LABEL_COLUMN] = pd.to_numeric(
    matched_excel[LABEL_COLUMN],
    errors="coerce"
)

n_missing_svi = matched_excel[LABEL_COLUMN].isna().sum()

print(f"Rows with missing SVI: {n_missing_svi:,}")

matched_excel = matched_excel[
    matched_excel[LABEL_COLUMN].notna()
].copy()

print(f"Rows remaining after removing missing SVI: {len(matched_excel):,}")

matched_excel = matched_excel[
    matched_excel[LABEL_COLUMN]<500
].copy()

print(f"Rows remaining after removing SVI higher than 500: {len(matched_excel):,}")


matched_excel = matched_excel.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

# =============================================================================
# CREATE GROUP TABLE
# =============================================================================

groups = (
    matched_excel
    .groupby("order_nr")
    .size()
    .reset_index(name="n_images")
)

print(f"\nNumber of groups (order_nr): {len(groups):,}")
print(f"Number of matched images: {len(matched_excel):,}")

target_test_images = len(matched_excel) * TEST_FRACTION

print(f"Target test images: {target_test_images:,.1f}")


# =============================================================================
# GROUPED TRAIN/TEST SPLIT
# =============================================================================

"""
We want approximately 15% of the IMAGE ROWS in test, but entire order_nr
groups must remain together.

The groups are randomly shuffled first. We then select groups for test
while trying to stay close to the target number of test images.
"""

random.seed(RANDOM_SEED)

groups_list = groups.to_dict("records")
random.shuffle(groups_list)

selected_test_groups = []
test_images = 0

remaining_groups = []

for group in groups_list:
    n = group["n_images"]

    current_distance = abs(test_images - target_test_images)
    new_distance = abs((test_images + n) - target_test_images)

    if new_distance < current_distance:
        selected_test_groups.append(group["order_nr"])
        test_images += n
    else:
        remaining_groups.append(group)



test_groups = set(selected_test_groups)

matched_excel["_split"] = matched_excel["order_nr"].apply(
    lambda x: "test" if x in test_groups else "train"
)


# =============================================================================
# VERIFY NO GROUP IS MIXED
# =============================================================================

group_split_counts = (
    matched_excel
    .groupby("order_nr")["_split"]
    .nunique()
)

mixed_groups = group_split_counts[group_split_counts > 1]

if len(mixed_groups) > 0:
    raise RuntimeError(
        f"ERROR: {len(mixed_groups)} order_nr groups occur in both "
        "train and test."
    )

print("\nGroup split verified: no order_nr is mixed.")


# =============================================================================
# CREATE OUTPUT TABLES
# =============================================================================

# Exact requested column order
output_columns = [
    "experiment_id",
    "group_column",
    "group_id",
    "SVI",
    "images_in_experiment",
    "image_index_within_experiment",
    "resolved_image_path",
    "image_path",
    "sample_barcode",
    "order_nr",
    "labo_nummer",
    "installatie_naam",
    "datum_monstername",
]

# Make sure every requested column exists
for col in output_columns:
    if col not in matched_excel.columns:
        matched_excel[col] = pd.NA

train = matched_excel.loc[
    matched_excel["_split"] == "train",
    output_columns
].copy()

test = matched_excel.loc[
    matched_excel["_split"] == "test",
    output_columns
].copy()


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

print("\n" + "=" * 60)
print("FINAL SPLIT")
print("=" * 60)

print(f"Train images : {len(train):,}")
print(f"Test images  : {len(test):,}")
print(f"Total images : {len(train) + len(test):,}")

print(
    f"Train fraction: "
    f"{len(train) / len(matched_excel) * 100:.2f}%"
)

print(
    f"Test fraction : "
    f"{len(test) / len(matched_excel) * 100:.2f}%"
)

print(f"\nTrain groups : {train['order_nr'].nunique():,}")
print(f"Test groups  : {test['order_nr'].nunique():,}")


# =============================================================================
# WRITE EXCEL
# =============================================================================

print(f"\nWriting output to:")
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



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

plt.hist(
    train["SVI"].dropna(),
    bins=50,
    alpha=0.6,
    label="Train",
)

plt.hist(
    test["SVI"].dropna(),
    bins=50,
    alpha=0.6,
    label="Test",
)

plt.xlabel("SVI")
plt.ylabel("Number of images")
plt.title("SVI Distribution by Train/Test Split")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


train_svi = train["SVI"].dropna()
test_svi = test["SVI"].dropna()

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