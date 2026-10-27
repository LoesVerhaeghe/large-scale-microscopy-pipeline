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

OUTPUT_FILE =  "data/analysis/train_test_splits_newdata_test/filamentabundance_train_test_split.xlsx" # to be changed!
LABEL_COLUMN = "Semikwantitatieve beoordeling (SSCO_FIG_2)" # to be changed!
CORRECT_LABELS = ['Matig', 'Veel', 'Zeer veel']

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


def remove_incorrect_label_rows(df, label_column, correct_labels):
    """Keep rows whose label matches one of the accepted labels."""
    valid_labels = {label.strip().casefold() for label in correct_labels}
    labels = df[label_column].astype("string").str.strip().str.casefold()
    return df.loc[labels.isin(valid_labels)].copy()


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

matched_train = remove_incorrect_label_rows(
    matched_train, LABEL_COLUMN, CORRECT_LABELS
)
matched_test = remove_incorrect_label_rows(
    matched_test, LABEL_COLUMN, CORRECT_LABELS
)

matched_train = matched_train.drop_duplicates(
    subset=["image_path"]
).reset_index(drop=True)

matched_test = matched_test.drop_duplicates(
    subset=["image_path"]
).reset_index(drop=True)


# =============================================================================
# ASSIGN TRAIN AND TEST
# =============================================================================

matched_train["label"] = matched_train[LABEL_COLUMN].astype("string")
matched_test["label"] = matched_test[LABEL_COLUMN].astype("string")

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

label_mapping = {
        "Matig": 0,
        "Veel": 1,
        "Zeer veel": 2
}

output_columns = [
    "experiment_id",
    "label",
    "label_id",
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


def prepare_output(df):
    df = df.copy()

    df["label_id"] = (
        df[LABEL_COLUMN]
        .astype("string")
        .map(label_mapping)
    )

    for col in output_columns:
        if col not in df.columns:
            df[col] = pd.NA

    return df[output_columns].copy()


train = prepare_output(matched_train)
test = prepare_output(matched_test)


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
print("LABEL DISTRIBUTION: TRAIN VS TEST")
print("=" * 60)

distribution_counts = pd.DataFrame({
    "Train count": train["label"].value_counts(),
    "Test count": test["label"].value_counts(),
}).fillna(0).astype(int)

distribution_percentages = pd.DataFrame({
    "Train (%)": train["label"].value_counts(normalize=True) * 100,
    "Test (%)": test["label"].value_counts(normalize=True) * 100,
}).fillna(0)

distribution = distribution_counts.join(distribution_percentages)
distribution = distribution[
    ["Train count", "Train (%)", "Test count", "Test (%)"]
]

print(distribution.round(2).to_string())

print("\nLabel proportions within each dataset:")
print(
    pd.concat(
        [
            train["label"].value_counts(normalize=True).mul(100).rename("Train (%)"),
            test["label"].value_counts(normalize=True).mul(100).rename("Test (%)"),
        ],
        axis=1,
    ).fillna(0).round(2).to_string()
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


