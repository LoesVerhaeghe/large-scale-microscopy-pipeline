import pandas as pd

original_file = "data/analysis/train_test_splits_newdata_test/filamentabundance_train_test_split.xlsx"
majority_labels_file = "data/analysis/filaments/SSCO_FIG_2_GroupSplit_Seed0_Majority.xlsx"
new_file = "data/analysis/train_test_splits_newdata_test/filamentabundance_train_test_split_majority.xlsx"

id_col = "resolved_image_path"  # Change to your actual shared identifier

# Combine majority labels from BOTH sheets
majority_train = pd.read_excel(majority_labels_file, sheet_name="train")
majority_test = pd.read_excel(majority_labels_file, sheet_name="test")

majority_df = pd.concat(
    [majority_train, majority_test],
    ignore_index=True
)

if majority_df[id_col].duplicated().any():
    raise ValueError(
        "Duplicate image identifiers found across majority train and test."
    )

majority_map = majority_df.set_index(id_col)

total_changed = 0

with pd.ExcelWriter(new_file, engine="openpyxl") as writer:
    for sheet in ["train", "test"]:
        original_df = pd.read_excel(original_file, sheet_name=sheet)

        matched = original_df[id_col].isin(majority_map.index)

        new_labels = original_df[id_col].map(majority_map["label"])
        new_label_ids = original_df[id_col].map(majority_map["label_id"])

        # Compare labels ignoring capitalization and surrounding spaces
        original_norm = (
            original_df["label"].fillna("").astype(str).str.strip().str.lower()
        )
        majority_norm = (
            new_labels.fillna("").astype(str).str.strip().str.lower()
        )

        changed = matched & new_labels.notna() & original_norm.ne(majority_norm)

        # Update both columns
        original_df.loc[changed, "label"] = new_labels[changed]
        original_df.loc[changed, "label_id"] = new_label_ids[changed]

        original_df.to_excel(writer, sheet_name=sheet, index=False)

        n_changed = int(changed.sum())
        total_changed += n_changed

        print(f"\n{sheet}:")
        print(f"  Original rows: {len(original_df)}")
        print(f"  Matched images: {int(matched.sum())}")
        print(f"  Unmatched images: {int((~matched).sum())}")
        print(f"  Labels changed: {n_changed}")

print(f"\nTotal labels changed: {total_changed}")
print(f"Saved to: {new_file}")