import pandas as pd
import os
from pathlib import Path
import shutil

### check how many files are in the data folder, and how many of them are microscopic images
def get_all_files(folder_path):
    """Recursively get all files in a folder with their relative paths"""
    files = {}
    if not folder_path.exists():
        print(f"Warning: {folder_path} does not exist")
        return files
    
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            full_path = Path(root) / filename
            rel_path = full_path.relative_to(folder_path) # Get path relative to the folder root
            if 'negative' in str(rel_path).lower():  # skip files with 'negative' in the path
                continue
            files[str(rel_path)] = full_path # Store the full path for later use (e.g., to get file size)

    return files

data_path_all=Path("data/Aquafin_data_cleaned")
files_all = get_all_files(data_path_all)
print(f"Total files found in data folder: {len(files_all)}") 

data_path=Path("data/Aquafin data (Sorted_Nusret)")
files_sorted = get_all_files(data_path)
print(f"Total microscopic images found in sorted data folder: {len(files_sorted)}") 


### add info about lab experiments to the matches

# read excel overview sheet
excel_path="data/Aquafin_data_cleaned/other_files/microscopie_compleet_overzicht (slims databank + oude access databank).xlsx"
overview_df = pd.read_excel(excel_path, sheet_name="Overzicht")

# read table with images that are a match to the overview table
microscopic_match_table = pd.read_excel("outputs/microscopic_match_table.xlsx")

# Merge all overview information into the match table
microscopic_match_table_extended = microscopic_match_table.merge(
    overview_df,
    on="order_nr",
    how="left"
)

### check duplicate matches
# original duplicated file was manually checked to put files in correct folder
# around 630 duplicates remain because multiple lab test results belong to them
# only 2 paths need to be deleted since they do not fit the results
# duplicates = microscopic_match_table_extended[microscopic_match_table_extended["image_path"].duplicated(keep=False)]
# len(duplicates)

path1 = "data/Aquafin_data_cleaned/microscopie historische foto's (en aanverwante documenten)/J/Jabbeke/M388055/KIV_M388055250703140331.jpg"
path2 = "data/Aquafin_data_cleaned/microscopie historische foto's (en aanverwante documenten)/J/Jabbeke/M388055/KIV_M388055250703140331.jpg"
microscopic_match_table_extended = microscopic_match_table_extended[~microscopic_match_table_extended["image_path"].isin([path1, path2])]

# duplicates.to_excel("outputs/microscopic_match_table_extended_duplicates.xlsx", index=False)

#### check matches with low match confidence

# review_df = microscopic_match_table_extended.copy()
# review_df["manual_check"] = ""
# review_df.to_excel("outputs/matches_review.xlsx", index=False)

# manually check the matches in the excel file 
# if confidence is lower than 0.75, match is checked
# all matches were ok or fixed in folder

### check matches with low microscopic image confidence

low_microscopic_conf = microscopic_match_table_extended[microscopic_match_table_extended["microscopic_prob"] < 0.9].copy()

review_rows = []
review_dir = Path("outputs/classifier_microscopic_review")
review_dir.mkdir(parents=True, exist_ok=True)

for idx, row in low_microscopic_conf.iterrows():

    src = Path(row["image_path"])

    # Maak unieke bestandsnaam zodat niets overschreven wordt
    dst_name = f"{idx}_{src.name}"
    dst = review_dir / dst_name

    shutil.copy2(src, dst)

    review_rows.append({
        "index": idx,
        "image_path": row["image_path"],
        "review_image": str(dst),
        "is_microscopic": ""
    })

# review_df = pd.DataFrame(review_rows)
# review_df.to_excel(review_dir / "classifier_microscopic_review.xlsx", index=False)

# all images were checked and confirmed to be microscopic images, so no changes were made to the match table




# unique order nrs that can be linked to images in the match_table
matched_orders = set(microscopic_match_table["order_nr"]) 

unmatched_experiments = overview_df[
    ~overview_df["order_nr"].isin(matched_orders)
]
# there is not really a pattern in the unmatched experiments, 
# they are just experiments for which no data was saved? 

matched_experiments = overview_df[
    overview_df["order_nr"].isin(matched_orders)
]

microscopic_match_table_extended.to_excel("data/microscopic_match_table_extended.xlsx", index=False) # save it here for Nusret
microscopic_match_table_extended.to_excel("outputs/microscopic_match_table_extended.xlsx", index=False)





def get_all_files_without_extracted(folder_path):
    """Recursively get all files in a folder with their relative paths"""
    files = {}
    if not folder_path.exists():
        print(f"Warning: {folder_path} does not exist")
        return files
    
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            full_path = Path(root) / filename
            rel_path = full_path.relative_to(folder_path) # Get path relative to the folder root
            if 'extracted_images' in str(rel_path).lower():  # skip files with 'negative' in the path
                continue
            files[str(rel_path)] = full_path # Store the full path for later use (e.g., to get file size)
    return files

data_path_all=Path("data/Aquafin_data_cleaned")
files_all_without_extracted = get_all_files_without_extracted(data_path_all)

# table that links files to the overview table, based on filename and folder structure
matched_file_paths = []
others = []

for index, row in overview_df.iterrows():
    # extract relevant information from the overview table
    order_nr = str(row['order_nr'])
    labo_nr = str(row['labo_nummer'])
    sample_barcode = str(row['sample_barcode'])

    file_names = []
    

    for rel_path, full_path in files_all_without_extracted.items():
        file_name = os.path.basename(rel_path).lower()
        if file_name in ["thumbs.db", ".ds_store"]:
            continue
        if (order_nr in rel_path or labo_nr in rel_path or sample_barcode in rel_path):
            if file_name.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                if file_name in file_names: # if image is already there, skip to avoid duplicates
                    continue

                file_names.append(file_name)
                matched_file_paths.append(str(full_path))

            elif file_name.endswith(".pdf"):
                matched_file_paths.append(str(full_path))

            elif file_name.endswith(".docx"):
                matched_file_paths.append(str(full_path))

            else:
                others.append(str(full_path))
        
unmatched_file_paths = []
for rel_path, full_path in files_all_without_extracted.items():
    if str(full_path) not in matched_file_paths:
        unmatched_file_paths.append(rel_path)
