import pandas as pd
import os
from pathlib import Path

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

# table that indexes all files (folders, loose images, files like doc and pdf)
data_path_all=Path("data/Aquafin_data_cleaned")
files_all = get_all_files(data_path_all)
print(f"Total files found in data folder: {len(files_all)}") 

data_path=Path("data/Aquafin data (Sorted_Nusret)")
files_sorted = get_all_files(data_path)
print(f"Total microscopic images found in sorted data folder: {len(files_sorted)}") 


# read excel overview sheet
excel_path="data/Aquafin_data_cleaned/other_files/microscopie_compleet_overzicht (slims databank + oude access databank).xlsx"
overview_df = pd.read_excel(excel_path, sheet_name="Overzicht")

# read table with images that are a match to the overview table
microscopic_match_table = pd.read_excel("microscopic_match_table.xlsx")

# unique order nrs that can be linked to images in the match_table
# matched_orders = set(microscopic_match_table["order_nr"]) 

# unmatched_experiments = overview_df[
#     ~overview_df["order_nr"].isin(matched_orders)
# ]

# matched_experiments = overview_df[
#     overview_df["order_nr"].isin(matched_orders)
# ]


# Merge all overview information into the match table
microscopic_match_table_extended = microscopic_match_table.merge(
    overview_df,
    on="order_nr",
    how="left"
)


microscopic_match_table_extended.to_excel("data/microscopic_match_table_extended.xlsx", index=False)
