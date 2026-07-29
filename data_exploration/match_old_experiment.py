# here I could write a code that also matches the older experiments to the images (other sheets in the excel file)

from pathlib import Path
import pandas as pd
import os 

# read excel overview sheet
excel_path="data/Aquafin_data_cleaned/other_files/microscopie_compleet_overzicht (slims databank + oude access databank).xlsx"
probleem_link_labo_order_nr = pd.read_excel(excel_path, sheet_name="Probleem_monster")
# 2196 distinct problems
# 6467 lab nrs
# 6467 order nr

probleem_feedback = pd.read_excel(excel_path, sheet_name="Probleem_feedback") # vaak info van rwzi of waarom staal moet onderzocht worden
probleem_actie = pd.read_excel(excel_path, sheet_name="Probleem_actie") # actie te nemen 
probleem_waarneming = pd.read_excel(excel_path, sheet_name="Probleem_actie") # beschrijving resultaten labo onderzoek

probleem_overzicht = pd.read_excel(excel_path, sheet_name="Probleem_overzicht") 




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
            files[str(rel_path)] = full_path # Store the full path for later use (e.g., to get file size)
    return files

# table that indexes all files (folders, loose images, files like doc and pdf)
data_path_all=Path("data/Aquafin_data_cleaned")
files_all = get_all_files(data_path_all)


match_table = []
leftovers = []

for index, row in probleem_link_labo_order_nr.iterrows():
    # extract relevant information from the overview table
    probleem_nr = str(row['ProbleemNr'])
    labo_nr = str(row['oud_labo_nummer'])
    order_nr = str(row['slims_order_nr'])
    #experiment_date = pd.to_datetime(str(row['datum_monstername']), format="%Y-%m-%d %H:%M:%S")

    images = []
    pdfs = []
    docs = []
    others = []
    file_names = []

    for rel_path, full_path in files_all.items():
        file_name = os.path.basename(rel_path).lower()
        if file_name in ["thumbs.db", ".ds_store"]:
            continue
        if (order_nr in rel_path or labo_nr in rel_path in rel_path):
            if file_name.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                if file_name in file_names: # if image is already there, skip to avoid duplicates
                    continue

                file_names.append(file_name)
                images.append(str(full_path))

            elif file_name.endswith(".pdf"):
                pdfs.append(str(full_path))

            elif file_name.endswith(".docx"):
                docs.append(str(full_path))

            else:
                others.append(str(full_path))

    for f in others:
        leftovers.append({
            "order_nr": order_nr,
            "file": f
            })
        
    images = list(dict.fromkeys(images)) # preserve order, remove duplicates
    # for every matched file, we can calculate a confidence score based on how well the filename and folder structure match the experiment details
    for img in images:
        match_type = None
        if order_nr in img:
            match_type = "order_nr"
        elif labo_nr in img:
            match_type = "labo_nr"
        match_table.append({
                "order_nr": order_nr,
                "image_path": img,
                "match_type": match_type
            })
    for pdf in pdfs:
        match_type = None
        if order_nr in pdf:
            match_type = "order_nr"
        elif labo_nr in pdf:
            match_type = "labo_nr"
        match_table.append({
                "order_nr": order_nr,
                "image_path": pdf,
                "match_type": match_type
            })   
    for word in docs:
        match_type = None
        if order_nr in word:
            match_type = "order_nr"
        elif labo_nr in word:
            match_type = "labo_nr"
        match_table.append({
                "order_nr": order_nr,
                "image_path": word,
                "match_type": match_type
            })           
