import pandas as pd
from pathlib import Path
import os
import subprocess
from PIL import ExifTags, Image
import zipfile
import xml.etree.ElementTree as ET


### check excel files microscope investigations

# here just check how many investigations are available per installation, and how many of those have an image file available.
# outcome: df with installations, number of investigations +date , number of investigations with image file available +date


# read excel overview sheet
excel_path="data/Aquafin_data_cleaned/other_files/microscopie_compleet_overzicht (slims databank + oude access databank).xlsx"
overview_df = pd.read_excel(excel_path, sheet_name="Overzicht")

microscopic_match_table = pd.read_excel("outputs/data_exploration/microscopic_match_table.xlsx")

# unique order nrs that can be linked to images in the match_table
matched_orders = set(microscopic_match_table["order_nr"]) 

# Set these to the exact column names in the "Overzicht" sheet.
installation_col = "installatie_naam"
date_col = "datum_monstername"

overview_df[date_col] = pd.to_datetime(overview_df[date_col], errors="coerce")

investigation_availability_df = pd.DataFrame({
    "installatie": overview_df[installation_col],
    "investigation_available": overview_df[date_col],
    "investigation_available_with_images": overview_df[date_col].where(
        overview_df["order_nr"].isin(matched_orders)
    ),
})

investigation_availability_df.to_csv("process_timeseries/investigation_availability_df.csv", index=False)

### check image files  historical dataset

# here check how many image files/word/pdf doc with word fotobijlage are available per installation and their date
# outcome: df with installations, number of images/word/pdf docs +date



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
            if '_niet gerangschikt' in str(rel_path).lower():  # skip files with 'negative' in the path
                continue
            if '_oud' in str(rel_path).lower():  # skip files with 'negative' in the path
                            continue
            files[str(rel_path)] = full_path # Store the full path for later use (e.g., to get file size)
    return files

data_path_all=Path("data/Aquafin_data_cleaned/microscopie historische foto's (en aanverwante documenten)")
files_all = get_all_files(data_path_all)


def get_image_created_date(image_path: Path) -> tuple[pd.Timestamp | None, bool]:
    """Return an EXIF date, or the file modification date as a fallback."""
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)

            # Prefer the capture date; fall back to other EXIF date fields.
            date_value = (
                exif_ifd.get(ExifTags.Base.DateTimeOriginal)
                or exif_ifd.get(ExifTags.Base.DateTimeDigitized)
                or exif.get(ExifTags.Base.DateTime)
            )

        if date_value:
            date = pd.to_datetime(
                date_value,
                format="%Y:%m:%d %H:%M:%S",
                errors="coerce",
            )
            if not pd.isna(date):
                return date, False

    except (OSError, ValueError, KeyError):
        pass

    try:
        return pd.Timestamp.fromtimestamp(image_path.stat().st_mtime), True
    except OSError:
        return None, False

def get_word_doc_created_date(doc_path: Path) -> pd.Timestamp | None:
    """Return a Word document's embedded creation date, or None if unavailable."""
    try:
        if doc_path.suffix.lower() == ".docx":
            with zipfile.ZipFile(doc_path) as docx:
                core_xml = docx.read("docProps/core.xml")

            root = ET.fromstring(core_xml)
            created_tag = "{http://purl.org/dc/terms/}created"
            date_value = root.findtext(created_tag)

        elif doc_path.suffix.lower() == ".doc":
            import olefile

            with olefile.OleFileIO(str(doc_path)) as doc:
                date_value = doc.get_metadata().create_time

        else:
            return None

        if not date_value:
            return None

        date = pd.to_datetime(date_value, errors="coerce")
        return None if pd.isna(date) else date

    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None

def contains_fotobijlage(text: str) -> bool:
    return "fotobijlage" in text.casefold()

file_counts = {}
images_without_date = 0
images_using_file_date = 0
image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
document_extensions = {".doc", ".docx"}

for relative_path, file_path in files_all.items():
    path_parts = Path(relative_path).parts
    if len(path_parts) < 2:
        print(f"Warning: cannot determine installation from path: {file_path}")
        continue

    installation = path_parts[1]
    extension = file_path.suffix.lower()

    if extension in image_extensions:
        created_date, used_file_date = get_image_created_date(file_path)
        if used_file_date:
            images_using_file_date += 1
        count_column = "nr_images"
    elif extension in document_extensions:
        try:
            if not contains_fotobijlage(file_path.name):
                continue
            created_date = get_word_doc_created_date(file_path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.SubprocessError, FileNotFoundError) as error:
            print(f"Warning: could not read document {file_path}: {error}")
            continue
        count_column = "nr_fotobijlage_docs"
    else:
        continue

    if created_date is None or pd.isna(created_date):
        if count_column == "nr_images":
            images_without_date += 1
        print(f"Warning: no usable creation date; skipping {file_path}")
        continue

    date = pd.Timestamp(created_date).normalize().tz_localize(None)
    counts = file_counts.setdefault(
        (installation, date),
        {"nr_images": 0, "nr_fotobijlage_docs": 0},
    )
    counts[count_column] += 1

print(f"Images skipped because no creation date was available: {images_without_date}")
print(f"Images dated using file modification time: {images_using_file_date}")

historical_file_availability_df = pd.DataFrame(
    [
        {
            "installatie": installation,
            "datum": date,
            "nr_images": counts["nr_images"],
            "nr_fotobijlage_docs": counts["nr_fotobijlage_docs"],
        }
        for (installation, date), counts in file_counts.items()
    ],
    columns=["installatie", "datum", "nr_images", "nr_fotobijlage_docs"],
).sort_values(["installatie", "datum"]).reset_index(drop=True)


historical_file_availability_df.to_csv("process_timeseries/historical_file_availability_df.csv", index=False)
