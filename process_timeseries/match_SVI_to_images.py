"""
SCRIPT DOES NOT WORK YET
script that matches SVI measurements to images in the historical dataset
there is an issue with the location match (rwzi should not be accounted for as a match)
and there is an issue with the dates, function to retrieve image dates does not work
I continues with match_SVI_to_excel instead
"""


import pandas as pd
from pathlib import Path
import os
import time

# ---------------------------------------------------------
# 1. Prepare measurement data
# ---------------------------------------------------------

csv_path = "data/Aquafin data (vertrouwelijk NDA)-ORIGINAL/MLSS_turbiditeit_SVI_ZS_alle_zuiveringen.csv"  # Replace with the CSV file path.
df_timeseries = pd.read_csv(csv_path)

### pivot table to have one row per installation and date, with columns for each parameter
# Make sure dates are treated as dates
df_timeseries["datum"] = pd.to_datetime(df_timeseries["datum"])

# fix duplicates
df_clean = (
    df_timeseries.groupby(["installatie", "datum", "parameter"], as_index=False)
      .agg(waarde=("waarde", "mean"))
)

# Reshape from long → wide
df_wide = (
    df_clean.pivot(
        index=["installatie", "datum"],
        columns="parameter",
        values="waarde"
    )
    .reset_index()
)

# Remove the columns index name
df_wide.columns.name = None

print("New shape:", df_wide.shape)
print(df_wide.head())

# ---------------------------------------------------------
# 2. Select the two SVI parameters
# ---------------------------------------------------------

svi = df_wide.copy()

# Keep rows where at least one of the two SVI values is available
svi = svi[
    svi["SVI"].notna() | svi["SVI_historisch"].notna()
].copy().reset_index(drop=True)

print("Rows with at least one SVI value:", len(svi))


# ---------------------------------------------------------
# 3. Find the closest SVI measurement for every image
# ---------------------------------------------------------


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
print(f"Total files found in data folder: {len(files_all)}") 


# Match existing image files by installation word and calendar date.
from PIL import Image, ExifTags
from datetime import datetime
import re
import fitz
from docx import Document

TAGS = {value: key for key, value in ExifTags.TAGS.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
EXTRACTED_IMAGES_ROOT = data_path_all / "extracted_images"


def get_image_date(path):
    try:
        with Image.open(path) as image:
            exif = image.getexif()

        for field in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            tag_id = TAGS.get(field)
            if tag_id in exif:
                parsed_date = pd.to_datetime(exif[tag_id], errors="coerce")
                if not pd.isna(parsed_date):
                    return parsed_date.normalize()
    except Exception as error:
        print(f"Error reading image date from {path}: {error}")
    return None


def get_document_creation_date(path):
    try:
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as document:
                date_text = document.metadata.get("creationDate")
            if not date_text:
                return None
            date_digits = "".join(character for character in date_text if character.isdigit())
            date_format = "%Y%m%d%H%M%S" if len(date_digits) >= 14 else "%Y%m%d"
            date_value = date_digits[:14] if len(date_digits) >= 14 else date_digits[:8]
            return pd.Timestamp(datetime.strptime(date_value, date_format)).normalize()

        if path.suffix.lower() == ".docx":
            created = Document(path).core_properties.created
            return pd.Timestamp(created).tz_localize(None).normalize() if created else None
    except Exception as error:
        print(f"Error reading creation date from {path}: {error}")
    return None


def path_words(value): #returns a set of individual words, normalized to lowercase.
    return set(re.findall(r"[^\W_]+", str(value).casefold()))


document_paths = [
    path for path in files_all.values()
    if path.suffix.lower() in {".pdf", ".docx"}
]
document_date_cache = {}


def get_match_date(image_path):
    image_date = get_image_date(image_path)
    if image_date is not None:
        return image_date, "image_exif"

    try:
        extracted_relative_path = image_path.relative_to(EXTRACTED_IMAGES_ROOT)
    except ValueError:
        return None, None

    extracted_folder = extracted_relative_path.parts[0].casefold()
    matching_documents = [
        path for path in document_paths
        if path.stem.casefold() in extracted_folder
    ]
    if not matching_documents:
        return None, None

    source_document = max(matching_documents, key=lambda path: len(path.stem))
    if source_document not in document_date_cache:
        document_date_cache[source_document] = get_document_creation_date(source_document)
    return document_date_cache[source_document], "source_document_creation_date"


image_paths = [
    path for path in files_all.values()
    if path.suffix.lower() in IMAGE_EXTENSIONS
]
print(f"Image paths to process: {len(image_paths)}")
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import time


# Wrapper because get_match_date is the function we want to time out
def process_image(image_path):
    return get_match_date(image_path)


match_table = []

with ProcessPoolExecutor(max_workers=1) as executor:

    for image_count, image_path in enumerate(image_paths[:46000], start=1):

        future = executor.submit(process_image, image_path)

        try:
            image_date, date_source = future.result(timeout=5)

        except TimeoutError:
            print(
                f"\nTIMEOUT - skipping image {image_count}: "
                f"{image_path}"
            )
            future.cancel()
            continue

        except Exception as error:
            print(
                f"\nERROR - skipping image {image_count}: "
                f"{image_path}\n"
                f"Error: {error}"
            )
            continue

        if image_date is None:
            print(image_path, ': no image date')

        if image_date is not None:

            matching_rows = []

            for svi_row_index, row in svi.iterrows():

                installation = row["installatie"]
                experiment_date = pd.to_datetime(
                    row["datum"],
                    errors="coerce"
                )

                if pd.isna(installation) or pd.isna(experiment_date):
                    continue

                installation_words = path_words(installation)
                experiment_date = experiment_date.normalize()

                if not installation_words.intersection(
                    path_words(image_path)
                ):
                    continue

                day_diff = abs(
                    (image_date - experiment_date).days
                )

                if day_diff <= 10:
                    matching_rows.append(
                        (day_diff, svi_row_index, row)
                    )

            if matching_rows:

                day_diff, svi_row_index, closest_row = min(
                    matching_rows,
                    key=lambda match: (match[0], match[1])
                )

                match_table.append({
                    **closest_row.to_dict(),
                    "svi_row_index": svi_row_index,
                    "image_path": str(image_path),
                    "image_date": image_date,
                    "image_date_source": date_source,
                    "day_diff": day_diff,
                    "match_type": "installation_and_closest_date",
                })

        if image_count % 1000 == 0:
            print(
                f"Processed {image_count} of {len(image_paths)} image paths"
            )

        if 46000 <= image_count <= 47000:
            print(f"CURRENT IMAGE {image_count}: {image_path}", flush=True)


match_df = pd.DataFrame(match_table)
match_df.to_excel("process_timeseries/svi_matches_table.xlsx", index=False)
#filtered_match_df = match_df


# # #### classify matched images based on classifier model Nusret: microscopic vs non microscopic images
# # from PIL import Image
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision import transforms
# # from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
# # from tqdm import tqdm

# # checkpoint_path = Path("data/Microscopic Image Classifier/weights/best_resnext101_microscopyX3.pt")

# # batch_size = 16
# # num_workers = 4
# # threshold = 0.5


# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".psd", ".jp2"}

# # # --------------------------------------------------
# # # Dataset for unlabeled images
# # # --------------------------------------------------

# # class UnlabeledImageDataset(Dataset):
# #     def __init__(self, paths_to_images, transform=None):
# #         self.paths_to_images = paths_to_images
# #         self.transform = transform
# #         self.samples = []

# #         for path in paths_to_images:
# #             path = Path(path)
# #             if path.is_file() and path.suffix.lower() in valid_extensions:
# #                 self.samples.append(path)

# #         if len(self.samples) == 0:
# #             raise RuntimeError(f"No images found in {self.paths_to_images}")

# #     def __len__(self):
# #         return len(self.samples)

# #     def __getitem__(self, idx):
# #         path = self.samples[idx]

# #         try:
# #             with Image.open(path) as img:
# #                 img = img.convert("RGB")

# #                 if self.transform is not None:
# #                     img = self.transform(img)

# #             return img, str(path)

# #         except Exception as e:
# #             print(f"SKIPPING unreadable image: {path} | {e}")
# #             return None

# # def skip_bad_images_collate_fn(batch):
# #     ''' Custom collate function to skip None items returned by the dataset when an image fails to load.'''
# #     batch = [item for item in batch if item is not None]
# #     if len(batch) == 0:
# #         return None
# #     images, paths = zip(*batch)
# #     images = torch.stack(images, dim=0)
# #     return images, paths
    

# # # --------------------------------------------------
# # # Transform
# # # --------------------------------------------------
# # weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1
# # val_transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(
# #         mean=weights.transforms().mean,
# #         std=weights.transforms().std,),])


# # # --------------------------------------------------
# # # DataLoader
# # # --------------------------------------------------

# # dataset = UnlabeledImageDataset(filtered_match_df['image_path'], transform=val_transform)
# # loader = DataLoader(
# #     dataset,
# #     batch_size=batch_size,
# #     shuffle=False,
# #     num_workers=num_workers,
# #     pin_memory=True,
# #     collate_fn=skip_bad_images_collate_fn)

# # print(f"Images found: {len(dataset)}")


# # # --------------------------------------------------
# # # Load model
# # # --------------------------------------------------
# # model = resnext101_64x4d(weights=weights)
# # in_features = model.fc.in_features 
# # model.fc = nn.Linear(in_features, 1) # replace final layer for binary classification
# # checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
# # model.load_state_dict(checkpoint["model_state_dict"])
# # model = model.to(device)
# # model.eval()
# # print(f"Loaded model from: {checkpoint_path}")

# # microscopic_match = []
# # non_microscopic_match = []
# # failed = []

# # with torch.no_grad():
# #     for batch in tqdm(loader, desc="Classifying"):
# #         if batch is None:
# #             continue
    
# #         images, paths = batch
# #         images = images.to(device, non_blocking=True)
# #         logits = model(images)
# #         probs = torch.sigmoid(logits)

# #         for path_str, prob in zip(paths, probs):
# #             path = Path(path_str)
# #             positive_prob = float(prob.item())
# #             try:
# #                 if positive_prob >= threshold:
# #                     microscopic_match.append((path, positive_prob))
# #                 else:
# #                     non_microscopic_match.append((path, positive_prob))

# #             except Exception as e:
# #                 failed.append((path, str(e)))
# #                 print(f"FAILED | {path} | {e}")

# # microscopic_match_df = pd.DataFrame(microscopic_match)
# # microscopic_match_df.columns = ["image_path", "microscopic_prob"]
# # microscopic_match_df['image_path'] = microscopic_match_df['image_path'].astype(str)
# # microscopic_match_table = microscopic_match_df.merge(
# #     filtered_match_df,
# #     on="image_path",
# #     how="left"
# # )


# # microscopic_match_table.to_excel("outputs/data_exploration/microscopic_match_table.xlsx", index=False)