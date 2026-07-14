
from pathlib import Path
import pandas as pd
import os 

# read excel overview sheet
excel_path="data/Aquafin_data_cleaned/other_files/microscopie_compleet_overzicht (slims databank + oude access databank).xlsx"
overview_df = pd.read_excel(excel_path, sheet_name="Overzicht")

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

# check how many images we have 
data_path=Path("data/Aquafin data (Sorted_Nusret)")

files_sorted = get_all_files(data_path)
print(f"Total images found in sorted data folder: {len(files_sorted)}") # 44 219

# table that indexes all files (folders, loose images, files like doc and pdf)
data_path_all=Path("data/Aquafin_data_cleaned")

files_all = get_all_files(data_path_all)
print(f"Total files found in data folder: {len(files_all)}") 


# table that links files to the overview table, based on filename and folder structure, and add column for confidence of match

from PIL import Image, ExifTags
from datetime import datetime
from typing import List, Union
import io
import fitz
from docx import Document
import subprocess
import tempfile

TAGS = {v: k for k, v in ExifTags.TAGS.items()}
def get_image_date(path):
    try:
        img = Image.open(path)
        exif = img.getexif()

        if not exif:
            return None

        # Try different EXIF date fields in order of reliability
        for field in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
            tag_id = TAGS.get(field)
            if tag_id in exif:
                date_str = exif[tag_id]
                return date_str

        return None

    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
    except:
        return date_str

def get_pdf_creation_date(pdf):
    date = pdf.metadata.get("creationDate")

    if not date:
        return None

    # PDF format: D:YYYYMMDDHHmmSS...
    try:
        if isinstance(date, str) and date.startswith("D:"):
            date = date[2:]

        # keep only digit characters to be robust against variations
        date_digits = ''.join([c for c in str(date) if c.isdigit()])
        return datetime.strptime(date_digits[:14], "%Y%m%d%H%M%S")
    except Exception:
        return None

def get_docx_creation_date(doc):
    return doc.core_properties.created

def extract_images_from_pdfs_and_docs(pdfs, docs, identifier):
    """
    Extract images from PDF and DOCX files.
    """
    extracted = []

    for file_path in pdfs:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            pdf = fitz.open(file_path)
            creation_date = get_pdf_creation_date(pdf)
            if creation_date is None:
                creation_date = datetime.fromtimestamp(os.path.getctime(file_path))

            for page_index in range(len(pdf)):
                page = pdf[page_index]
                images = page.get_images(full=True)

                for img in images:
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    extracted.append({
                    "image": image,
                    "source_path": str(file_path),
                    "creation_date": creation_date
                })
            pdf.close()
    for file_path in docs:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix != ".docx":
            continue

        try:
            doc = Document(file_path)
            creation_date = get_docx_creation_date(doc)
            if creation_date is None:
                creation_date = datetime.fromtimestamp(os.path.getctime(file_path))

            for rel in doc.part.rels.values():
                target_ref = getattr(rel, 'target_ref', '')
                if target_ref and 'image' in str(target_ref).lower():
                    image_bytes = rel.target_part.blob
                    try:
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        extracted.append({
                            "image": image,
                            "source_path": str(file_path),
                            "creation_date": creation_date
                        })
                    except OSError as e:
                        print(f"Error occurred while processing image in {file_path}: {e}")

        except Exception as e:
            print(f"Error occurred while processing {file_path}: {e}")

    output_dir = Path("data/Aquafin_data_cleaned/extracted_images") / identifier
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    original_paths = []
    creation_dates = []

    for i, item in enumerate(extracted):
        save_path = output_dir / f"image_{i:04d}.png"
        item["image"].save(save_path)
        image_paths.append(str(save_path))
        original_paths.append(item["source_path"])
        creation_dates.append(item["creation_date"])
    return image_paths, original_paths, creation_dates

match_table = []
leftovers = []

for index, row in overview_df.iterrows():
    # extract relevant information from the overview table
    order_nr = str(row['order_nr'])
    labo_nr = str(row['labo_nummer'])
    sample_barcode = str(row['sample_barcode'])
    location = str(row['zuiveringsgebied'])
    experiment_date = pd.to_datetime(str(row['datum_monstername']), format="%Y-%m-%d %H:%M:%S")

    images = []
    pdfs = []
    docs = []
    others = []
    file_names = []

    for rel_path, full_path in files_all.items():
        file_name = os.path.basename(rel_path).lower()
        if file_name in ["thumbs.db", ".ds_store"]:
            continue
        if (order_nr in rel_path or labo_nr in rel_path or sample_barcode in rel_path):
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
        location_base = location.lower().split("-")[0].strip()
        location_score = 1 if location_base in str(img).lower() else 0

        image_date = get_image_date(img)
        if image_date:
            try:
                image_date_parsed = parse_date(image_date)
                day_diff = abs((image_date_parsed - experiment_date).days)
                time_score = max(0, 1 - (day_diff / 30))  # 30-day decay
            except Exception:
                day_diff = None
                time_score = 0.5
        else:
            day_diff = None
            time_score = 0.5  # default score if no date info
        confidence_score = (location_score * 0.5) + (time_score * 0.5) # simple weighted average

        match_type = None
        if order_nr in img:
            match_type = "order_nr"
        elif sample_barcode in img:
            match_type = "sample_barcode"
        elif labo_nr in img:
            match_type = "labo_nr"
        match_table.append({
                "order_nr": order_nr,
                "image_path": img,
                "confidence": confidence_score,
                "location_match": location_score,
                "day_diff": day_diff,
                "match_type": match_type
            })
        
    # if there are images, we consider those as matched files, otherwise we can try to extract images from pdfs and docs
    identifier = os.path.dirname(img)
    word_pdf_original_paths = []
    if not images:
        if pdfs or docs:
            print("extracting images from: ", pdfs, docs)
            images, word_pdf_original_paths, creation_dates = extract_images_from_pdfs_and_docs(pdfs, docs, identifier)

    images = list(dict.fromkeys(images)) # preserve order, remove duplicates

    if word_pdf_original_paths:
        for i, img in enumerate(images):
            location_base = location.lower().split("-")[0].strip()
            orig_path = word_pdf_original_paths[i].lower() if i < len(word_pdf_original_paths) else ''
            location_score = 1 if location_base in orig_path else 0

            image_date = creation_dates[i] if i < len(creation_dates) else None
            image_date = pd.Timestamp(image_date).tz_localize(None)
            experiment_date = pd.Timestamp(experiment_date).tz_localize(None)
            if isinstance(image_date, datetime):
                day_diff = abs((image_date - experiment_date).days)
                time_score = max(0, 1 - (day_diff / 30))  # 30-day decay
            else:
                day_diff = None
                time_score = 0.5  # default score if no date info
            confidence_score = (location_score * 0.5) + (time_score * 0.5) # simple weighted average

            match_type = None
            if order_nr in word_pdf_original_paths[i]:
                match_type = "order_nr"
            elif sample_barcode in word_pdf_original_paths[i]:
                match_type = "sample_barcode"
            elif labo_nr in word_pdf_original_paths[i]:
                match_type = "labo_nr"
            match_table.append({
                    "order_nr": order_nr,
                    "image_path": img,
                    "confidence": confidence_score,
                    "location_match": location_score,
                    "day_diff": day_diff,
                    "match_type": match_type
                })

match_df = pd.DataFrame(match_table)
leftovers_df = pd.DataFrame(leftovers)

match_df.to_excel("all_matches_table.xlsx", index=False)
leftovers_df.to_excel("leftovers.xlsx", index=False)

# # if the match type is based on labo_nr, but the location does not match, we can consider this a low confidence match and remove it from the final match table
filtered_match_df = match_df[
    ~(
        (match_df["match_type"] == "labo_nr") &
        (match_df["location_match"] == 0)
    )
]

filtered_match_df = filtered_match_df[
    ~(filtered_match_df["day_diff"] > 14)
]


#### classify matched images based on classifier model Nusret: microscopic vs non microscopic images
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from tqdm import tqdm

checkpoint_path = Path("data/Microscopic Image Classifier/weights/best_resnext101_microscopyX3.pt")

batch_size = 16
num_workers = 4
threshold = 0.5


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".psd", ".jp2"}

# --------------------------------------------------
# Dataset for unlabeled images
# --------------------------------------------------

class UnlabeledImageDataset(Dataset):
    def __init__(self, paths_to_images, transform=None):
        self.paths_to_images = paths_to_images
        self.transform = transform
        self.samples = []

        for path in paths_to_images:
            path = Path(path)
            if path.is_file() and path.suffix.lower() in valid_extensions:
                self.samples.append(path)

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.paths_to_images}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")

                if self.transform is not None:
                    img = self.transform(img)

            return img, str(path)

        except Exception as e:
            print(f"SKIPPING unreadable image: {path} | {e}")
            return None

def skip_bad_images_collate_fn(batch):
    ''' Custom collate function to skip None items returned by the dataset when an image fails to load.'''
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    images, paths = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, paths
    

# --------------------------------------------------
# Transform
# --------------------------------------------------
weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,),])


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

dataset = UnlabeledImageDataset(filtered_match_df['image_path'], transform=val_transform)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=skip_bad_images_collate_fn)

print(f"Images found: {len(dataset)}")


# --------------------------------------------------
# Load model
# --------------------------------------------------
model = resnext101_64x4d(weights=weights)
in_features = model.fc.in_features 
model.fc = nn.Linear(in_features, 1) # replace final layer for binary classification
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
print(f"Loaded model from: {checkpoint_path}")

microscopic_match = []
non_microscopic_match = []
failed = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Classifying"):
        if batch is None:
            continue
    
        images, paths = batch
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits)

        for path_str, prob in zip(paths, probs):
            path = Path(path_str)
            positive_prob = float(prob.item())
            try:
                if positive_prob >= threshold:
                    microscopic_match.append((path, positive_prob))
                else:
                    non_microscopic_match.append((path, positive_prob))

            except Exception as e:
                failed.append((path, str(e)))
                print(f"FAILED | {path} | {e}")

microscopic_match_df = pd.DataFrame(microscopic_match)
microscopic_match_df.columns = ["image_path", "microscopic_prob"]
microscopic_match_df['image_path'] = microscopic_match_df['image_path'].astype(str)
microscopic_match_table = microscopic_match_df.merge(
    filtered_match_df,
    on="image_path",
    how="left"
)


microscopic_match_table.to_excel("microscopic_match_table.xlsx", index=False)