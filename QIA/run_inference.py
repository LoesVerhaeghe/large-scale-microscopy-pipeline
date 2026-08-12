import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from pathlib import Path 

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

def predict_full_image(model, image_np, device, val_transform=val_transform, tile_size=1024, overlap=64, num_classes=3):
    model.eval()

    stride = tile_size - overlap
    H, W, _ = image_np.shape

    prob_map = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            tile = image_np[y:y+tile_size, x:x+tile_size]

            h_tile, w_tile = tile.shape[:2]

            # pad if at border
            if h_tile < tile_size or w_tile < tile_size:
                pad_img = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                pad_img[:h_tile, :w_tile] = tile
                tile = pad_img

            # transform
            augmented = val_transform(image=tile)
            tile_tensor = augmented["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tile_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()

            probs = probs[:, :h_tile, :w_tile]

            prob_map[:, y:y+h_tile, x:x+w_tile] += probs
            count_map[y:y+h_tile, x:x+w_tile] += 1

    prob_map /= count_map # (3,H,W)
    #final_mask = np.argmax(prob_map, axis=0)
    # -----------------------------------
    # threshold-based classification
    # -----------------------------------

    bg_prob = prob_map[0]
    floc_prob = prob_map[1]
    filament_prob = prob_map[2]

    # start with background
    final_mask = np.zeros((H, W), dtype=np.uint8)

    floc_pixels = floc_prob >= 0.7814933
    filament_pixels = filament_prob >= 0.6403097
    

    # assign flocs
    final_mask[floc_pixels] = 1

    # assign filaments
    final_mask[filament_pixels] = 2

    # resolve pixels where both pass
    both = floc_pixels & filament_pixels

    final_mask[both & (floc_prob >= filament_prob)] = 1
    final_mask[both & (filament_prob > floc_prob)] = 2

    return prob_map, final_mask

# read match script
match_table_extended=pd.read_excel("data/microscopic_match_table_extended.xlsx")

# filter out phase contrast images and 100x images
classification_df=pd.read_csv("data/PhaseContrast Classifier/microscopyClassificationsV2_OCR_clean.csv")

ph_100_paths = classification_df.loc[
    (classification_df["classification_category"] == "Phase Contrast")
    & (classification_df["ocr_text_clean"] == "100 um"),
    "image_path"
].tolist()

#filter path from match tables
match_table_extended_ph100 = match_table_extended[
    match_table_extended["image_path"].isin(ph_100_paths)
].copy()

match_table_extended_ph100["mask_path_v2"] = (
    match_table_extended_ph100.index
    .map(lambda x: str(f"QIA/ph100_masks_v2/image_{x}.png"))
)
### inference 

torch.cuda.set_device(0) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(25)

model_path = 'models/trained_SegFormer_noVal.pt'
trained_SegFormer = torch.load(model_path, map_location=device)
trained_SegFormer.eval()

with torch.no_grad():
    for index, row in match_table_extended_ph100.iterrows():
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path_v2"])

        # Skip if input image does not exist
        if not image_path.exists():
            print(f"Skipping, image not found: {image_path}")
            continue

        # Skip if mask already exists
        if mask_path.exists():
            print(f"Skipping, mask already exists: {mask_path}")
            continue
        
        image = np.array(Image.open(image_path).convert("RGB"))

        prob_map, final_mask = predict_full_image(trained_SegFormer, 
                                    image, 
                                    device, 
                                    val_transform=val_transform, 
                                    tile_size=1024, 
                                    overlap=64, 
                                    num_classes=3)
        Image.fromarray(final_mask.astype(np.uint8), mode="L").save(row["mask_path_v2"])

match_table_extended_ph100.to_excel(
    "QIA/match_table_extended_ph100_masks_v2.xlsx",
    index=False
)