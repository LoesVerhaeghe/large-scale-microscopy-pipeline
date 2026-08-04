import pandas as pd
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from pathlib import Path 
import shutil

val_transform = A.Compose([
    A.CenterCrop(height=512, width=512),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

def predict_512x512_image(model, image_np, device, val_transform=val_transform):
    model.eval()

    # transform
    augmented = val_transform(image=image_np)
    input_image = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    final_mask = np.argmax(probs, axis=0)

    return final_mask

def decode_mask(mask, COLORS):
    """Convert [H, W] class mask → RGB image"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask == cls] = color
    return rgb

COLORS = {
    0: [0, 0, 0],        # background
    1: [255, 0, 0],      # class 1 - red
    2: [0, 255, 0],      # class 2 - green
}

# read match script
match_table_extended=pd.read_excel("data/microscopic_match_table_extended.xlsx")

# filter out phase contrast images and 100x images
classification_df=pd.read_csv("data/PhaseContrast Classifier/microscopyClassificationsV2_OCR_clean.csv")

sample_df = classification_df.loc[
    (classification_df["classification_category"] == "Phase Contrast")
    & (classification_df["ocr_text_clean"] == "100 um")
].sample(n=50, random_state=42)

ph_100_paths = sample_df["image_path"].tolist()

### inference 

torch.cuda.set_device(3) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(25)

model_path = 'models/trained_SegFormer.pt'
trained_SegFormer = torch.load(model_path, map_location=device)
trained_SegFormer.eval()

output_dir = Path("QIA/annotate_images/to_finetune")

crop_transform = A.Compose([
    A.CenterCrop(height=512, width=512)
])


with torch.no_grad():
    for index, row in sample_df.iterrows():
        image_path = Path(row["image_path"])

        # Skip if input image does not exist
        if not image_path.exists():
            print(f"Skipping, image not found: {image_path}")
            continue

        image_out = output_dir / f"{index}_image.png"
        mask_out = output_dir / f"{index}_mask.png"
        
        image = np.array(Image.open(image_path).convert("RGB"))

        pred_np = predict_512x512_image(trained_SegFormer, 
                                    image, 
                                    device, 
                                    val_transform=val_transform)
        pred_rgb = decode_mask(pred_np, COLORS)


        # Save original but cropped image
        cropped = crop_transform(image=image)["image"]
        cropped_pil = Image.fromarray(cropped)
        cropped_pil.save(image_out)

        # Save predicted mask
        Image.fromarray(pred_rgb).save(mask_out)

