import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


# =========================
# Settings
# =========================

EXCEL_FILE = "data/captions2/fotobijlage_cleaned_checked.xlsx"
OUTPUT_DIR = Path("outputs/data_exploration/plot_captions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load data
# =========================

df = pd.read_excel(EXCEL_FILE)

print(f"Loaded {len(df)} rows.")


# =========================
# Interactive image viewer
# =========================

while True:

    user_input = input(
        f"\nEnter a row number (0-{len(df)-1}), or 'q' to quit: "
    )

    if user_input.lower() == "q":
        break

    try:
        idx = int(user_input)
    except ValueError:
        print("Please enter an integer.")
        continue

    if idx < 0 or idx >= len(df):
        print(f"Please enter a number between 0 and {len(df)-1}.")
        continue

    row = df.iloc[idx]

    image_path = row["image_path"]
    caption = row["image_caption"]
    classification = row["classification"]

    print(f"\nRow: {idx}")
    print(f"Image: {image_path}")
    print(f"Classification: {classification}")

    try:
        image = Image.open(image_path)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.imshow(image)
        ax.axis("off")

        # Title = classification
        ax.set_title(
            str(classification),
            fontsize=16,
            fontweight="bold"
        )

        # Caption below image
        fig.text(
            0.5,
            0.02,
            str(caption),
            ha="center",
            va="bottom",
            fontsize=16,
            wrap=True
        )
        # Leave more space for caption
        plt.tight_layout(rect=[0, 0.18, 1, 0.90])


        # Save figure
        output_path = OUTPUT_DIR / f"row_{idx}.png"
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Could not load image: {e}")