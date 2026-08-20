from pathlib import Path
import shutil
from docx import Document
import pandas as pd


df=pd.read_excel("data/microscopic_match_table_extended.xlsx")

output_folder = Path("outputs/data_exploration/check_advies_tekst_outputfolders")
output_folder.mkdir(parents=True, exist_ok=True)

# Select 30 random experiments (order_nrs)
random_orders = (
    df["order_nr"]
    .dropna()
    .drop_duplicates()
    .sample(n=30, random_state=42)
)


for order_nr, group in df[df["order_nr"].isin(random_orders)].groupby("order_nr"):

    # create folder for this experiment
    experiment_folder = output_folder / str(order_nr)
    experiment_folder.mkdir(exist_ok=True)

    print(f"Processing order {order_nr}")

    # -------------------------
    # Save advies tekst to Word
    # -------------------------

    doc = Document()

    # if multiple rows have text, combine them
    texts = group["advies_tekst"].dropna().unique()

    for text in texts:
        doc.add_paragraph(str(text))
        doc.add_paragraph("")  # empty line between texts

    doc_path = experiment_folder / "advies_tekst.docx"
    doc.save(doc_path)


    # -------------------------
    # Copy images
    # -------------------------

    for _, row in group.iterrows():

        image_path = Path(row["image_path"])

        if image_path.exists():

            destination = experiment_folder / image_path.name

            shutil.copy2(
                image_path,
                destination
            )

        else:
            print(f"Image not found: {image_path}")

print("Done!")