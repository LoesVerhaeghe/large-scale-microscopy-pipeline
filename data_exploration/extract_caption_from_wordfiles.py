"""
Batch-extract embedded images and their captions from a folder of Word
documents. Assumes all inputs are ALREADY .docx.

- Walks each document's body (including inside tables) in reading order.
- For every embedded image, tries several caption strategies in order:
  same paragraph, next paragraph(s), previous paragraph(s). Records which one worked, and
  flags anything uncertain via needs_review.
- Resolves "idem" / "idem foto N" / "idem boven" captions by copying over
  the referenced image's already-resolved caption, propagating uncertainty
  through idem chains.
- Only looks at the main document body, so images that live in headers/footers
  (company logos, letterhead) are already excluded.
- Hashes every image across the whole corpus; anything that recurs
  identically across many documents (default: 5+ docs, or 5%+ of the corpus)
  is treated as boilerplate (e.g. a logo sitting in the body) and excluded.

"""
import csv
import hashlib
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import docx
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
import unicodedata
from io import BytesIO


## import for microscopy classification
from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    resnext101_64x4d,
    ResNeXt101_64X4D_Weights,
)
from tqdm import tqdm


###### get docx files from both source folders


def get_all_files(folder_path):
    """Recursively get all files in a folder, keyed by path relative to folder_path."""
    files = {}
    if not folder_path.exists():
        print(f"Warning: {folder_path} does not exist")
        return files

    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            full_path = Path(root) / filename
            rel_path = full_path.relative_to(folder_path)
            files[str(rel_path)] = full_path
    return files


def find_fotobijlage_entries(src_root, root_label):
    """
    Returns a list of dicts: {root_label, rel_path (Path, relative to src_root,
    including subfolders), full_path (Path, absolute/original location)}.
    """
    files = get_all_files(src_root)
    entries = []
    for rel_str, full_path in files.items():
        if full_path.suffix.lower() == ".docx" and "fotobijlage" in full_path.name.lower():
            entries.append({
                "root_label": root_label,
                "rel_path": Path(rel_str),
                "full_path": full_path,
            })
    return entries


src_root = Path("data/Aquafin_data_cleaned")
src_root_doc = Path("data/aquafin_converted_doc_to_docx")

fotobijlage_entries = (
    find_fotobijlage_entries(src_root, "Aquafin_data_cleaned")
    + find_fotobijlage_entries(src_root_doc, "aquafin_converted_doc_to_docx")
)

print("Number of DOCX files containing 'fotobijlage':", len(fotobijlage_entries))

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = "data/captions"
BOILERPLATE_MIN_COUNT = 5
BOILERPLATE_MIN_RATIO = 0.05

WEAK_SOURCES = {"previous_paragraph"}


IDEM_NUM_RE = re.compile(r"^\s*idem\.?\s*(?:als\s+)?foto\.?\s*(\d+)", re.IGNORECASE)
IDEM_GENERIC_RE = re.compile(r"^\s*idem\b", re.IGNORECASE)
FOTO_NUM_RE = re.compile(r"\bfoto\.?\s*(\d+)", re.IGNORECASE)
FOTO_PREFIX_RE = re.compile(r"^\s*foto\.?\s*\d+\s*[:.]?\s*", re.IGNORECASE)

FOTO_GROUP_RE = re.compile(
    r"\bfoto(?:'s)?\.?\s*(?:nr\.?\s*)?"
    r"(\d+(?:\s*(?:,|en|&)\s*\d+)*)",
    re.IGNORECASE,
)

CLASSIFIER_CHECKPOINT = Path(
    "data/Microscopic Image Classifier/weights/best_resnext101_microscopyX3.pt"
)

CLASSIFIER_THRESHOLD = 0.5
CLASSIFIER_BATCH_SIZE = 16
CLASSIFIER_NUM_WORKERS = 4

################

def extract_caption_photo_numbers(text):
    """
    Extract photo numbers from captions such as:
        Foto 2 en 3: ...
        Foto 2, 3 en 4: ...
        Foto nr. 2 en 3: ...

    Does NOT interpret things like:
        Foto nr. SVD784 en SVD785
    as photo numbers.
    """
    if not text:
        return []

    match = FOTO_GROUP_RE.search(text)

    if not match:
        return []

    numbers = re.findall(r"\d+", match.group(1))
    return [int(n) for n in numbers]

def propagate_multi_photo_captions(items):
    """
    If a caption explicitly refers to multiple photos, assign that caption
    to all referenced photos.

    Example:
        item 2 -> "Foto 2 en 3: Vermoedelijke defosfateerders..."
    
    Result:
        item 2 -> same caption
        item 3 -> same caption
    """

    num_to_pos = {
        item["index"]: pos
        for pos, item in enumerate(items)
    }

    # Keep the original captions first.
    original_captions = [
        {
            "caption": item["caption"],
            "caption_source": item["caption_source"],
        }
        for item in items
    ]

    for pos, item in enumerate(items):
        caption = original_captions[pos]["caption"]

        if not caption:
            continue

        photo_numbers = extract_caption_photo_numbers(caption)

        # Only relevant when more than one photo is explicitly mentioned
        if len(photo_numbers) < 2:
            continue

        for photo_number in photo_numbers:
            target_pos = num_to_pos.get(photo_number)

            if target_pos is None:
                continue

            # Don't overwrite a stronger, explicitly assigned caption
            # with something coming from another image.
            target_item = items[target_pos]

            if target_pos == pos:
                continue

            if not target_item["caption"] or target_item["caption"].lower().startswith("idem"):
                target_item["caption"] = caption
                target_item["caption_source"] = (
                    f"multi_photo:{item['index']}"
                )

def clean_caption(text):
    """Clean common Word/conversion artefacts in extracted captions."""
    if not text:
        return ""

    text = str(text)

    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)

    # Fix detached diaeresis/circumflex characters.
    # Examples:
    #   ¨e -> ë
    #   ¨a -> ä
    #   ^e -> ê
    #   e¨ -> ë
    #   e^ -> ê
    replacements = {
        "¨a": "ä", "¨e": "ë", "¨i": "ï", "¨o": "ö", "¨u": "ü",
        "¨A": "Ä", "¨E": "Ë", "¨I": "Ï", "¨O": "Ö", "¨U": "Ü",

        "^a": "â", "^e": "ê", "^i": "î", "^o": "ô", "^u": "û",
        "^A": "Â", "^E": "Ê", "^I": "Î", "^O": "Ô", "^U": "Û",

        "a¨": "ä", "e¨": "ë", "i¨": "ï", "o¨": "ö", "u¨": "ü",
        "A¨": "Ä", "E¨": "Ë", "I¨": "Ï", "O¨": "Ö", "U¨": "Ü",

        "a^": "â", "e^": "ê", "i^": "î", "o^": "ô", "u^": "û",
        "A^": "Â", "E^": "Ê", "I^": "Î", "O^": "Ô", "U^": "Û",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Also handle a space between accent and letter:
    # ¨ e -> ë, ^ e -> ê
    text = re.sub(
        r"¨\s*([aeiouAEIOU])",
        lambda m: replacements.get("¨" + m.group(1), m.group(0)),
        text,
    )

    text = re.sub(
        r"\^\s*([aeiouAEIOU])",
        lambda m: replacements.get("^" + m.group(1), m.group(0)),
        text,
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def strip_own_foto_prefix(text):
    return FOTO_PREFIX_RE.sub("", text, count=1).strip()
# ---------------------------------------------------------------------------


def iter_block_items(parent):
    parent_elm = parent.element.body if isinstance(parent, _Document) else parent._tc
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def iter_paragraphs_recursive(parent):
    for block in iter_block_items(parent):
        if isinstance(block, Paragraph):
            yield block
        else:
            for row in block.rows:
                for cell in row.cells:
                    yield from iter_paragraphs_recursive(cell)


def paragraph_has_image(paragraph) -> bool:
    return paragraph._p.find(".//" + qn("a:blip")) is not None


def get_images_from_paragraph(paragraph, document):
    blips = paragraph._p.findall(".//" + qn("a:blip"))
    out = []
    for blip in blips:
        r_id = blip.get(qn("r:embed"))
        if not r_id:
            continue
        image_part = document.part.related_parts.get(r_id)
        if image_part is None:
            continue
        ext = image_part.partname.ext or "png"
        out.append((image_part.blob, ext))
    return out


def collect_forward(paragraphs, start, n):
    j = start
    while j < n and not paragraphs[j].text.strip() and not paragraph_has_image(paragraphs[j]):
        j += 1
    if j >= n or paragraph_has_image(paragraphs[j]):
        return ""
    parts = []
    while j < n and paragraphs[j].text.strip() and not paragraph_has_image(paragraphs[j]):
        parts.append(paragraphs[j].text.strip())
        j += 1
    return " ".join(parts).strip()


def collect_backward(paragraphs, start):
    j = start
    while j >= 0 and not paragraphs[j].text.strip() and not paragraph_has_image(paragraphs[j]):
        j -= 1
    if j < 0 or paragraph_has_image(paragraphs[j]):
        return ""
    parts = []
    while j >= 0 and paragraphs[j].text.strip() and not paragraph_has_image(paragraphs[j]):
        parts.append(paragraphs[j].text.strip())
        j -= 1
    parts.reverse()
    return " ".join(parts).strip()


def extract_images_and_captions(docx_path: Path):
    document = docx.Document(str(docx_path))
    paragraphs = list(iter_paragraphs_recursive(document))
    n = len(paragraphs)

    results = []
    img_counter = 0
    for i, p in enumerate(paragraphs):
        if not paragraph_has_image(p):
            continue
        found = get_image_from_paragraph(p, document)
        if not found:
            continue
        image_bytes, ext = found
        img_counter += 1

        caption, source = "", "none"

        same_para_text = clean_caption(p.text.strip())
        if same_para_text:
            caption, source = same_para_text, "same_paragraph"

        if not caption:
            fwd = clean_caption(collect_forward(paragraphs, i + 1, n))
            if fwd:
                caption, source = fwd, "next_paragraph"

        if not caption:
            bwd = clean_caption(collect_backward(paragraphs, i - 1))
            if bwd:
                caption, source = bwd, "previous_paragraph"

        results.append({
            "index": img_counter,
            "image_bytes": image_bytes,
            "ext": ext,
            "caption": caption,
            "caption_source": source,
        })
    return results


def resolve_idem_references(items):
    num_to_pos = {}
    for pos, item in enumerate(items):
        m = FOTO_NUM_RE.match(item["caption"].strip()) or FOTO_NUM_RE.search(item["caption"])
        if m:
            num_to_pos.setdefault(int(m.group(1)), pos)

    for item in items:
        item["_weak"] = (not item["caption"]) or (item["caption_source"] in WEAK_SOURCES)

    def is_idem(text):
        remainder = strip_own_foto_prefix(text.strip())
        return bool(IDEM_GENERIC_RE.match(remainder)) if remainder else False

    for _ in range(len(items) + 1):
        changed = False
        for pos, item in enumerate(items):
            text = item["caption"].strip()
            if not is_idem(text):
                continue

            remainder = strip_own_foto_prefix(text)
            m_num = IDEM_NUM_RE.match(remainder)
            target_pos = None
            if m_num:
                target_pos = num_to_pos.get(int(m_num.group(1)))
            else:
                if pos > 0:
                    target_pos = pos - 1

            if target_pos is None or target_pos == pos:
                continue

            target_caption = items[target_pos]["caption"].strip()
            if not target_caption or is_idem(target_caption):
                continue

            item["caption"] = target_caption
            item["caption_source"] = f"idem_resolved:{items[target_pos]['index']}"
            item["_weak"] = items[target_pos]["_weak"]
            changed = True
        if not changed:
            break

    for item in items:
        if item["caption"].strip() and is_idem(item["caption"]):
            item["caption"] = ""
            item["caption_source"] = "idem_unresolved"

#### microscopic image classifier

def load_microscopy_classifier():
    """Load the ResNeXt microscopy classifier."""

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1

    model = resnext101_64x4d(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    checkpoint = torch.load(
        CLASSIFIER_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    print(f"Loaded microscopy classifier from: {CLASSIFIER_CHECKPOINT}")
    print(f"Classifier device: {device}")

    return model, transform, device
def classify_extracted_images(items, model, transform, device):
    """
    Classify extracted images directly from their image bytes.

    Adds:
        microscopic_prob
        is_microscopic
    """

    results = []

    for start in range(0, len(items), CLASSIFIER_BATCH_SIZE):
        batch_items = items[start:start + CLASSIFIER_BATCH_SIZE]

        tensors = []
        valid_items = []

        for item in batch_items:
            try:
                with Image.open(BytesIO(item["image_bytes"])) as img:
                    img = img.convert("RGB")
                    tensor = transform(img)

                tensors.append(tensor)
                valid_items.append(item)

            except Exception as e:
                print(
                    f"Classifier failed for "
                    f"foto {item['index']}: {e}"
                )

                item["microscopic_prob"] = None
                item["is_microscopic"] = False
                item["classifier_error"] = str(e)

        if not tensors:
            continue

        images = torch.stack(tensors).to(
            device,
            non_blocking=True,
        )

        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits).flatten()

        for item, prob in zip(valid_items, probs):
            prob = float(prob.item())

            item["microscopic_prob"] = prob
            item["is_microscopic"] = (
                prob >= CLASSIFIER_THRESHOLD
            )
            item["classifier_error"] = ""

            results.append(item)

    return items



def process_corpus(entries, output_dir: str):
    """
    entries: list of {root_label, rel_path (Path incl. subfolders, relative to
             its source root), full_path (Path, actual file on disk)}.
    output_dir: base folder to mirror the structure into.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(entries)} document(s)")

    classifier_model, classifier_transform, classifier_device = (
        load_microscopy_classifier()
    )

    all_extracted = {}          # output_key -> {"items": [...], "entry": entry}
    hash_doc_count = defaultdict(set)
    failed = []
    seen_output_dirs = {}

    for k, entry in enumerate(entries, 1):
        doc_path = entry["full_path"]
        # Mirror original structure: <output_dir>/<root_label>/<rel_path_without_ext>/
        leaf_dir = output_dir / entry["root_label"] / entry["rel_path"].parent / entry["rel_path"].stem
        output_key = str(leaf_dir.relative_to(output_dir))

        if output_key in seen_output_dirs:
            msg = f"duplicate output path, skipped (collides with {seen_output_dirs[output_key]})"
            print(f"[{k}/{len(entries)}] FAILED on {doc_path}: {msg}")
            failed.append((str(doc_path), msg))
            continue
        seen_output_dirs[output_key] = str(doc_path)

        try:
            items = extract_images_and_captions(doc_path)
            propagate_multi_photo_captions(items)
            resolve_idem_references(items)

            for item in items:
                item["hash"] = hashlib.md5(item["image_bytes"]).hexdigest()
                hash_doc_count[item["hash"]].add(output_key)

            all_extracted[output_key] = {"items": items, "entry": entry}
            print(f"[{k}/{len(entries)}] {doc_path.name}: {len(items)} image(s)")
        except Exception as e:
            print(f"[{k}/{len(entries)}] FAILED on {doc_path}: {e}")
            failed.append((str(doc_path), str(e)))

    n_docs = max(len(all_extracted), 1)
    boilerplate_hashes = {
        h for h, docs in hash_doc_count.items()
        if len(docs) >= BOILERPLATE_MIN_COUNT and len(docs) / n_docs >= BOILERPLATE_MIN_RATIO
    }
    print(f"\nAuto-detected {len(boilerplate_hashes)} boilerplate image(s) repeated across many docs (excluded).\n")

    captions_csv = output_dir / "captions.csv"
    excluded_csv = output_dir / "excluded_boilerplate.csv"
    failed_csv = output_dir / "failed.csv"

    n_flagged = 0
    with open(captions_csv, "w", newline="", encoding="utf-8-sig") as f_cap, \
         open(excluded_csv, "w", newline="", encoding="utf-8-sig") as f_excl, \
         open(failed_csv, "w", newline="", encoding="utf-8-sig") as f_fail:

        cap_writer = csv.writer(f_cap)
        cap_writer.writerow([
            "word_doc_path", "image_path", "image_caption",
            "caption_source", "microscopic_prob", "needs_review",
        ])

        excl_writer = csv.writer(f_excl)
        excl_writer.writerow(["source_root", "source_relpath", "original_position", "reason"])

        fail_writer = csv.writer(f_fail)
        fail_writer.writerow(["file", "error"])
        for name, err in failed:
            fail_writer.writerow([name, err])

        for output_key, bundle in all_extracted.items():
            items = bundle["items"]
            entry = bundle["entry"]
            leaf_dir = output_dir / output_key

            filtered_items = []
            # 1. Remove boilerplate
            for item in items:
                if item["hash"] in boilerplate_hashes:
                    excl_writer.writerow([
                        entry["root_label"],
                        str(entry["rel_path"]),
                        item["index"],
                        "boilerplate (repeats across many docs)",
                    ])
                else:
                    filtered_items.append(item)

            # 2. Microscopy classifier
            classify_extracted_images(
                filtered_items,
                classifier_model,
                classifier_transform,
                classifier_device,
            )

            # 3. Remove non-microscopic images
            microscopic_items = []

            for item in filtered_items:
                if not item.get("is_microscopic", False):
                    excl_writer.writerow([
                        entry["root_label"],
                        str(entry["rel_path"]),
                        item["index"],
                        (
                            "non-microscopic "
                            f"(microscopic_prob="
                            f"{item.get('microscopic_prob')})"
                        ),
                    ])
                    continue
                microscopic_items.append(item)

            # 4. Save only microscopic images + captions

            saved = 0

            for item in microscopic_items:
                saved += 1
                leaf_dir.mkdir(parents=True, exist_ok=True)
                filename = f"foto{saved}.{item['ext']}"
                with open(leaf_dir / filename, "wb") as f_img:
                    f_img.write(item["image_bytes"])

                needs_review = (not item["caption"]) or item.get("_weak", False) or (item["caption_source"] == "idem_unresolved")
                if needs_review:
                    n_flagged += 1
                image_path = leaf_dir / filename
                cap_writer.writerow([
                    str(entry["full_path"].resolve()),
                    str(image_path.resolve()),
                    item["caption"],
                    item["caption_source"],
                    item.get("microscopic_prob"),
                    needs_review,
                ])


    print("Done.")
    print(f"  Images + folders : {output_dir}")
    print(f"  Captions manifest: {captions_csv}  ({n_flagged} row(s) flagged needs_review=True)")
    print(f"  Excluded log     : {excluded_csv}")
    print(f"  Failures log     : {failed_csv}")


if __name__ == "__main__":
    process_corpus(fotobijlage_entries, OUTPUT_DIR)