from __future__ import annotations

import os
import re
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

from openpyxl import Workbook


# ============================================================
# CONFIGURATION
# ============================================================

SRC_ROOTS = [
    (
        Path("data/Aquafin_data_cleaned"),
        "Aquafin_data_cleaned",
    ),
    (
        Path("data/aquafin_converted_doc_to_docx"),
        "aquafin_converted_doc_to_docx",
    ),
]

# Change this to wherever you want the output.
OUTPUT_DIR = Path("data/captions2")

EXCEL_NAME = "fotobijlage_images.xlsx"
IMAGE_DIR_NAME = "images"

DOCX_NAME_CONTAINS = "fotobijlage"

# True  -> Excel contains absolute image paths
# False -> Excel contains paths relative to OUTPUT_DIR
IMAGE_PATHS_ABSOLUTE = True


# ============================================================
# XML NAMESPACES
# ============================================================

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}

W_P = f"{{{NS['w']}}}p"
W_T = f"{{{NS['w']}}}t"
W_DRAWING = f"{{{NS['w']}}}drawing"

A_BLIP = f"{{{NS['a']}}}blip"
PIC_CNVPR = f"{{{NS['pic']}}}cNvPr"


# ============================================================
# CAPTION REGEX
# ============================================================
# These regexes are used ONLY for parsing an already-recognized
# caption. They are NOT used to decide whether text is a caption.
#
# Caption recognition itself is simply:
#     starts with "Foto" or starts with "Image"
# ============================================================
IDEM_FOTO_RE = re.compile(
    r"idem\s+foto\s+(\d+)",
    re.IGNORECASE,
)


# ============================================================
# FILE DISCOVERY
# ============================================================

def get_all_files(folder_path: Path) -> dict[str, Path]:
    files = {}

    if not folder_path.exists():
        print(f"Warning: {folder_path} does not exist")
        return files

    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            full_path = Path(root) / filename
            rel_path = full_path.relative_to(folder_path)

            files[str(rel_path)] = full_path

    return files


def find_fotobijlage_entries(src_root: Path, root_label: str):
    """
    Find DOCX files recursively.

    No DOC conversion is performed.
    Only existing .docx files are processed.
    """

    entries = []

    for rel_str, full_path in get_all_files(src_root).items():

        if full_path.suffix.lower() != ".docx":
            continue

        if (
            DOCX_NAME_CONTAINS
            and DOCX_NAME_CONTAINS.lower() not in full_path.name.lower()
        ):
            continue

        entries.append(
            {
                "root_label": root_label,
                "rel_path": Path(rel_str),
                "full_path": full_path,
            }
        )

    return entries


# ============================================================
# DOCX XML HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """Collapse whitespace into single spaces."""

    return re.sub(r"\s+", " ", text).strip()


def parse_relationships(z: zipfile.ZipFile) -> dict[str, str]:
    """
    Read word/_rels/document.xml.rels.

    Returns:
        relationship ID -> media target
    """

    root = ET.fromstring(
        z.read("word/_rels/document.xml.rels")
    )

    relationships = {}

    for rel in root:

        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target")

        if rid and target:
            relationships[rid] = target

    return relationships


def get_document_root(z: zipfile.ZipFile):
    return ET.fromstring(
        z.read("word/document.xml")
    )


def get_paragraph_text(paragraph) -> str:
    """
    Extract visible Word text from w:t nodes.

    Important:
    - does NOT inspect alt text
    - does NOT inspect headers/footers
    """

    text = "".join(
        t.text or ""
        for t in paragraph.iter(W_T)
    )

    return normalize_text(text)


def get_image_object_name(drawing) -> str | None:
    """
    Get the embedded picture object's name.

    Example:
        Image_689.jpg

    It is used only for matching a visible caption such as:

        Image_689.jpg – Detail korte filamenten...
    """

    c_nv_pr = drawing.find(
        f".//{PIC_CNVPR}"
    )

    if c_nv_pr is None:
        return None

    return c_nv_pr.attrib.get("name")


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def collect_images(
    z: zipfile.ZipFile,
    document_root,
) -> list[dict]:
    """
    Collect all embedded image occurrences from document.xml
    in document order.

    Headers and footers are automatically excluded because we
    only inspect word/document.xml.

    Images inside tables are also found.
    """

    relationships = parse_relationships(z)

    body = document_root.find(
        "w:body",
        NS,
    )

    if body is None:
        return []

    images = []

    # iter() preserves XML/document order.
    for xml_position, drawing in enumerate(body.iter()):

        if drawing.tag != W_DRAWING:
            continue

        blip = drawing.find(
            f".//{A_BLIP}"
        )

        if blip is None:
            continue

        rid = blip.attrib.get(
            f"{{{NS['r']}}}embed"
        )

        if not rid:
            continue

        if rid not in relationships:
            continue

        target = relationships[rid]

        media_name = Path(target).name

        media_path = (
            "word/media/"
            + media_name
        )

        if media_path not in z.namelist():
            continue

        object_name = get_image_object_name(
            drawing
        )

        images.append(
            {
                "index": len(images) + 1,
                "rid": rid,
                "media_name": media_name,
                "media_path": media_path,
                "object_name": object_name,
                "xml_position": xml_position,
            }
        )

    return images


# ============================================================
# CAPTION EXTRACTION
# ============================================================

def extract_caption_candidates(document_root):
    """Extract visible Foto/Image caption candidates with document position."""
    body = document_root.find("w:body", NS)
    if body is None:
        return []
    candidates = []
    for xml_position, element in enumerate(body.iter()):
        if element.tag != W_P:
            continue
        text = normalize_text(get_paragraph_text(element))
        if not text:
            continue
        # ONLY caption filter: visible text starts with Foto or Image.
        if not (text.lower().startswith("foto") or text.lower().startswith("image")):
            continue
        starts = find_caption_starts(text)
        for i, start_pos in enumerate(starts):
            end_pos = starts[i + 1] if i + 1 < len(starts) else len(text)
            candidate = normalize_text(text[start_pos:end_pos])
            if candidate:
                candidates.append({"text": candidate, "xml_position": xml_position})
    return candidates


def find_caption_starts(text):
    """
    Find the starts of genuinely new Foto/Image captions inside one
    paragraph.

    IMPORTANT:
    This function is only for splitting a paragraph that has already
    passed the caption filter. It must NOT split ordinary references
    such as:

        beeld foto 3, SVD1203
        idem foto 1
        Foto nr. SVD860
        ..., Foto nr. SVD860, SVD858
        Foto 1 en foto 2: ...

    A later Foto caption is therefore recognised here only when it is
    written as a new, capitalised "Foto" followed by a number. This
    keeps references inside the description attached to the original
    caption.
    """

    starts = [0]
    position = 1

    while position < len(text):
        # --------------------------------------------------------
        # Look for a later capitalised "Foto" followed by a number.
        # We intentionally do NOT use case-insensitive matching here:
        # lowercase "foto 3" is normally a reference in the caption
        # text, not the start of a second caption.
        # --------------------------------------------------------
        foto_pos = text.find("Foto", position)

        # Look for a later capitalised Image_... caption as well.
        image_pos = text.find("Image_", position)

        possible = [
            pos for pos in (foto_pos, image_pos)
            if pos != -1
        ]

        if not possible:
            break

        start = min(possible)

        # --------------------------------------------------------
        # A new Foto caption must be:
        #   Foto <number>
        # and must be a separate word.
        # --------------------------------------------------------
        is_foto_start = False
        if text.startswith("Foto", start):
            after_foto = text[start + 4:]
            is_foto_start = bool(
                after_foto
                and after_foto[0].isspace()
                and len(after_foto.lstrip()) > 0
                and after_foto.lstrip()[0].isdigit()
            )

        # --------------------------------------------------------
        # A new Image caption must start with Image_ and have a
        # separate word boundary. The first token is the image name.
        # --------------------------------------------------------
        is_image_start = text.startswith("Image_", start)

        # A new caption can occasionally be concatenated directly to the
        # previous caption by Word (e.g. ``poelslakjesFoto 17: ...``).
        # For Foto/Image we therefore rely on the strong start pattern
        # itself, not on a preceding whitespace character.
        if is_foto_start or is_image_start:
            starts.append(start)

        position = start + 1

    return starts


# ============================================================
# FOTO NUMBER PARSING
# ============================================================

def parse_foto_numbers(number_part):
    """
    Parse photo numbers from text such as:

        1
        1 en 2
        2,3,4 en 5
        2, 3, 4 en 5
        1,2 en 3

    Returns a list of integers.
    """

    number_part = normalize_text(number_part)
    numbers = re.findall(r"\d+", number_part)

    return [int(n) for n in numbers]


def get_foto_number_part(candidate):
    """
    Get the part of a Foto caption that contains the photo numbers.

    This is PARSING only. It is not used to decide whether the
    candidate is a caption.

    Examples:
        Foto 1: something
            -> "1"

        Foto 1 en 2: something
            -> "1 en 2"

        Foto 2,3,4 en 5 – something
            -> "2,3,4 en 5"
    """

    remainder = candidate[4:].strip()

    separator_positions = []

    for separator in (":", "–", "—", "-"):
        position = remainder.find(separator)

        if position != -1:
            separator_positions.append(position)

    if separator_positions:
        return remainder[:min(separator_positions)]

    return remainder


# ============================================================
# BUILD CAPTION MAPS
# ============================================================

def build_caption_maps(caption_candidates):
    """Build Foto captions and positional Image captions."""
    foto_captions = {}
    image_captions = []
    warnings = []
    for item in caption_candidates:
        candidate = normalize_text(item["text"])
        if not candidate:
            continue
        lower = candidate.lower()
        if lower.startswith("foto"):
            number_part = get_foto_number_part(candidate)
            numbers = parse_foto_numbers(number_part)
            if not numbers:
                continue
            for number in numbers:
                if number in foto_captions and foto_captions[number] != candidate:
                    warnings.append(f"Conflicting captions for Foto {number}; using the last one.")
                foto_captions[number] = candidate
        elif lower.startswith("image"):
            # The visible Image_... name is NOT matched to any internal Word name.
            image_captions.append({
                "filename": candidate.split(None, 1)[0].strip().lower(),
                "caption": candidate,
                "xml_position": item["xml_position"],
            })
    return foto_captions, image_captions, warnings


# ============================================================
# RESOLVE "IDEM FOTO X"
# ============================================================

def resolve_idem_captions(
    foto_captions: dict[int, str],
    warnings: list[str],
):
    """
    Resolve ``idem foto X`` references to the actual caption of Foto X.

    Example:

        Foto 1: Detail van de vlokken
        Foto 2: idem foto 1

    Then Foto 2 is mapped to the caption of Foto 1, so the correct
    caption is used in Excel. References can be chained, e.g.
    Foto 3 -> idem foto 2 -> idem foto 1.
    """

    resolved = {}

    def resolve(number: int, stack: set[int]):
        if number in resolved:
            return resolved[number]

        if number not in foto_captions:
            return None

        if number in stack:
            warnings.append(
                f"Circular 'idem foto' reference involving Foto {number}."
            )
            return None

        caption = foto_captions[number]
        match = IDEM_FOTO_RE.search(caption)

        if not match:
            resolved[number] = caption
            return caption

        referenced_number = int(match.group(1))
        referenced_caption = resolve(
            referenced_number,
            stack | {number},
        )

        if referenced_caption is None:
            warnings.append(
                f"Foto {number} says 'idem foto {referenced_number}', "
                f"but Foto {referenced_number} has no caption."
            )
            # Keep the original caption if the reference cannot be resolved.
            resolved[number] = caption
            return caption

        # IMPORTANT: use the actual referenced caption.
        resolved[number] = referenced_caption
        return referenced_caption

    for number in foto_captions:
        resolve(number, set())

    return resolved


# ============================================================
# FILENAME HELPERS
# ============================================================

def sanitize_filename(
    name: str,
) -> str:
    """Make a safe Windows-compatible filename."""

    name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        name,
    )

    name = re.sub(
        r"\s+",
        "_",
        name,
    )

    name = name.strip(
        " ."
    )

    return name or "image"


# ============================================================
# PROCESS ONE DOCX
# ============================================================

def extract_and_create_rows(
    doc_entry: dict,
    output_dir: Path,
):
    """
    Process one DOCX.

    Returns:
        rows
        warnings
        number_of_images
        number_of_caption_candidates
    """

    doc_path = doc_entry[
        "full_path"
    ]

    doc_output_dir = (
        output_dir
        / IMAGE_DIR_NAME
        / sanitize_filename(
            doc_path.stem
        )
    )

    doc_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    warnings = []

    with zipfile.ZipFile(
        doc_path,
        "r",
    ) as z:

        document_root = (
            get_document_root(z)
        )

        images = collect_images(
            z,
            document_root,
        )

        caption_candidates = (
            extract_caption_candidates(
                document_root
            )
        )

        (
            foto_captions_raw,
            image_captions,
            caption_warnings,
        ) = build_caption_maps(
            caption_candidates
        )

        warnings.extend(
            caption_warnings
        )

        foto_captions = (
            resolve_idem_captions(
                foto_captions_raw,
                warnings,
            )
        )

        # ----------------------------------------------------
        # EXTRACT ALL EMBEDDED IMAGES
        # ----------------------------------------------------

        extracted_paths = {}

        for image in images:

            suffix = (
                Path(
                    image["media_name"]
                ).suffix.lower()
            )

            if not suffix:
                suffix = ".bin"

            output_name = (
                f"foto_{image['index']:03d}"
                f"{suffix}"
            )

            output_path = (
                doc_output_dir
                / output_name
            )

            with z.open(
                image["media_path"]
            ) as src, open(
                output_path,
                "wb",
            ) as dst:

                shutil.copyfileobj(
                    src,
                    dst,
                )

            extracted_paths[
                image["index"] - 1
            ] = output_path

        # ----------------------------------------------------
        # FOTO CAPTIONS
        # ----------------------------------------------------
        #
        # Foto 1 -> first image
        # Foto 2 -> second image
        # etc.
        #
        # If:
        #
        # Foto 2,3,4 en 5: caption
        #
        # the same caption goes to all four rows.
        #
        # Any extra uncaptioned images are NOT put into Excel.

        for photo_number, caption in sorted(
            foto_captions.items()
        ):

            image_index = (
                photo_number - 1
            )

            if (
                image_index < 0
                or image_index >= len(images)
            ):

                warnings.append(
                    f"Foto {photo_number} has a "
                    f"caption, but only "
                    f"{len(images)} image occurrences "
                    f"were found."
                )

                continue

            output_path = (
                extracted_paths[
                    image_index
                ]
            )

            if IMAGE_PATHS_ABSOLUTE:
                image_path_value = str(
                    output_path.resolve()
                )
            else:
                image_path_value = str(
                    output_path.relative_to(
                        output_dir
                    )
                )

            rows.append(
                {
                    "word_doc_path": str(
                        doc_path.resolve()
                    ),
                    "image_path": image_path_value,
                    "image_caption": caption,
                }
            )

        # ----------------------------------------------------
        # IMAGE CAPTIONS — positional matching only
        # ----------------------------------------------------
        # Visible Image_....jpg labels are not reliable identifiers for
        # the embedded object. We therefore match each Image caption to
        # the nearest unused embedded image in document order, preferring
        # the image immediately before the caption.
        used_image_indices = set()

        for image_caption in image_captions:
            caption_position = image_caption["xml_position"]

            preceding = [
                (image["xml_position"], idx)
                for idx, image in enumerate(images)
                if image["xml_position"] < caption_position
                and idx not in used_image_indices
            ]
            following = [
                (image["xml_position"], idx)
                for idx, image in enumerate(images)
                if image["xml_position"] >= caption_position
                and idx not in used_image_indices
            ]

            if preceding:
                _, image_index = max(preceding, key=lambda x: x[0])
            elif following:
                _, image_index = min(following, key=lambda x: x[0])
            else:
                warnings.append(
                    f"Image caption '{image_caption['filename']}' could not be matched to an embedded image by document position."
                )
                continue

            used_image_indices.add(image_index)
            output_path = extracted_paths[image_index]
            image_path_value = (
                str(output_path.resolve())
                if IMAGE_PATHS_ABSOLUTE
                else str(output_path.relative_to(output_dir))
            )
            rows.append({
                "word_doc_path": str(doc_path.resolve()),
                "image_path": image_path_value,
                "image_caption": image_caption["caption"],
            })

    return (
        rows,
        warnings,
        len(images),
        len(caption_candidates),
    )


# ============================================================
# WRITE EXCEL
# ============================================================

def write_excel(
    rows: list[dict],
    output_path: Path,
):
    """Create the final Excel file."""

    wb = Workbook()

    ws = wb.active
    ws.title = "images"

    headers = [
        "word_doc_path",
        "image_path",
        "image_caption",
    ]

    ws.append(headers)

    for row in rows:

        ws.append(
            [
                row["word_doc_path"],
                row["image_path"],
                row["image_caption"],
            ]
        )

    ws.freeze_panes = "A2"

    ws.auto_filter.ref = (
        ws.dimensions
    )

    # Excel column widths
    ws.column_dimensions["A"].width = 70
    ws.column_dimensions["B"].width = 70
    ws.column_dimensions["C"].width = 100

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wb.save(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # FIND DOCX FILES
    # --------------------------------------------------------

    fotobijlage_entries = []

    for (
        src_root,
        root_label,
    ) in SRC_ROOTS:

        fotobijlage_entries.extend(
            find_fotobijlage_entries(
                src_root,
                root_label,
            )
        )

    fotobijlage_entries.sort(
        key=lambda x: str(
            x["full_path"]
        ).lower()
    )

    print(
        "Number of DOCX files found:",
        len(fotobijlage_entries),
    )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    all_rows = []

    total_images = 0
    total_captions = 0

    all_warnings = []

    for entry in fotobijlage_entries:

        print()
        print(
            "Processing:",
            entry["full_path"],
        )

        try:

            (
                rows,
                warnings,
                image_count,
                caption_count,
            ) = extract_and_create_rows(
                entry,
                OUTPUT_DIR,
            )

            all_rows.extend(rows)

            total_images += image_count
            total_captions += caption_count

            print(
                "  Embedded images:",
                image_count,
            )

            print(
                "  Caption candidates:",
                caption_count,
            )

            print(
                "  Excel rows:",
                len(rows),
            )

            for warning in warnings:

                message = (
                    f"{entry['full_path']}: "
                    f"{warning}"
                )

                all_warnings.append(
                    message
                )

                print(
                    "  WARNING:",
                    warning,
                )

        except Exception as exc:

            message = (
                f"{entry['full_path']}: "
                f"ERROR: {exc}"
            )

            all_warnings.append(
                message
            )

            print(
                "  ERROR:",
                exc,
            )

    # --------------------------------------------------------
    # WRITE EXCEL
    # --------------------------------------------------------

    excel_path = (
        OUTPUT_DIR
        / EXCEL_NAME
    )

    write_excel(
        all_rows,
        excel_path,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)

    print(
        "DOCX files processed:",
        len(fotobijlage_entries),
    )

    print(
        "Embedded image occurrences:",
        total_images,
    )

    print(
        "Caption candidates:",
        total_captions,
    )

    print(
        "Excel rows:",
        len(all_rows),
    )

    print(
        "Excel output:",
        excel_path.resolve(),
    )

    # --------------------------------------------------------
    # SAVE WARNINGS
    # --------------------------------------------------------

    if all_warnings:

        warning_path = (
            OUTPUT_DIR
            / "extraction_warnings.txt"
        )

        warning_path.write_text(
            "\n".join(all_warnings),
            encoding="utf-8",
        )

        print(
            "Warnings:",
            warning_path.resolve(),
        )


if __name__ == "__main__":
    main()