
import re
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "data/captions/captions.csv"
)

OUTPUT_CSV = Path(
    "data/captions/captions_filtered.csv"
)

REMOVED_CSV = Path(
    "data/captions/captions_removed.csv"
)


# ============================================================
# CAPTION FILTER
# ============================================================

def should_remove_caption(caption):
    """
    Return:
        True  -> remove this row
        False -> keep this row
    """

    if pd.isna(caption):
        return True

    caption = str(caption).strip()

    # --------------------------------------------------------
    # 1. Empty caption
    # --------------------------------------------------------

    if not caption:
        return True

    # --------------------------------------------------------
    # 2. Contains "idem" anywhere
    #    Case-insensitive
    # --------------------------------------------------------

    if re.search(r"\bidem\b", caption, re.IGNORECASE):
        return True

    # --------------------------------------------------------
    # 3. Caption must start with "Foto"
    #    Case-insensitive
    # --------------------------------------------------------

    if not re.match(r"^\s*foto\b", caption, re.IGNORECASE):
        return True

    # --------------------------------------------------------
    # Remove surrounding brackets for pattern matching
    #
    # Example:
    #     (Foto nr. 37)
    # becomes:
    #     Foto nr. 37
    # --------------------------------------------------------

    clean = caption.strip()

    if clean.startswith("(") and clean.endswith(")"):
        clean = clean[1:-1].strip()

    # --------------------------------------------------------
    # 4. Only a photo number/reference
    #
    # Examples removed:
    #
    # Foto 37
    # Foto nr 37
    # Foto nr. 37
    # Foto 37 -
    # Foto nr. 37 -
    # (Foto nr. 37)
    # --------------------------------------------------------

    photo_reference_only = re.fullmatch(
        r"""
        foto
        \s*
        (?:nr\.?\s*)?
        \d+
        \s*
        [-–—]?
        \s*
        """,
        clean,
        re.IGNORECASE | re.VERBOSE,
    )

    if photo_reference_only:
        return True

    # --------------------------------------------------------
    # 5. Only a photo number + SVD code
    #
    # Examples:
    #
    # Foto 5: SVD2516
    # Foto 4: SVD2512
    #
    # Case-insensitive
    # --------------------------------------------------------

    photo_svd_only = re.fullmatch(
        r"""
        foto
        \s*
        (?:nr\.?\s*)?
        \d+
        \s*:\s*
        SVD\d+
        \s*
        """,
        clean,
        re.IGNORECASE | re.VERBOSE,
    )

    if photo_svd_only:
        return True

    # --------------------------------------------------------
    # 6. Only microscope/sample codes
    #
    # Examples:
    #
    # Foto 1: 100XPH1, KVEY2414
    # Foto 2: 100XPH1, KVEY2419
    # Foto 4 : 200XHF, KVEY2180
    #
    # We treat these as codes rather than real descriptions.
    # --------------------------------------------------------

    photo_codes_only = re.fullmatch(
        r"""
        foto
        \s*
        (?:nr\.?\s*)?
        \d+
        \s*:\s*
        [A-Z0-9]+
        (?:\s*,\s*[A-Z0-9]+)*
        \s*
        """,
        clean,
        re.IGNORECASE | re.VERBOSE,
    )

    if photo_codes_only:
        return True

    # --------------------------------------------------------
    # Otherwise keep the caption
    # --------------------------------------------------------

    return False


# ============================================================
# LOAD CSV
# ============================================================

print(f"Reading: {INPUT_CSV}")

df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig",
)

print(f"Total rows: {len(df)}")


# ============================================================
# APPLY FILTER
# ============================================================

remove_mask = df["image_caption"].apply(should_remove_caption)

filtered_df = df[~remove_mask].copy()
removed_df = df[remove_mask].copy()


# ============================================================
# SAVE
# ============================================================

filtered_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)

removed_df.to_csv(
    REMOVED_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FILTERING COMPLETE")
print("=" * 70)

print(f"Original rows:     {len(df)}")
print(f"Removed rows:      {len(removed_df)}")
print(f"Remaining rows:    {len(filtered_df)}")

print()
print(f"Filtered CSV:")
print(f"  {OUTPUT_CSV}")

print()
print(f"Removed CSV:")
print(f"  {REMOVED_CSV}")
