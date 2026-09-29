"""
Script to remove useless captions from the dataset. Script was already cleaned by Nusret prior to this.

"""
"""
Classify microscopy captions in an Excel file as 'delete', 'keep', or 'check'.

Conservative by design: a row is only ever marked 'delete' when the caption
is PURELY boilerplate (sample type + magnification + technique, nothing else).
Anything with real observational content, or anything the heuristic isn't
sure about, is marked 'keep' or 'check' instead of 'delete'.

HOW TO USE
----------
1. Edit the three settings below (INPUT_FILE, SHEET_NAME, CAPTION_COLUMN).
2. Run:  python classify_captions.py
3. Open the output file, sort/filter by the new column, and:
     - spot-check a sample of 'delete' rows (should be safe to remove)
     - read through the 'check' rows and manually decide keep/delete
     - 'keep' rows are left alone
4. Once you're happy, filter out the 'delete' rows for your training set.
"""

import re
import sys
import pandas as pd

# ============ EDIT THESE THREE LINES ============
INPUT_FILE = "data/captions2/fotobijlage_cleaned.xlsx"        # path to your Excel file
SHEET_NAME = 0                      # sheet name (string) or index (0 = first sheet)
CAPTION_COLUMN = "image_caption"          # exact column header that holds the caption text
# ==================================================

OUTPUT_FILE = "data/captions2/fotobijlage_cleaned_tocheck.xlsx"

# ---- Vocabulary that signals REAL observational content (-> keep) ----
SIGNAL_PATTERNS = [
    r"\bzeer veel\b", r"\bfrequent\b", r"\bveelvuldig\b", r"\baanwezigheid\b",
    r"\bwaargenomen\b", r"\bvermoedelijke?\b", r"\bongeïdentificeerd\w*\b",
    r"\bonidentificeerbaar\w*\b", r"\bniet identificeerbaar\w*\b",
    r"\bstructuur\b", r"\bstructuren\b", r"\bfilament\w*\b", r"\bvlok\w*\b",
    r"\bdeflocculatie\w*\b", r"\bzetmeelkorrels?\b", r"\bvezels?\b",
    r"\bvegetatieve resten\b", r"\bvetdruppel\w*\b", r"\bolieachtige\b",
    r"\bgistcel\w*\b", r"\bcelcluster\w*\b", r"\btetrad\w*\b",
    r"\bbacterie\w*\b", r"\bbacteriën\b", r"\bgroepjes\b", r"\bgekorrelde\b",
    r"\bagglomerat\w*\b", r"\bmeetbaar\b",
    r"\bphb\b", r"type\s*\d+",
    r"\bfungi\b", r"\bspp\.",
    r"\bciliaa?t\w*\b", r"\bflagellaa?t\w*\b", r"\bamoebe\w*\b", r"\brotifeer\w*\b",
    r"\bdraadvormer\w*\b", r"\bmicrothrix\w*\b", r"\bparvicella\w*\b", r"\bgedispergeerd\w*\b", r"\banorganisch\w*\b",
    r"\bwatervloo?i?e?n?\b", r"\bdaphnia\w*\b", r"\bkreeftje\w*\b", r"\blarve\w*\b",r"\bnaald\w*\b",r"\bkiezelwier\w*\b",
    r"\bcocci\w*\b", r"\bovoïde\b", r"\bkristallijn\w*\b", r"\bzandkorrel\w*\b", r"\bmonocolonie\w*\b",
    r"\bzwavel\w*\b", r"\bnucleus\b", r"\bbloedcel\w*\b", r"\bzo(?:ö|o)glea\w*\b", r"\beenoogkreeftje\w*\b",
]
SIGNAL_RE = re.compile("|".join(SIGNAL_PATTERNS), re.IGNORECASE)

# ---- Vocabulary that is purely procedural / boilerplate ----
SAMPLE_NOUNS = (
    r"bezonken|materiaal|influent(?:staal|mengsel)?|effluent(?:staal)?|actief slib|slib(?:staal)?|"
    r"centraat(?:water)?|collectorstaal|bezinksel|schuimlaag|drijflaag|vuilwaterput|condensaat|"
    r"gemengde staal|staal|mengsel|lozingspunt|opschudden"
)
TECH_NOUNS = (
    r"vergroting|fasecontrast|helderveld|polarisator|neisserkleuring|gramkleuring|sudan\s*kleuring|"
    r"natte preparaat|met polarisator|zonder polarisator"
)
BOILERPLATE_RE = re.compile(
    r"^(algemeen\s+)?(microscopisch\s+)?beeld\s+(van\s+)?(het|de)?\s*",
    re.IGNORECASE,
)


def classify(caption) -> str:
    text = "" if pd.isna(caption) else str(caption).strip()
    if not text:
        return "check"

    # Strong signals of real content -> KEEP, never delete these
    if SIGNAL_RE.search(text):
        return "keep"

    # Strip the generic opening phrase
    stripped = BOILERPLATE_RE.sub("", text)

    # Remove parenthetical technical metadata, e.g. (100x vergroting, fasecontrast, -1591)
    stripped_no_paren = re.sub(r"\([^)]*\)", " ", stripped)

    # Remove known sample-type / technique words and light connective glue
    residual = re.sub(SAMPLE_NOUNS, " ", stripped_no_paren, flags=re.IGNORECASE)
    residual = re.sub(TECH_NOUNS, " ", residual, flags=re.IGNORECASE)
    residual = re.sub(
        r"\b(in|van|bij|de|het|een|met|zonder|na|uit|aan|genomen|punt|water|voor|op)\b",
        " ", residual, flags=re.IGNORECASE,
    )
    residual_clean = re.sub(r"[\-–:,.]", " ", residual)
    residual_clean = re.sub(r"\b\d+\b", " ", residual_clean)
    residual_clean = re.sub(r"\bmc\s*cain\b", " ", residual_clean, flags=re.IGNORECASE)
    residual_clean = residual_clean.strip()

    if residual_clean == "":
        return "delete"

    # Something unexplained is left over -> be conservative, flag for human review
    return "check"


def main():
    try:
        df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    except FileNotFoundError:
        sys.exit(f"Could not find '{INPUT_FILE}'. Check the path/filename.")

    if CAPTION_COLUMN not in df.columns:
        sys.exit(
            f"Column '{CAPTION_COLUMN}' not found. Available columns: {list(df.columns)}"
        )

    df["classification"] = df[CAPTION_COLUMN].apply(classify)

    counts = df["classification"].value_counts().to_dict()
    print(f"Classified {len(df)} rows: {counts}")

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()