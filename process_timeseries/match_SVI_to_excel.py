import pandas as pd
from pathlib import Path
import os
import time
import re

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


## read investtigation excel
# read excel overview sheet
excel_path="data/microscopic_match_table_extended.xlsx"
microscopic_match_table = pd.read_excel(excel_path)

# unique order nrs that can be linked to images in the match_table
matched_orders = set(microscopic_match_table["order_nr"]) 

# read excel overview sheet
excel_path="data/Aquafin_data_cleaned/other_files/microscopie_compleet_overzicht (slims databank + oude access databank).xlsx"
overview_df = pd.read_excel(excel_path, sheet_name="Overzicht")


overview_df_matched = overview_df[
    overview_df["order_nr"].isin(matched_orders)
]


# ---------------------------------------------------------
# 3. Find the closest SVI measurement for every investigation
# ---------------------------------------------------------

def path_words(value):
    return tuple(re.findall(r"[^\W_]+", str(value).casefold()))


svi_dates = pd.to_datetime(svi["datum"], errors="coerce").dt.normalize()
svi_rows_by_word = {}
svi_location_words = {}

for svi_row_index, (installation, svi_date) in enumerate(
    zip(svi["installatie"], svi_dates)
):
    if pd.isna(installation) or pd.isna(svi_date):
        continue
    installation_words = path_words(installation)
    if not installation_words:
        continue
    svi_location_words[svi_row_index] = installation_words
    for installation_word in set(installation_words):
        svi_rows_by_word.setdefault(installation_word, []).append(svi_row_index)


match_table_all = []
match_table_svi_only = []
start_time = time.perf_counter()
total_investigations = len(overview_df_matched)

for investigation_count, (investigation_row_index, investigation_row) in enumerate(overview_df_matched.iterrows(), start=1,):
    investigation_date = pd.to_datetime(
        investigation_row["datum_monstername"],
        errors="coerce",
    )
    investigation_installatie = investigation_row["installatie_naam"]
    if pd.isna(investigation_date) or pd.isna(investigation_installatie):
        matched_record = None
    else:
        investigation_date = investigation_date.normalize()
        investigation_words = path_words(investigation_installatie)
        candidate_row_indices = set()
        for word in set(investigation_words):
            candidate_row_indices.update(svi_rows_by_word.get(word, ()))

        matched_record = None
        exact_match_indices = sorted(
            svi_row_index
            for svi_row_index in candidate_row_indices
            if svi_location_words[svi_row_index] == investigation_words
        )
        if exact_match_indices:
            candidate_row_indices = exact_match_indices
            candidate_dates = svi_dates.iloc[candidate_row_indices]
            day_diffs = (candidate_dates - investigation_date).dt.days
            eligible_diffs = day_diffs[day_diffs.between(-10, 10)]

            if not eligible_diffs.empty:
                svi_row_index = int(eligible_diffs.abs().idxmin())
                day_diff = int(eligible_diffs.loc[svi_row_index])
                closest_row = svi.iloc[svi_row_index]
                matched_record = {
                    **closest_row.to_dict(),
                    "day_diff_SVI": day_diff,
                    "investigation_row_index": investigation_row_index,
                    **investigation_row.to_dict(),
                }
                matched_record_SVIonly = {
                                    **closest_row.to_dict(),
                                    "day_diff_SVI": day_diff,
                                    "investigation_row_index": investigation_row_index,
                                    "order_nr": investigation_row["order_nr"]
                                }

    if matched_record is not None:
        match_table_all.append(matched_record)
        match_table_svi_only.append(matched_record_SVIonly)

    if investigation_count % 100 == 0 or investigation_count == total_investigations:
        elapsed_seconds = time.perf_counter() - start_time
        print(
            f"Processed {investigation_count}/{total_investigations} investigations; "
            f"matches: {len(match_table_all)}; elapsed: {elapsed_seconds:.1f}s"
        )

match_df_svi_only = pd.DataFrame(match_table_svi_only)
match_df_all = pd.DataFrame(match_table_all)
#match_df_all.to_excel("process_timeseries/svi_matches_table.xlsx", index=False)

# match_df["order_nr"].duplicated().sum() # no duplicates!


### match found SVIs to image match table

# read table with images that are a match to the overview table
microscopic_match_table_extended = pd.read_excel("data/microscopic_match_table_extended.xlsx")

# Merge all overview information into the match table
microscopic_match_table_extended_withsvi = microscopic_match_table_extended.merge(
    match_df_svi_only,
    on="order_nr",
    how="left"
)

microscopic_match_table_extended_withsvi["SVI"] = (
    microscopic_match_table_extended_withsvi["SVI"]
    .combine_first(microscopic_match_table_extended_withsvi["SVI_historisch"])
)

microscopic_match_table_extended_withsvi = (
    microscopic_match_table_extended_withsvi
    .drop(columns="SVI_historisch")
)


microscopic_match_table_extended_withsvi.to_excel("data/microscopic_match_table_extended_withsvi.xlsx", index=False) # save it here for Nusret

