import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# load measurement data
# ---------------------------------------------------------

csv_path = "data/Aquafin data (vertrouwelijk NDA)-ORIGINAL/MLSS_turbiditeit_SVI_ZS_alle_zuiveringen.csv"  # Replace with the CSV file path.

df_timeseries = pd.read_csv(csv_path)

df_timeseries["datum"] = pd.to_datetime(df_timeseries["datum"], errors="coerce")
df_timeseries["waarde_num"] = pd.to_numeric(df_timeseries["waarde"], errors="coerce")

# fix duplicates
df_clean = (
    df_timeseries.groupby(["installatie", "datum", "parameter"], as_index=False)
    .agg(waarde=("waarde_num", "mean"))
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
# plot some parameters for a single installation for visual inspection
# ---------------------------------------------------------

# Choose an installation to inspect; by default, use the first available one.
installation_name = "RWZI Vosselaar"  # Replace with the desired installation name.
df_installation = (
    df_wide.loc[df_wide["installatie"] == installation_name]
    .sort_values("datum")
    .set_index("datum")
)

parameter_groups = {
    "MLSS": [
        "MLSS",
        "MLSS_maandrapport",
        "MLSS_schep",
    ],

    "SVI": [
        "SVI",
        "SVI_historisch",
    ],

    "ZS effluent": [
        "ZS_effluent",
        "ZS_effluent_VMM",
    ],

    "Flowrate": [
        "debiet_hist",
    ],

    "Turbidity": [
        "turb_historisch",
        "turb_maandrapport",
        "turbiditeit",
    ],
}



for figure_title, parameters in parameter_groups.items():

    # Only keep parameters that actually exist in the dataframe
    available_parameters = [
        parameter
        for parameter in parameters
        if parameter in df_installation.columns
    ]

    if not available_parameters:
        print(f"No data found for {figure_title}")
        continue

    fig, ax = plt.subplots(figsize=(12, 5))

    for parameter in available_parameters:

        # Only plot actual measurements (ignore NaN)
        data = df_installation[parameter].dropna()

        if len(data) == 0:
            continue

        ax.scatter(
            data.index,
            data.values,
            marker="o",
            linestyle="-",
            #markersize=4,
            label=parameter,
        )

    ax.set_title(f"{figure_title}: {installation_name}")
    ax.set_xlabel("datum")
    ax.set_ylabel("waarde")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


historical_file_availability = pd.read_csv(
    "process_timeseries/historical_file_availability_df.csv"
)
investigation_availability = pd.read_csv(
    "process_timeseries/investigation_availability_df.csv"
)

historical_file_availability["datum"] = pd.to_datetime(
    historical_file_availability["datum"], errors="coerce"
)
investigation_availability["investigation_available"] = pd.to_datetime(
    investigation_availability["investigation_available"], errors="coerce"
)
investigation_availability["investigation_available_with_images"] = pd.to_datetime(
    investigation_availability["investigation_available_with_images"], errors="coerce"
)

def normalize_installation_name(name):
    return str(name).strip().casefold().removeprefix("rwzi ").strip()


installation_key = normalize_installation_name(installation_name)
historical_for_installation = historical_file_availability.loc[
    historical_file_availability["installatie"].map(normalize_installation_name)
    == installation_key
].copy()
investigations_for_installation = investigation_availability.loc[
    investigation_availability["installatie"].map(normalize_installation_name)
    == installation_key
].copy()

fig, (historical_ax, investigations_ax) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True
)

if not historical_for_installation.empty:
    historical_by_date = (
        historical_for_installation.groupby("datum")[["nr_images", "nr_fotobijlage_docs"]]
        .sum()
        .sort_index()
    )
    historical_ax.scatter(
        historical_by_date.index,
        historical_by_date["nr_images"],
        marker="o",
        label="Historical images",
    )
    historical_ax.scatter(
        historical_by_date.index,
        historical_by_date["nr_fotobijlage_docs"],
        marker="o",
        label="Fotobijlage documents",
    )
else:
    print(f"No historical file availability found for {installation_name}")

investigations_by_date = (
    investigations_for_installation["investigation_available"]
    .dropna()
    .dt.normalize()
    .value_counts()
    .sort_index()
)
investigations_with_images_by_date = (
    investigations_for_installation["investigation_available_with_images"]
    .dropna()
    .dt.normalize()
    .value_counts()
    .sort_index()
)

if not investigations_by_date.empty:
    investigations_ax.scatter(
        investigations_by_date.index,
        investigations_by_date.values,
        marker="o",
        label="Investigations",
    )
if not investigations_with_images_by_date.empty:
    investigations_ax.scatter(
        investigations_with_images_by_date.index,
        investigations_with_images_by_date.values,
        marker="o",
        label="Investigations with matched images",
    )
if investigations_by_date.empty:
    print(f"No investigation availability found for {installation_name}")

historical_ax.set_title(f"Historical image files: {installation_name}")
historical_ax.set_ylabel("File count")
investigations_ax.set_title("Microscopy investigations")
investigations_ax.set_ylabel("Investigation count")
investigations_ax.set_xlabel("Date")
for ax in (historical_ax, investigations_ax):
    ax.grid(True, alpha=0.3)
    if ax.collections:
        ax.legend()

fig.tight_layout()
plt.show()



# # ---------------------------------------------------------
# # Check timeseries quality and generate a report
# # ---------------------------------------------------------
# gap_multiplier = 2
# report_path = Path("process_timeseries/timeseries_quality_report.xlsx")
# report_path.parent.mkdir(parents=True, exist_ok=True)

# print("Calculating utility, observation, and source summaries...")
# utility_date_ranges = (
#     df_timeseries.groupby("installatie", as_index=False)
#     .agg(
#         first_date=("datum", "min"),
#         last_date=("datum", "max"),
#         observation_rows=("datum", "size"),
#     )
# )

# observations_by_parameter = (
#     df_timeseries.groupby(["installatie", "parameter"], observed=True)
#     .size()
#     .unstack(fill_value=0)
#     .reset_index()
#     .rename_axis(columns=None)
# )

# observations_by_source = (
#     df_timeseries.assign(source_report=df_timeseries["bron"].fillna("<missing>"))
#     .groupby(["installatie", "source_report"], observed=True)
#     .size()
#     .unstack(fill_value=0)
#     .reset_index()
#     .rename_axis(columns=None)
# )

# print("Calculating measurement frequencies and gaps...")
# dated_observations = (
#     df_clean.assign(date_day=df_clean["datum"].dt.normalize())
#     [["installatie", "parameter", "date_day"]]
#     .drop_duplicates()
#     .sort_values(["installatie", "parameter", "date_day"])
# )
# series_keys = ["installatie", "parameter"]
# dated_observations["previous_date"] = (
#     dated_observations.groupby(series_keys)["date_day"].shift()
# )
# dated_observations["interval_days"] = (
#     dated_observations["date_day"] - dated_observations["previous_date"]
# ).dt.days

# frequency_by_utility = (
#     dated_observations.groupby(series_keys, as_index=False)
#     .agg(
#         observation_dates=("date_day", "size"),
#         first_date=("date_day", "min"),
#         last_date=("date_day", "max"),
#         median_interval_days=("interval_days", "median"),
#         mean_interval_days=("interval_days", "mean"),
#     )
# )
# frequency_by_parameter = (
#     frequency_by_utility.groupby("parameter", as_index=False)
#     .agg(
#         utilities=("installatie", "nunique"),
#         total_observation_dates=("observation_dates", "sum"),
#         median_of_utility_median_interval_days=("median_interval_days", "median"),
#         mean_of_utility_median_interval_days=("median_interval_days", "mean"),
#     )
# )

# gaps = dated_observations.merge(
#     frequency_by_utility[series_keys + ["median_interval_days"]],
#     on=series_keys,
#     how="left",
# )
# gaps = gaps.loc[
#     gaps["interval_days"] > gap_multiplier * gaps["median_interval_days"],
#     series_keys + ["previous_date", "date_day", "interval_days", "median_interval_days"],
# ].rename(columns={
#     "previous_date": "gap_start",
#     "date_day": "gap_end",
#     "interval_days": "gap_days",
#     "median_interval_days": "typical_interval_days",
# })


# print("Flagging negative and Tukey-IQR extreme values...")
# value_quartiles = (
#     df_timeseries.groupby("parameter")["waarde_num"]
#     .quantile([0.25, 0.75])
#     .unstack()
#     .rename(columns={0.25: "q1", 0.75: "q3"})
# )
# value_quartiles["iqr"] = value_quartiles["q3"] - value_quartiles["q1"]
# value_quartiles["lower_fence"] = value_quartiles["q1"] - 1.5 * value_quartiles["iqr"]
# value_quartiles["upper_fence"] = value_quartiles["q3"] + 1.5 * value_quartiles["iqr"]

# lower_fence = df_timeseries["parameter"].map(value_quartiles["lower_fence"])
# upper_fence = df_timeseries["parameter"].map(value_quartiles["upper_fence"])
# negative_values = df_timeseries["waarde_num"] < 0
# iqr_outliers = (
#     (df_timeseries["waarde_num"] < lower_fence)
#     | (df_timeseries["waarde_num"] > upper_fence)
# )
# non_numeric_values = df_timeseries["waarde"].notna() & df_timeseries["waarde_num"].isna()

# value_quality = (
#     df_timeseries.assign(
#         negative_value=negative_values,
#         iqr_outlier=iqr_outliers,
#         non_numeric_value=non_numeric_values,
#     )
#     .groupby(["installatie", "parameter"], observed=True)
#     .agg(
#         observations=("parameter", "size"),
#         negative_values=("negative_value", "sum"),
#         iqr_outliers=("iqr_outlier", "sum"),
#         non_numeric_values=("non_numeric_value", "sum"),
#     )
#     .reset_index()
#     .merge(value_quartiles.reset_index(), on="parameter", how="left")
# )

# extreme_mask = negative_values | iqr_outliers
# extreme_examples = df_timeseries.loc[
#     extreme_mask,
#     ["installatie", "datum", "parameter", "waarde", "bron", "waarde_num"],
# ].copy()
# extreme_examples["negative_value"] = negative_values.loc[extreme_mask].to_numpy()
# extreme_examples["iqr_outlier"] = iqr_outliers.loc[extreme_mask].to_numpy()
# extreme_examples = pd.concat([
#     extreme_examples.sort_values("waarde_num").groupby("parameter").head(10),
#     extreme_examples.sort_values("waarde_num").groupby("parameter").tail(10),
# ]).drop_duplicates()

# print("Calculating pairwise parameter date overlap...")
# parameter_names = sorted(df_clean["parameter"].dropna().unique())
# date_presence = (
#     df_clean.assign(date_day=df_clean["datum"].dt.normalize())
#     [["installatie", "date_day", "parameter"]]
#     .drop_duplicates()
#     .assign(present=1)
#     .pivot_table(
#         index=["installatie", "date_day"],
#         columns="parameter",
#         values="present",
#         fill_value=0,
#     )
#     .reindex(columns=parameter_names, fill_value=0)
# )

# overlap_rows = []
# for installation, utility_presence in date_presence.groupby(level="installatie", sort=False):
#     presence_matrix = utility_presence.to_numpy(dtype=np.int32)
#     date_counts = presence_matrix.sum(axis=0)
#     shared_date_counts = presence_matrix.T @ presence_matrix
#     for first_index in range(len(parameter_names)):
#         for second_index in range(first_index + 1, len(parameter_names)):
#             first_count = int(date_counts[first_index])
#             second_count = int(date_counts[second_index])
#             if not first_count or not second_count:
#                 continue
#             overlap_count = int(shared_date_counts[first_index, second_index])
#             overlap_rows.append({
#                 "installatie": installation,
#                 "parameter_1": parameter_names[first_index],
#                 "parameter_2": parameter_names[second_index],
#                 "overlapping_dates": overlap_count,
#                 "dates_parameter_1": first_count,
#                 "dates_parameter_2": second_count,
#                 "overlap_pct_of_smaller_series": (
#                     100 * overlap_count / min(first_count, second_count)
#                 ),
#             })
# parameter_overlap = pd.DataFrame(overlap_rows)

# report_summary = pd.DataFrame([{
#     "utilities": df_timeseries["installatie"].nunique(),
#     "parameters": df_timeseries["parameter"].nunique(),
#     "sources": df_timeseries["bron"].nunique(),
#     "raw_observation_rows": len(df_timeseries),
#     "invalid_dates": int(df_timeseries["datum"].isna().sum()),
#     "non_numeric_values": int(non_numeric_values.sum()),
#     "first_date": df_timeseries["datum"].min(),
#     "last_date": df_timeseries["datum"].max(),
#     "gap_rule": f"> {gap_multiplier}x median interval per utility-parameter series",
#     "extreme_rule": "negative values and values outside parameter-level 1.5x IQR fences",
# }])

# with pd.ExcelWriter(report_path) as writer:
#     report_summary.to_excel(writer, sheet_name="summary", index=False)
#     utility_date_ranges.to_excel(writer, sheet_name="utility_date_ranges", index=False)
#     observations_by_parameter.to_excel(writer, sheet_name="obs_by_parameter", index=False)
#     observations_by_source.to_excel(writer, sheet_name="obs_by_source", index=False)
#     frequency_by_utility.to_excel(writer, sheet_name="frequency_by_utility", index=False)
#     frequency_by_parameter.to_excel(writer, sheet_name="frequency_by_parameter", index=False)
#     gaps.to_excel(writer, sheet_name="gaps", index=False)
#     value_quality.to_excel(writer, sheet_name="value_quality", index=False)
#     extreme_examples.to_excel(writer, sheet_name="extreme_examples", index=False)
#     parameter_overlap.to_excel(writer, sheet_name="parameter_overlap", index=False)

# print(f"Utilities: {df_timeseries['installatie'].nunique()}")
# print(f"Timeseries quality report saved to {report_path}")

# # ---------------------------------------------------------
# # check how same parameters from different sources compare to each other
# # ---------------------------------------------------------

# ## When two sources measure turbidity around the same time, how similar are they?
# # You could calculate:

# # correlation
# # median absolute difference
# # relative difference
# # systematic bias
# # overlap duration

# # create harmonized turbidity dataframe


# # ---------------------------------------------------------
# # Clean individual parameters first
# # ---------------------------------------------------------



# # MLSS

# # negative values
# # zero values
# # implausibly high values
# # sudden enormous jumps
# # repeated identical measurements
# # unit inconsistencies


# # SVI

# # negative values
# # zero
# # extremely high values
# # potentially incorrect units
# # suspicious repeated values


# # Flowrate

# # This one needs special treatment because it's fundamentally different from the others.

# # A flowrate of 10,000 m³/d and an MLSS measurement of 3,000 mg/L aren't comparable simply because they have the same timestamp.

# # You may ultimately want things such as:

# # daily mean flow
# # daily total flow
# # flow percentile
# # flow relative to plant capacity

# # rather than simply correlating raw flow measurements.

# # Effluent TSS / turbidity

# # Also consider whether these are:

# # instantaneous measurements
# # laboratory samples
# # online measurements
# # daily averages
# # composite samples

# # ---------------------------------------------------------
# # temporal alignment
# # ---------------------------------------------------------

# # How does the relationship between MLSS and effluent TSS change depending on the temporal scale?

# # ---------------------------------------------------------
# # calculate correlations
# # ---------------------------------------------------------

# #between plants
# #within plants - different temperol scales: same day, same week, same month, same season, same year
# #also cross correlations, e.g. MLSS today vs effluent TSS tomorrow, etc.