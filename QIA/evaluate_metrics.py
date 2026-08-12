import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


match_table_w_metrics = pd.read_excel("match_table_extended_ph100_masks_with_metrics_v2.xlsx")
match_table_w_metrics_dispersed = pd.read_excel("match_table_extended_ph100_masks_with_metrics_w_dispersed_v2.xlsx")


################## check correlation between aquafin and image analysis metrics for floc size categories
## with equivalent diameter
agg_dict = {
    # sums
    "n_flocs": "sum",
    "n_flocs_diameter_<150um": "mean",
    "n_flocs_diameter_150_500um": "mean",
    "n_flocs_diameter_>500um": "mean",

    # Utility measurements
    "Klein (KLEI_VGR_3) [%]": "first",
    "Middelgroot (MIDG_VGR_3) [%]": "first",
    "Groot (GROO_VGR_3) [%]": "first",
}

sample_df = (
    match_table_w_metrics
    .groupby("order_nr", as_index=False)
    .agg(agg_dict)
)

comparisons = [
    ("Klein (KLEI_VGR_3) [%]", "n_flocs_diameter_<150um"),
    ("Middelgroot (MIDG_VGR_3) [%]", "n_flocs_diameter_150_500um"),
    ("Groot (GROO_VGR_3) [%]", "n_flocs_diameter_>500um"),
]


for aquafin_col, my_col in comparisons:

    df = sample_df[[aquafin_col, my_col]].dropna()

    pearson_r, pearson_p = pearsonr(df[aquafin_col], df[my_col])
    spearman_rho, spearman_p = spearmanr(df[aquafin_col], df[my_col])

    print(f"\n{aquafin_col} vs {my_col}")
    print(f"Pearson  r   = {pearson_r:.3f} (p={pearson_p:.4f})")
    print(f"Spearman rho = {spearman_rho:.3f} (p={spearman_p:.4f})")

    plt.figure(figsize=(5,5))
    plt.scatter(df[aquafin_col], df[my_col], alpha=0.2, s=15)

    plt.xlabel(f"Aquafin")
    plt.ylabel(f"Image analysis")
    plt.title(f"{my_col}")
    plt.text(
        0.05, 0.95,
        f"Pearson r = {pearson_r:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.5)
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

## with feret diameter
agg_dict = {
    # sums
    "n_flocs": "sum",
    "feret_n_flocs_diameter_<150um": "mean",
    "feret_n_flocs_diameter_150_500um": "mean",
    "feret_n_flocs_diameter_>500um": "mean",

    # Utility measurements
    "Klein (KLEI_VGR_3) [%]": "first",
    "Middelgroot (MIDG_VGR_3) [%]": "first",
    "Groot (GROO_VGR_3) [%]": "first",
}

sample_df = (
    match_table_w_metrics
    .groupby("order_nr", as_index=False)
    .agg(agg_dict)
)

comparisons = [
    ("Klein (KLEI_VGR_3) [%]", "feret_n_flocs_diameter_<150um"),
    ("Middelgroot (MIDG_VGR_3) [%]", "feret_n_flocs_diameter_150_500um"),
    ("Groot (GROO_VGR_3) [%]", "feret_n_flocs_diameter_>500um"),
]


for aquafin_col, my_col in comparisons:

    df = sample_df[[aquafin_col, my_col]].dropna()

    pearson_r, pearson_p = pearsonr(df[aquafin_col], df[my_col])
    spearman_rho, spearman_p = spearmanr(df[aquafin_col], df[my_col])

    print(f"\n{aquafin_col} vs {my_col}")
    print(f"Pearson  r   = {pearson_r:.3f} (p={pearson_p:.4f})")
    print(f"Spearman rho = {spearman_rho:.3f} (p={spearman_p:.4f})")

    plt.figure(figsize=(5,5))
    plt.scatter(df[aquafin_col], df[my_col], alpha=0.2, s=15)

    plt.xlabel(f"Aquafin")
    plt.ylabel(f"Image analysis")
    plt.title(f"{my_col}")
    plt.text(
        0.05, 0.95,
        f"Pearson r = {pearson_r:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.5)
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


### with dispersed fraction accounted for

## with equivalent diameter
agg_dict = {
    # sums
    "n_small_flocs": "mean",
    "n_medium_flocs": "mean",
    "n_large_flocs": "mean", 
    "n_dispersed_flocs": "mean",

    # Utility measurements
    "Klein (KLEI_VGR_3) [%]": "first",
    "Middelgroot (MIDG_VGR_3) [%]": "first",
    "Groot (GROO_VGR_3) [%]": "first",
    "Gedispergeerd (GEDI_VGR_3) [%]": "first"
}

sample_df = (
    match_table_w_metrics_dispersed
    .groupby("order_nr", as_index=False)
    .agg(agg_dict)
)

comparisons = [
    ("Klein (KLEI_VGR_3) [%]", "n_small_flocs"),
    ("Middelgroot (MIDG_VGR_3) [%]", "n_medium_flocs"),
    ("Groot (GROO_VGR_3) [%]", "n_large_flocs"),
    ("Gedispergeerd (GEDI_VGR_3) [%]", "n_dispersed_flocs")
]


for aquafin_col, my_col in comparisons:

    df = sample_df[[aquafin_col, my_col]].dropna()

    pearson_r, pearson_p = pearsonr(df[aquafin_col], df[my_col])
    spearman_rho, spearman_p = spearmanr(df[aquafin_col], df[my_col])

    print(f"\n{aquafin_col} vs {my_col}")
    print(f"Pearson  r   = {pearson_r:.3f} (p={pearson_p:.4f})")
    print(f"Spearman rho = {spearman_rho:.3f} (p={spearman_p:.4f})")

    plt.figure(figsize=(5,5))
    plt.scatter(df[aquafin_col], df[my_col], alpha=0.2, s=15)

    plt.xlabel(f"Aquafin")
    plt.ylabel(f"Image analysis")
    plt.title(f"{my_col}")
    plt.text(
        0.05, 0.95,
        f"Pearson r = {pearson_r:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.5)
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


################## check correlation between aquafin and image analysis metrics for floc shape categories

###### structuur, vorm, stevigheid

agg_dict = {
    "eccentricity": "mean",
    "aspect_ratio": "mean",
    "compactness": "mean",
    "mean_floc_crofton_perimeter_um": "mean",

    # Utility measurements
    "Structuur (STRU_VMF_2)": "first", # Diffuus / Compact
    "Vorm (VORM_VMF_2)": "first",  # Agglomeraten / Onregelmatig / Afgerond / 
    "Stevigheid (STEV_VMF_2)": "first", #Sterk / Zwak
}


sample_df = (
    match_table_w_metrics
    .groupby("order_nr", as_index=False)
    .agg(agg_dict)
)

### Structuur
# Keep only complete rows
df = sample_df[["Structuur (STRU_VMF_2)", "compactness"]].dropna()

# Keep the desired order
order = ["Diffuus", "Compact"]

data = [
    df.loc[df["Structuur (STRU_VMF_2)"] == cat, "compactness"]
    for cat in order
]

plt.figure(figsize=(5,5))

plt.boxplot(data, tick_labels=order)

for i, cat in enumerate(order, start=1):
    y = df.loc[df["Structuur (STRU_VMF_2)"] == cat, "compactness"]
    x = np.random.normal(i, 0.05, len(y))
    plt.scatter(x, y, alpha=0.7)

plt.xlabel("Aquafin structure")
plt.ylabel(" compactness")
plt.tight_layout()
plt.show()

### stevigheid
# Keep only complete rows
df = sample_df[["Stevigheid (STEV_VMF_2)", "compactness"]].dropna()

# Keep the desired order
order = ["Sterk", "Zwak"]

data = [
    df.loc[df["Stevigheid (STEV_VMF_2)"] == cat, "compactness"]
    for cat in order
]

plt.figure(figsize=(5,5))

plt.boxplot(data, tick_labels=order)

for i, cat in enumerate(order, start=1):
    y = df.loc[df["Stevigheid (STEV_VMF_2)"] == cat, "compactness"]
    x = np.random.normal(i, 0.05, len(y))
    plt.scatter(x, y, alpha=0.7)

plt.xlabel("Aquafin sturdiness")
plt.ylabel(" compactness")
plt.tight_layout()
plt.show()


### vorm
# Keep only complete rows
df = sample_df[["Vorm (VORM_VMF_2)", "compactness"]].dropna()

# Keep the desired order
order = ["Agglomeraten", "Onregelmatig", "Afgerond"]

data = [
    df.loc[df["Vorm (VORM_VMF_2)"] == cat, "compactness"]
    for cat in order
]

plt.figure(figsize=(5,5))

plt.boxplot(data, tick_labels=order)

for i, cat in enumerate(order, start=1):
    y = df.loc[df["Vorm (VORM_VMF_2)"] == cat, "compactness"]
    x = np.random.normal(i, 0.05, len(y))
    plt.scatter(x, y, alpha=0.7)

plt.xlabel("Aquafin shape")
plt.ylabel(" compactness")
plt.tight_layout()
plt.show()





################## check correlation between aquafin and image analysis metrics for filament abundance


agg_dict = {
    "total_filament_area": "max",
    "mean_filament_length": "max",
    "median_filament_length": "max",
    "filament_to_floc_ratio": "max",

    # Utility measurements
    "Semikwantitatieve beoordeling (SSCO_FIG_2)": "first", # veel matig zeer veel weinig
    "% Microthrix Parvicella (PROC_MP_QPCR) [%]": "first"
}


sample_df = (
    match_table_w_metrics
    .groupby("order_nr", as_index=False)
    .agg(agg_dict)
)

### Filament abundance
# Keep only complete rows
df = sample_df[["Semikwantitatieve beoordeling (SSCO_FIG_2)", "filament_to_floc_ratio"]].dropna()

# Keep the desired order
order = ["Weinig", "Matig", "Veel", "Zeer veel"]

data = [
    df.loc[df["Semikwantitatieve beoordeling (SSCO_FIG_2)"] == cat, "filament_to_floc_ratio"]
    for cat in order
]

plt.figure(figsize=(5,5))

plt.boxplot(data, tick_labels=order)

for i, cat in enumerate(order, start=1):
    y = df.loc[df["Semikwantitatieve beoordeling (SSCO_FIG_2)"] == cat, "filament_to_floc_ratio"]
    x = np.random.normal(i, 0.05, len(y))
    plt.scatter(x, y, alpha=0.7)

plt.xlabel("Aquafin structure")
plt.ylabel("filament_to_floc_ratio")
plt.tight_layout()
plt.show()



df = sample_df[["% Microthrix Parvicella (PROC_MP_QPCR) [%]", "filament_to_floc_ratio"]].dropna()

pearson_r, pearson_p = pearsonr(df["% Microthrix Parvicella (PROC_MP_QPCR) [%]"], df["filament_to_floc_ratio"])
spearman_rho, spearman_p = spearmanr(df["% Microthrix Parvicella (PROC_MP_QPCR) [%]"], df["filament_to_floc_ratio"])

plt.figure(figsize=(5,5))
plt.scatter(df["% Microthrix Parvicella (PROC_MP_QPCR) [%]"], df["filament_to_floc_ratio"], alpha=0.2, s=15)

plt.xlabel(f"Aquafin")
plt.ylabel(f"Image analysis")
plt.title(f"filament_to_floc_ratio")
plt.text(
    0.05, 0.95,
    f"Pearson r = {pearson_r:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.5)
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


################## check correlation between aquafin and image analysis metrics for floc shape categories

###### structuur, vorm, stevigheid

agg_dict = {
    "fraction_area_dispersed": "max",

    # Utility measurements
    "Structuur (STRU_VMF_2)": "first", # Diffuus / Compact
    "Vorm (VORM_VMF_2)": "first",  # Agglomeraten / Onregelmatig / Afgerond / 
    "Stevigheid (STEV_VMF_2)": "first", #Sterk / Zwak
}


sample_df = (
    match_table_w_metrics_dispersed
    .groupby("order_nr", as_index=False)
    .agg(agg_dict)
)

### Structuur
# Keep only complete rows
df = match_table_w_metrics_dispersed[["Structuur (STRU_VMF_2)", "fraction_area_dispersed"]].dropna()

# Keep the desired order
order = ["Diffuus", "Compact"]

data = [
    df.loc[df["Structuur (STRU_VMF_2)"] == cat, "fraction_area_dispersed"]
    for cat in order
]

plt.figure(figsize=(5,5))

plt.boxplot(data, tick_labels=order)

for i, cat in enumerate(order, start=1):
    y = df.loc[df["Structuur (STRU_VMF_2)"] == cat, "fraction_area_dispersed"]
    x = np.random.normal(i, 0.05, len(y))
    plt.scatter(x, y, alpha=0.7)

plt.xlabel("Aquafin structure")
plt.ylabel("Fraction of dispersed area")
plt.tight_layout()
plt.show()



######## investigate annomalies
# Show the largest filament-to-floc ratios
outliers = (
    match_table_w_metrics
    .sort_values("filament_to_floc_ratio", ascending=False)
    [[
        "filament_to_floc_ratio",
        "image_path"   
    ]]
    .head(10)
)

print(outliers.to_string(index=False))

# Get the 2 largest outliers
top_2 = outliers.head(2)

for _, row in top_2.iterrows():
    image = Image.open(row["image_path"])

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(
        f"Filament-to-floc ratio: {row['filament_to_floc_ratio']:.2f}"
    )
    plt.axis("off")
    plt.show()



# Show the largest total_filament_area
outliers = (
    match_table_w_metrics
    .sort_values("total_filament_area", ascending=False)
    [[
        "total_filament_area",
        "image_path"   
    ]]
    .head(10)
)

print(outliers.to_string(index=False))


# Get the 2 largest outliers
top_2 = outliers.head(2)

for _, row in top_2.iterrows():
    image = Image.open(row["image_path"])

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(
        f"total_filament_area: {row['total_filament_area']:.2f}"
    )
    plt.axis("off")
    plt.show()