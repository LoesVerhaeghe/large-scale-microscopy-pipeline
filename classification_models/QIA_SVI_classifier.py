"""
Random Forest regressor for predicting continuous SVI values from QIA
morphology metrics.

  - per-experiment: features are mean-aggregated per experiment_id

Train/test assignment comes from the fixed split workbook, not a random split,
so the model is evaluated on the same held-out experiments as the rest of the
pipeline.

Reads output/<run_name>/metrics.xlsx (from 03_calculate_metrics.py), joined
against the split workbook on cfg.split_merge_col ("image_path").
Saves regression summary metrics to
output/<run_name>/classification_models/QIA_SVI_regressor/.
"""

from config.config import cfg

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt
import numpy as np

features = [
    "n_flocs",
    "total_floc_area_um2",
    "mean_floc_area_um2",
    "median_floc_area_um2",
    "mean_floc_eq_diameter_um",
    #"median_floc_eq_diameter_um",
    #"mean_floc_crofton_perimeter_um",
    #"median_floc_crofton_perimeter_um",
    #"mean_major_axis_um",
    #"mean_minor_axis_um",
    "fraction_microflocs",
    "eccentricity",
    "aspect_ratio",
    "form_factor",
    "roundness",
    "compactness",
    #"dispersed_area_px",
    "fraction_area_dispersed",
    "n_filaments",
    "total_filament_length_um",
    #"total_filament_area_um2",
    "mean_filament_length_um",
    #"median_filament_length_um",
    "max_filament_length_um",
    "filament_area_fraction",
    "filament_length_density",
    "filament_to_floc_ratio",
    "n_junctions",
    "branching_density"
]

target = "SVI"

def normalize_image_path(path):
    path = str(path).replace("\\", "/")
    marker = "Aquafin_data_cleaned/"
    
    if marker in path:
        return path[path.index(marker):]
    
    return path

def load_fixed_split(cfg):
    """
    Loads the authoritative train/test split and joins it against the metrics
    table. The train/test assignment comes from the split workbook, while
    the target label and structure features come from the metrics table.
    """
    metrics_df = pd.read_excel(cfg.metrics_path)

    train_split = pd.read_excel("/data/nvme3/loesv/analysis/train_test_splits_newdata_test/SVI_train_test_split.xlsx", sheet_name='train')
    test_split = pd.read_excel("/data/nvme3/loesv/analysis/train_test_splits_newdata_test/SVI_train_test_split.xlsx", sheet_name='test')

    metrics_df = metrics_df.drop_duplicates().reset_index(drop=True)
    train_split = train_split.drop_duplicates().reset_index(drop=True)
    test_split = test_split.drop_duplicates().reset_index(drop=True)

    # Normalize paths before merging
    metrics_df["merge_image_path"] = metrics_df[cfg.split_merge_col].apply(normalize_image_path)
    train_split["merge_image_path"] = train_split[cfg.split_merge_col].apply(normalize_image_path)
    test_split["merge_image_path"] = test_split[cfg.split_merge_col].apply(normalize_image_path)

    join_cols = ["merge_image_path", *features ]

    assert set(train_split[cfg.split_group_col]).isdisjoint(test_split[cfg.split_group_col]), \
            "train/test experiment_id sets overlap -- split workbook is not group-disjoint"

    train_df = train_split.merge(metrics_df[join_cols], on="merge_image_path", how="inner")
    test_df = test_split.merge(metrics_df[join_cols], on="merge_image_path", how="inner")

    print(f"Train: {len(train_split)} rows in split workbook -> {len(train_df)} matched to metrics "
          f"({train_df[cfg.split_group_col].nunique()} experiments)")
    print(f"Test:  {len(test_split)} rows in split workbook -> {len(test_df)} matched to metrics "
          f"({test_df[cfg.split_group_col].nunique()} experiments)")

    return train_df, test_df


def train_rf(X_train, y_train, cfg) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=1000,
        criterion="squared_error",
        random_state=cfg.seed,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    return model

def evaluate_and_report(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = {}
 
    metrics["Total_MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["Total_RMSE"] = root_mean_squared_error(y_true, y_pred)
    metrics["Total_R2"] = r2_score(y_true, y_pred)
 
    print(f"Total MAE: {metrics['Total_MAE']:.2f} mL/g")
    print(f"Total RMSE: {metrics['Total_RMSE']:.2f} mL/g")
    print(f"Total R²: {metrics['Total_R2']:.3f}")
    return metrics


def report_feature_importance(model, features):
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance)
    return importance


# --------------------------------------------------------------------------- #
# Variant 2: per-experiment (mean-aggregated features)
# --------------------------------------------------------------------------- #

def run_per_experiment(train_df, test_df, cfg):
    agg_dict = {feat: "mean" for feat in features}
    agg_dict[target] = "first"

    train_agg = train_df.groupby(cfg.split_group_col, as_index=False).agg(agg_dict)
    test_agg = test_df.groupby(cfg.split_group_col, as_index=False).agg(agg_dict)

    X_train, y_train = train_agg[features], train_agg[target]
    X_test, y_test = test_agg[features], test_agg[target]
    print(f"Training experiments: {len(X_train)} | Test experiments: {len(X_test)}")

    model = train_rf(X_train, y_train, cfg)
    y_pred = model.predict(X_test)

    metrics = evaluate_and_report(y_test, y_pred=y_pred)
    report_feature_importance(model, features)
    return metrics



def main():
    output_dir = cfg.output_dir / "classification_models" / "QIA_SVI_regressor"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_fixed_split(cfg)

    results = {}
    results["per_experiment"] = run_per_experiment(train_df, test_df, cfg)

    summary_df = pd.DataFrame(results).T
    summary_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path)
    print(f"\nSaved summary metrics to {summary_path}")

if __name__ == "__main__":
    main()