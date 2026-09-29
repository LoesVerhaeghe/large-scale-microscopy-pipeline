"""
Random Forest regressors predicting technician severity labels from
filament-related QIA metrics:

  2. per-experiment    -- features mean-aggregated per experiment_id

Train/test assignment comes from the authoritative fixed split workbook, 
NOT a random split -- this keeps every model
(RF / ViT / SegFormer-encoder classifier / etc.) comparable on the exact
same held-out experiments.

Reads output/<run_name>/metrics.xlsx (from 03_calculate_metrics.py), joined
against the split workbook on cfg.split_merge_col ("image_path").
Saves confusion matrices + a summary metrics table to
output/<run_name>/classification_models/QIA_size_regressor/.
"""

from config.config import cfg

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
import matplotlib.pyplot as plt

general_features = [
    "n_flocs",
    "total_floc_area_um2",
    "mean_floc_area_um2",
    "median_floc_area_um2",
    "mean_floc_eq_diameter_um",
    "median_floc_eq_diameter_um",
    "mean_floc_crofton_perimeter_um",
    "median_floc_crofton_perimeter_um",
    "mean_major_axis_um",
    "mean_minor_axis_um",
    "fraction_microflocs",
    "eccentricity",
    "aspect_ratio",
    "form_factor",
    "roundness",
    "compactness",
    "dispersed_area_px",
    "fraction_area_dispersed"
]

size_features = [
    "n_flocs",
    "n_small_flocs",
    "n_medium_flocs",
    "n_large_flocs",
    "n_dispersed_flocs",
    # "n_flocs_eq_diameter_<150um",
    # "n_flocs_eq_diameter_150_500um",
    # "n_flocs_eq_diameter_>500um",
    # "n_flocs_feret_diameter_<150um",
    # "n_flocs_feret_diameter_150_500um",
    # "n_flocs_feret_diameter_>500um"
]

size_targets = [
    "Klein (KLEI_VGR_3) [%]",
    "Middelgroot (MIDG_VGR_3) [%]",
    "Groot (GROO_VGR_3) [%]",
    "Gedispergeerd (GEDI_VGR_3) [%]"
]
count_features = [f for f in size_features if f != "n_flocs"]
 

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

    train_split = pd.read_excel("/data/nvme3/loesv/analysis/vlokgrotte/vlokgroote_GroupSplit_Seed0_Phase100um.xlsx", sheet_name='train')
    test_split = pd.read_excel("/data/nvme3/loesv/analysis/vlokgrotte/vlokgroote_GroupSplit_Seed0_Phase100um.xlsx", sheet_name='validation')

    # Normalize paths before merging
    metrics_df["merge_image_path"] = metrics_df[cfg.split_merge_col].apply(normalize_image_path)
    train_split["merge_image_path"] = train_split[cfg.split_merge_col].apply(normalize_image_path)
    test_split["merge_image_path"] = test_split[cfg.split_merge_col].apply(normalize_image_path)
    
    feature_cols = list(dict.fromkeys([*size_features, *general_features]))

    targets_in_split = all(t in train_split.columns for t in size_targets)
    target_cols = [] if targets_in_split else size_targets
 
    missing = [c for c in feature_cols + target_cols if c not in metrics_df.columns]
    if missing:
        raise KeyError(f"Columns missing from metrics table: {missing}")
 
    join_cols = ["merge_image_path", *feature_cols, *target_cols]


    assert set(train_split[cfg.split_group_col]).isdisjoint(test_split[cfg.split_group_col]), \
            "train/test experiment_id sets overlap -- split workbook is not group-disjoint"

    train_df = train_split.merge(metrics_df[join_cols], on="merge_image_path", how="inner")
    test_df = test_split.merge(metrics_df[join_cols], on="merge_image_path", how="inner")


    print(f"Train: {len(train_split)} rows in split workbook -> {len(train_df)} matched to metrics "
          f"({train_df[cfg.split_group_col].nunique()} experiments)")
    print(f"Test:  {len(test_split)} rows in split workbook -> {len(test_df)} matched to metrics "
          f"({test_df[cfg.split_group_col].nunique()} experiments)")

    # Targets are percentages -> float, never int. Rows without a technician
    # label cannot be trained or scored on.
    for df, name in ((train_df, "train"), (test_df, "test")):
        df[size_targets] = df[size_targets].apply(pd.to_numeric, errors="coerce")
        n_before = len(df)
        df.dropna(subset=size_targets, inplace=True)
        if len(df) < n_before:
            print(f"Dropped {n_before - len(df)} {name} rows with missing size labels")
 

    return train_df, test_df


def build_features(df, cfg):
    """
    Aggregate per image -> per experiment, then convert the raw counts into
    within-experiment fractions.
 
    Summed counts scale with how many images an experiment contains, while the
    target is a scale-free percentage. Fractions remove that nuisance variable;
    log1p(n_flocs) is kept so the model still knows how much evidence backs
    each experiment.
    """
    agg_dict = {feat: "sum" for feat in size_features}
    agg_dict.update({feat: "mean" for feat in general_features if feat not in size_features})
    # One "first" per target column -- size_targets is a list, not a column.
    agg_dict.update({target: "first" for target in size_targets})
 
    agg = df.groupby(cfg.split_group_col, as_index=False).agg(agg_dict)
 
    denom = agg["n_flocs"].replace(0, np.nan)
    for feat in count_features:
        agg[f"frac_{feat}"] = (agg[feat] / denom).fillna(0.0)
 
    agg["log_n_flocs"] = np.log1p(agg["n_flocs"])
 
    feature_cols = (
        ["log_n_flocs"]
        + [f"frac_{feat}" for feat in count_features]
        + [f for f in general_features if f != "n_flocs"]
    )
 
    X = agg[feature_cols].astype(float)
    y = agg[size_targets].astype(float)
    return X, y, agg[cfg.split_group_col], feature_cols


def train_rf(X_train, y_train, cfg) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=cfg.seed,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    return model

def normalize_to_percentages(y_pred):
    """
    Clip negatives and rescale each row to sum to 100, since the four classes
    are a composition. Rows that predict all-zero fall back to a uniform split.
    """
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    row_sums = y_pred.sum(axis=1, keepdims=True)
    uniform = np.full_like(y_pred, 100.0 / y_pred.shape[1])
    return np.where(row_sums > 0, y_pred / np.where(row_sums == 0, 1, row_sums) * 100.0, uniform)
 

def evaluate_and_report(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = {}

    for i, target_name in enumerate(size_targets):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = root_mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
 
        print(f"{target_name}: MAE={mae:.2f} pp | RMSE={rmse:.2f} pp | R²={r2:.3f}")
 
        metrics[f"{target_name}_MAE"] = mae
        metrics[f"{target_name}_RMSE"] = rmse
        metrics[f"{target_name}_R2"] = r2
 
    metrics["Total_MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["Total_RMSE"] = root_mean_squared_error(y_true, y_pred)
 
    print(f"Total MAE: {metrics['Total_MAE']:.2f} percentage points")
    print(f"Total RMSE: {metrics['Total_RMSE']:.2f} percentage points")
    return metrics


def report_feature_importance(model, features, output_dir):
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance)
 
    importance.to_csv(output_dir / "feature_importance.csv", header=["importance"])
 
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(importance))))
    importance.iloc[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("QIA feature importance -- floc size distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
 
    return importance


# --------------------------------------------------------------------------- #
# Variant 2: per-experiment (mean-aggregated features)
# --------------------------------------------------------------------------- #

def run_per_experiment(train_df, test_df, cfg, output_dir):
    X_train, y_train, _, feature_cols = build_features(train_df, cfg)
    X_test, y_test, test_ids, _ = build_features(test_df, cfg)
    X_test = X_test[feature_cols]
    
    print(f"Training experiments: {len(X_train)} | Test experiments: {len(X_test)}")

    model = train_rf(X_train, y_train, cfg)
    y_pred = normalize_to_percentages(model.predict(X_test))

    metrics = evaluate_and_report(y_test, y_pred)
    report_feature_importance(model, feature_cols, output_dir)
    return metrics



def main():
    output_dir = cfg.output_dir / "classification_models" / "QIA_flocsize_regressor"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_fixed_split(cfg)

    # results = {}
    # results["per_experiment"] = run_per_experiment(train_df, test_df, cfg, output_dir)

    # summary_df = pd.DataFrame(results).T
    # summary_path = output_dir / "summary_metrics.csv"
    # summary_df.to_csv(summary_path)
    # print(f"\nSaved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()