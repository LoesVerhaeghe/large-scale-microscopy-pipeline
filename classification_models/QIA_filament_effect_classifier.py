"""
Random Forest classifiers predicting technician severity labels from
filament-related QIA metrics:

  2. per-experiment    -- features mean-aggregated per experiment_id

Train/test assignment comes from the authoritative fixed split workbook
(cfg.split_workbook_path), NOT a random split -- this keeps every model
(RF / ViT / SegFormer-encoder classifier / etc.) comparable on the exact
same held-out experiments.

Reads output/<run_name>/metrics.xlsx (from 03_calculate_metrics.py), joined
against the split workbook on cfg.split_merge_col ("image_path").
Saves confusion matrices + a summary metrics table to
output/<run_name>/classification_models/QIA_filament_effect_classifier/.
"""

from config.config import cfg

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

filament_features = [
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
target = "Effect op vlokstructuur (EFVS_FIG_2)"

classifier_class_names = ['Brugvorming', 'Weinig of geen', 'Open vlokstructuur',
       'Insnoering', 'Overheersing']


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
    the target label and filament features come from the metrics table.
    """
    metrics_df = pd.read_excel(cfg.metrics_path)

    train_split = pd.read_excel(cfg.split_workbook_path, sheet_name=cfg.split_train_sheet)
    test_split = pd.read_excel(cfg.split_workbook_path, sheet_name=cfg.split_test_sheet)

    # Normalize paths before merging
    metrics_df["merge_image_path"] = metrics_df[cfg.split_merge_col].apply(normalize_image_path)
    train_split["merge_image_path"] = train_split[cfg.split_merge_col].apply(normalize_image_path)
    test_split["merge_image_path"] = test_split[cfg.split_merge_col].apply(normalize_image_path)

    join_cols = ["merge_image_path", *filament_features, target ]

    assert set(train_split[cfg.split_group_col]).isdisjoint(test_split[cfg.split_group_col]), \
            "train/test experiment_id sets overlap -- split workbook is not group-disjoint"

    train_df = train_split.merge(metrics_df[join_cols], on="merge_image_path", how="inner")
    test_df = test_split.merge(metrics_df[join_cols], on="merge_image_path", how="inner")


    print(f"Train: {len(train_split)} rows in split workbook -> {len(train_df)} matched to metrics "
          f"({train_df[cfg.split_group_col].nunique()} experiments)")
    print(f"Test:  {len(test_split)} rows in split workbook -> {len(test_df)} matched to metrics "
          f"({test_df[cfg.split_group_col].nunique()} experiments)")

    label_to_id = {label: index for index, label in enumerate(classifier_class_names)}

    train_df["target_encoded"] = train_df[target].map(label_to_id)
    test_df["target_encoded"] = test_df[target].map(label_to_id)

    n_unmapped = train_df["target_encoded"].isna().sum() + test_df["target_encoded"].isna().sum()
    if n_unmapped > 0:
        print(f"Warning: {n_unmapped} rows have a label not in cfg.classifier_labels, dropping them")
        train_df = train_df.dropna(subset=["target_encoded"])
        test_df = test_df.dropna(subset=["target_encoded"])

    train_df["target_encoded"] = train_df["target_encoded"].astype(int)
    test_df["target_encoded"] = test_df["target_encoded"].astype(int)

    return train_df, test_df


def train_rf(X_train, y_train, cfg) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=cfg.seed,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_and_report(y_true, y_pred, title: str, out_path) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"\n---------------------------- {title} ----------------------------")
    print(f"Accuracy: {accuracy:.3f} | Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"Macro F1: {f1_macro:.3f} | Weighted F1: {f1_weighted:.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=classifier_class_names))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier_class_names)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def report_feature_importance(model, features):
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance)
    return importance


# --------------------------------------------------------------------------- #
# Variant 1: per-image
# --------------------------------------------------------------------------- #

def run_per_image(train_df, test_df, cfg, output_dir):
    X_train, y_train = train_df[filament_features], train_df["target_encoded"]
    X_test, y_test = test_df[filament_features], test_df["target_encoded"]

    model = train_rf(X_train, y_train, cfg)
    y_pred = model.predict(X_test)

    metrics = evaluate_and_report(y_test, y_pred, "Per-image", output_dir / "confusion_matrix_per_image.png")
    report_feature_importance(model, filament_features)
    return metrics, model


# --------------------------------------------------------------------------- #
# Variant 2: per-experiment (mean-aggregated features)
# --------------------------------------------------------------------------- #

def run_per_experiment(train_df, test_df, cfg, output_dir):
    agg_dict = {feat: "mean" for feat in filament_features}
    agg_dict["target_encoded"] = "first"

    train_agg = train_df.groupby(cfg.split_group_col, as_index=False).agg(agg_dict)
    test_agg = test_df.groupby(cfg.split_group_col, as_index=False).agg(agg_dict)

    X_train, y_train = train_agg[filament_features], train_agg["target_encoded"]
    X_test, y_test = test_agg[filament_features], test_agg["target_encoded"]
    print(f"Training experiments: {len(X_train)} | Test experiments: {len(X_test)}")

    model = train_rf(X_train, y_train, cfg)
    y_pred = model.predict(X_test)

    metrics = evaluate_and_report(y_test, y_pred, "Per-experiment (mean-aggregated features)", 
                                  output_dir / "confusion_matrix_per_experiment.png" )
    report_feature_importance(model, filament_features)
    return metrics


# --------------------------------------------------------------------------- #
# Variant 3: per-image trained, majority-vote aggregated to experiment level
# --------------------------------------------------------------------------- #

def majority_vote(labels):
    """Most frequent label in a group. Ties are broken by picking the
    higher class, so it stays consistent with a bias toward not
    under-calling severity. Change to min(...) if you'd rather break
    ties conservatively instead."""
    counts = labels.value_counts()
    top_count = counts.max()
    tied_labels = counts[counts == top_count].index
    return max(tied_labels)


def run_per_image_majority_vote(test_df, model, cfg, output_dir):
    """Reuses the per-image model already trained in run_per_image."""
    X_test = test_df[filament_features]

    test_results = test_df.copy()
    test_results["predicted_encoded"] = model.predict(X_test)

    experiment_pred = (
        test_results.groupby(cfg.split_group_col)["predicted_encoded"]
        .apply(majority_vote)
        .rename("predicted_experiment_label")
    )
    experiment_true = (
        test_results.groupby(cfg.split_group_col)["target_encoded"]
        .first()
        .rename("true_experiment_label")
    )
    experiment_eval = pd.concat([experiment_true, experiment_pred], axis=1)

    metrics = evaluate_and_report(
        experiment_eval["true_experiment_label"], experiment_eval["predicted_experiment_label"],
        "Per-image trained, majority-vote aggregated to experiment",
        output_dir / "confusion_matrix_per_image_majority_vote.png")
    return metrics


def main():
    output_dir = cfg.output_dir / "classification_models" / "QIA_filament_effect_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_fixed_split(cfg)

    # results = {}
    # _, per_image_model = run_per_image(train_df, test_df, cfg, output_dir)
    # results["per_experiment"] = run_per_experiment(train_df, test_df, cfg, output_dir)
    # results["per_image_majority_vote"] = run_per_image_majority_vote(test_df, per_image_model, cfg, output_dir)

    # summary_df = pd.DataFrame(results).T
    # summary_path = output_dir / "summary_metrics.csv"
    # summary_df.to_csv(summary_path)
    # print(f"\nSaved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()