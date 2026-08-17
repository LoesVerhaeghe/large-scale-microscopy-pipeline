import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

############# 
# RF based on all images (not grouped by order_nr) to predict technician labels based on QIA metrics
###############

# Load/organize QIA metrics + technician labels
match_table_w_metrics = pd.read_excel("match_table_extended_ph100_masks_with_metrics_v2.xlsx")

features = [
    "total_filament_area",
    "mean_filament_length",
    "median_filament_length",
    "filament_to_floc_ratio"
]

target = "Semikwantitatieve beoordeling (SSCO_FIG_2)"

data = match_table_w_metrics[features + [target]].copy()

print(data[target].value_counts(dropna=False))
print(data.head())
print(data.isna().sum())

# Remove rows with wrong or missing target values
data = data[data[target] != "Weinig"].copy()
data = data.dropna(subset=[target])

# Encode target variable
class_order = {
    "Matig": 1,
    "Veel": 2,
    "Zeer veel": 3
}

data["target_encoded"] = data[target].map(class_order)

print(data[[target, "target_encoded"]].drop_duplicates())

# Prepare train and test data
X = data[features]
y = data["target_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())

# Define the Random Forest model

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42, 
    n_jobs=2
)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\n----------------------------")
print("MODEL PERFORMANCE")
print("----------------------------")

print(f"Accuracy: {accuracy:.3f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Matig", "Veel", "Zeer veel"]
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Matig", "Veel", "Zeer veel"]
)

f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("Macro F1:", f1_macro)
print("Weighted F1:", f1_weighted)

disp.plot()
plt.tight_layout()
plt.show()

# check feature importance

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)



############
# RF based on one feature per group of images (grouped by order_nr) to predict technician labels based on QIA metrics
#################

# Load/organize QIA metrics + technician labels
match_table_w_metrics = pd.read_excel("match_table_extended_ph100_masks_with_metrics_v2.xlsx")

agg_dict = {
    "total_filament_area": "mean",
    "mean_filament_length": "mean",
    "median_filament_length": "mean",
    "filament_to_floc_ratio": "mean",

    # Utility measurements
    "Semikwantitatieve beoordeling (SSCO_FIG_2)": "first" # veel matig zeer veel weinig
}

sample_df = (
    match_table_w_metrics
    .groupby("order_nr", as_index=False)
    .agg(agg_dict)
)

features = [
    "total_filament_area",
    "mean_filament_length",
    "median_filament_length",
    "filament_to_floc_ratio"
]

target = "Semikwantitatieve beoordeling (SSCO_FIG_2)"

data = sample_df[features + [target]].copy()

print(data[target].value_counts(dropna=False))
print(data.head())
print(data.isna().sum())

# Remove rows with wrong or missing target values
data = data[data[target] != "Weinig"].copy()
data = data.dropna(subset=[target])

# Encode target variable
class_order = {
    "Matig": 1,
    "Veel": 2,
    "Zeer veel": 3
}

data["target_encoded"] = data[target].map(class_order)

print(data[[target, "target_encoded"]].drop_duplicates())

# Prepare train and test data
X = data[features]
y = data["target_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())

# Define the Random Forest model

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42, 
    n_jobs=2
)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\n----------------------------")
print("MODEL PERFORMANCE")
print("----------------------------")

print(f"Accuracy: {accuracy:.3f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Matig", "Veel", "Zeer veel"]
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Matig", "Veel", "Zeer veel"]
)

disp.plot()
plt.tight_layout()
plt.show()

f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("Macro F1:", f1_macro)
print("Weighted F1:", f1_weighted)

# check feature importance

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)



from sklearn.model_selection import GroupShuffleSplit

###############################################################################################################
# RF trained/predicted per image, then aggregated to experiment level
# by taking the MAX predicted label among the images of each order_nr
#################

match_table_w_metrics = pd.read_excel("match_table_extended_ph100_masks_with_metrics_v2.xlsx")

features = [
    "total_filament_area",
    "mean_filament_length",
    "median_filament_length",
    "filament_to_floc_ratio"
]

target = "Semikwantitatieve beoordeling (SSCO_FIG_2)"

data = match_table_w_metrics[features + [target, "order_nr"]].copy()

# Remove rows with wrong or missing target values
data = data[data[target] != "Weinig"].copy()
data = data.dropna(subset=[target])

# Encode target variable
class_order = {
    "Matig": 1,
    "Veel": 2,
    "Zeer veel": 3
}
data["target_encoded"] = data[target].map(class_order)

# Split by order_nr (group), so all images of one experiment
# stay together in either train or test -> no leakage when we
# re-aggregate predictions to the experiment level afterwards
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(data, groups=data["order_nr"]))

train_data = data.iloc[train_idx]
test_data = data.iloc[test_idx]

X_train, y_train = train_data[features], train_data["target_encoded"]
X_test, y_test = test_data[features], test_data["target_encoded"]

print("Training images:", len(X_train), "| unique experiments:", train_data["order_nr"].nunique())
print("Test images:", len(X_test), "| unique experiments:", test_data["order_nr"].nunique())

# Train RF on individual images
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=2
)
model.fit(X_train, y_train)

# Per-image predictions on the test set
y_pred_image = model.predict(X_test)

# print("\n----------------------------")
# print("PER-IMAGE PERFORMANCE (before aggregation)")
# print("----------------------------")
# print(classification_report(
#     y_test, y_pred_image, labels=[1, 2, 3],
#     target_names=["Matig", "Veel", "Zeer veel"]
# ))

# Attach predictions back to the test rows
test_results = test_data.copy()
test_results["predicted_encoded"] = y_pred_image

# # Aggregate to experiment level: max predicted label per order_nr
# experiment_pred = (
#     test_results
#     .groupby("order_nr")["predicted_encoded"]
#     .max()
#     .rename("predicted_experiment_label")
# )


############
# Same per-image predictions, aggregated to experiment level via
# MAJORITY VOTE instead of max
#################
 
def majority_vote(labels):
    """Most frequent label in a group. Ties are broken by picking the
    higher class, so it stays consistent with the 'max' method's bias
    toward not under-calling severity. Change to min(...) if you'd
    rather break ties conservatively instead."""
    counts = labels.value_counts()
    top_count = counts.max()
    tied_labels = counts[counts == top_count].index
    return max(tied_labels)
 
experiment_pred_majority = (
    test_results
    .groupby("order_nr")["predicted_encoded"]
    .apply(majority_vote)
    .rename("predicted_experiment_label")
)

# True experiment-level label (same for every image of that order_nr)
experiment_true = (
    test_results
    .groupby("order_nr")["target_encoded"]
    .first()
    .rename("true_experiment_label")
)

experiment_eval = pd.concat([experiment_true, experiment_pred_majority], axis=1)


# Evaluation

accuracy = accuracy_score(experiment_eval["true_experiment_label"],experiment_eval["predicted_experiment_label"])
balanced_accuracy = balanced_accuracy_score(experiment_eval["true_experiment_label"],experiment_eval["predicted_experiment_label"])

print("\n----------------------------")
print("MODEL PERFORMANCE")
print("----------------------------")

print(f"Accuracy: {accuracy:.3f}")

print("\nClassification report:")
print(
    classification_report(
        experiment_eval["true_experiment_label"],
        experiment_eval["predicted_experiment_label"],
        target_names=["Matig", "Veel", "Zeer veel"]
    )
)

# Confusion matrix
cm = confusion_matrix(experiment_eval["true_experiment_label"],experiment_eval["predicted_experiment_label"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Matig", "Veel", "Zeer veel"]
)

disp.plot()
plt.tight_layout()
plt.show()

f1_macro = f1_score(experiment_eval["true_experiment_label"],experiment_eval["predicted_experiment_label"], average="macro")
f1_weighted = f1_score(experiment_eval["true_experiment_label"],experiment_eval["predicted_experiment_label"], average="weighted")

print("Macro F1:", f1_macro)
print("Weighted F1:", f1_weighted)

# Feature importance (same model as used for per-image predictions)
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)