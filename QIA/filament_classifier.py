import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
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

from sklearn.metrics import f1_score

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

from sklearn.metrics import f1_score

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