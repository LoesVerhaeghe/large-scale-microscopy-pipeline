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
    "eccentricity",
    "aspect_ratio",
    "compactness",
    "mean_floc_crofton_perimeter_um",
    "mean_major_axis_um",
    "mean_minor_axis_um",
    "mean_floc_diameter_um",
    "mean_floc_area_um2",
    "n_flocs",
    "fraction_microflocs"
]


target =     "Structuur (STRU_VMF_2)" # Diffuus / Compact
# target = "Vorm (VORM_VMF_2)"  # Agglomeraten / Onregelmatig / Afgerond / 
# target = "Stevigheid (STEV_VMF_2)" #Sterk / Zwak"

data = match_table_w_metrics[features + [target]].copy()

print(data[target].value_counts(dropna=False))
print(data.head())
print(data.isna().sum())

# Remove rows with wrong or missing target values
data = data.dropna(subset=[target])

# Encode target variable
class_order = {
    "Diffuus": 1,
    "Compact": 2
}
# class_order = {
#     "Agglomeraten": 1,
#     "Onregelmatig": 2,
#     "Afgerond": 3
# }
# class_order = {
#     "Sterk": 1,
#     "Zwak": 2
# }

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
        target_names=["Diffuus", "Compact"] #["Agglomeraten", "Onregelmatig", "Afgerond"] # ["Sterk", "Zwak"]
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Diffuus", "Compact"] #["Agglomeraten", "Onregelmatig", "Afgerond"] # ["Sterk", "Zwak"]
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



############
# RF based on one feature per group of images (grouped by order_nr) to predict technician labels based on QIA metrics
#################

# Load/organize QIA metrics + technician labels
match_table_w_metrics = pd.read_excel("match_table_extended_ph100_masks_with_metrics_v2.xlsx")

agg_dict = {
    "eccentricity": "max",
    "aspect_ratio": "max",
    "compactness": "max",
    "mean_floc_crofton_perimeter_um": "max",
    "mean_major_axis_um": "max",
    "mean_minor_axis_um": "max",
    "mean_floc_diameter_um": "max",
    "mean_floc_area_um2": "max",
    "n_flocs": "max",
    "fraction_microflocs": "max",

    # Utility measurements
    "Structuur (STRU_VMF_2)": "first" # Diffuus / Compact
    #"Vorm (VORM_VMF_2)": "first"  # Agglomeraten / Onregelmatig / Afgerond / 
    # "Stevigheid (STEV_VMF_2)": "first", #Sterk / Zwak
}


sample_df = (
    match_table_w_metrics
    .groupby("order_nr", as_index=False)
    .agg(agg_dict)
)
features = [
    "eccentricity",
    "aspect_ratio",
    "compactness",
    "mean_floc_crofton_perimeter_um",
    "mean_major_axis_um",
    "mean_minor_axis_um",
    "mean_floc_diameter_um",
    "mean_floc_area_um2",
    "n_flocs",
    "fraction_microflocs"
]

target =     "Structuur (STRU_VMF_2)" # Diffuus / Compact
# target = "Vorm (VORM_VMF_2)"  # Agglomeraten / Onregelmatig / Afgerond / 
# target = "Stevigheid (STEV_VMF_2)" #Sterk / Zwak"


data = sample_df[features + [target]].copy()

print(data[target].value_counts(dropna=False))
print(data.head())
print(data.isna().sum())

# Remove rows with wrong or missing target values
data = data.dropna(subset=[target])

# Encode target variable
class_order = {
    "Diffuus": 1,
    "Compact": 2
}
# class_order = {
#     "Agglomeraten": 1,
#     "Onregelmatig": 2,
#     "Afgerond": 3
# }
# class_order = {
#     "Sterk": 1,
#     "Zwak": 2
# }

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
        target_names=["Diffuus", "Compact"] #["Agglomeraten", "Onregelmatig", "Afgerond"] # ["Sterk", "Zwak"]
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Diffuus", "Compact"] #["Agglomeraten", "Onregelmatig", "Afgerond"] # ["Sterk", "Zwak"]
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