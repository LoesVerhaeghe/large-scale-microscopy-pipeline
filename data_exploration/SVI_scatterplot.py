import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes
from pathlib import Path

# Load data
csv_path = Path("data/analysis/mergedModels/Results/DINO/SVI_head1/selection_predictions.csv")
df = pd.read_csv(csv_path)

# Keep valid predictions and ground truths
df = df[["SVI", "prediction_SVI"]].replace([np.inf, -np.inf], np.nan).dropna()

gt = df["SVI"].to_numpy()
pred = df["prediction_SVI"].to_numpy()

# Theil-Sen regression: pred as a function of gt

slope, intercept, low_slope, high_slope = theilslopes(pred, gt)

# Common axis limits for the 1:1 line
axis_min = min(gt.min(), pred.min())
axis_max = max(gt.max(), pred.max())
padding = 0.05 * (axis_max - axis_min)
axis_min -= padding
axis_max += padding


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

# Plot
fig, ax = plt.subplots(figsize=(7, 7), dpi=600)

ax.scatter(gt, pred, alpha=0.7, edgecolors="black", linewidths=0.4)