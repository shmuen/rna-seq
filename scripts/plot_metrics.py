import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import json
from sklearn.metrics import ConfusionMatrixDisplay


#load y_test, y_proba, cm and roc data
cm_df = pd.read_csv(snakemake.input.cm, index_col=0)
class_names = cm_df.columns.tolist()
cm = cm_df.values
roc = pd.read_csv(snakemake.input.roc)

#get mapping as labels for cm plot
# with open(snakemake.input.mapping) as f:
#     label_mapping = json.load(f)
# inv_label_dict = {v: k for k, v in label_mapping.items()}
# class_names = [inv_label_dict[i] for i in sorted(inv_label_dict.keys())]

#Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(cmap=plt.cm.Blues)
plt.title("confusion matrix - " + snakemake.wildcards.model)
plt.savefig(snakemake.output.cm_plot, dpi= 150)

#plot auc
plt.figure()
for cls in roc["class"].unique():
    subset = roc[roc["class"] == cls]
    auc_score = subset["auc"].iloc[0]
    plt.plot(subset["fpr"], subset["tpr"], label=f"{cls} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC (OvR) - " + snakemake.wildcards.model)
plt.legend()

plt.savefig(snakemake.output.roc_plot)
plt.close()

#calibration curve
calibration = pd.read_csv(snakemake.input.calibration)

classes = calibration["class"].unique()
colors = plt.colormaps["tab10"](np.linspace(0, 1, len(classes)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Calibration Curve
ax1 = axes[0]
ax1.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")

for cls, color in zip(classes, colors):
    sub = calibration[calibration["class"] == cls]
    ax1.plot(
        sub["mean_predicted_value"],
        sub["fraction_of_positives"],
        marker="o", label=cls, color=color
    )

ax1.set_xlabel("mean predicted value")
ax1.set_ylabel("fraction of positives")
ax1.set_title("Calibration Curve")
ax1.legend()
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

y_proba = pd.read_csv(snakemake.input.y_proba, index_col=0).values
confidence = y_proba.max(axis=1)  # hightest probability for samples

ax2 = axes[1]
ax2.hist(confidence, bins=20, color="steelblue", edgecolor="white")
ax2.set_xlabel("Confidence")
ax2.set_ylabel("amount of samples")
ax2.set_title("Confidence Distribution")
ax2.axvline(confidence.mean(), color="red", linestyle="--",
            label=f"mean: {confidence.mean():.2f}")
ax2.legend()

plt.tight_layout()
plt.savefig(snakemake.output.calibration_curve, dpi=150)
plt.close()