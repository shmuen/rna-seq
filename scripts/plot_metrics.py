import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import ConfusionMatrixDisplay



#load y_test, y_proba, cm and roc data
cm = pd.read_csv(snakemake.input.cm).values
roc = pd.read_csv(snakemake.input.roc)

#get mapping as labels for cm plot
with open(snakemake.input.mapping) as f:
    label_mapping = json.load(f)
inv_label_dict = {v: k for k, v in label_mapping.items()}
class_names = [inv_label_dict[i] for i in sorted(inv_label_dict.keys())]

#Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(cmap=plt.cm.Blues)
plt.title("confusion matrix - " + snakemake.wildcards.model)
plt.savefig(snakemake.output.cm_plot, dpi= 150)

#plot roc
plt.figure()
# fig, ax = plt.subplots()
for cls in roc["class"].unique():
    subset = roc[roc["class"] == cls]
    auc_score = subset["auc"].iloc[0]
    class_name = class_names[cls]
    plt.plot(subset["fpr"], subset["tpr"], label=f"{class_name} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC (OvR) - " + snakemake.wildcards.model)
plt.legend()

plt.savefig(snakemake.output.roc_plot)