import pandas as pd
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
import shap

#import data and model
X_test_pca = pd.read_csv(snakemake.input.X_test_pca)
y_test = pd.read_csv(snakemake.input.y_test)['Class']
n_components = snakemake.params.n_components
model = joblib.load(snakemake.input.model)

#predict y for test data
y_pred = model.predict(X_test_pca)

#create and save report of metrics
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
report.to_csv(snakemake.output.report)

cm =confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)
cm_df.to_csv(snakemake.output.cm)

#get mapping as labels for cm plot
with open(snakemake.input.mapping) as f:
    label_mapping = json.load(f)
inv_label_dict = {v: k for k, v in label_mapping.items()}
class_names = [inv_label_dict[i] for i in sorted(inv_label_dict.keys())]

#Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(cmap=plt.cm.Blues)
plt.title("Konfusionsmatrix")
plt.savefig(snakemake.output.cm_plot, dpi= 150)
