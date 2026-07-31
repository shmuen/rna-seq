import pandas as pd
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, cohen_kappa_score, matthews_corrcoef
    )

#import data
y_test = pd.read_csv(snakemake.input.y_test, index_col=0)['Class']
y_pred = pd.read_csv(snakemake.input.y_pred, index_col=0)['y_pred']
y_proba = pd.read_csv(snakemake.input.y_proba, index_col=0).values # numpy array for sklearn

#load model
model = joblib.load(snakemake.input.model)
classes = model.classes_

#create and save report of metrics
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T

#get accuracy for separate output for model metrics and remove accuracy from report
accuracy = report["precision"]["accuracy"]
report = report.drop("accuracy")
report.to_csv(snakemake.output.report)

#cohen-kappa-score and MCC as additional metrics
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

#combine accuracy, kappa and mcc as model summary and save to file
model_summary = pd.DataFrame({"accuracy": [accuracy], "kappa": [kappa], "mcc": [mcc]})
model_summary.to_csv(snakemake.output.summary)

#calculate confusion matrix
cm =confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.to_csv(snakemake.output.cm)

#roc 
#bin y_test data
y_test_bin = label_binarize(y_test, classes = classes)

#ROC curve: One-vs-Rest for each class
roc_data = []
for cls_idx, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:,cls_idx], y_proba[:,cls_idx])
    auc_score = auc(fpr, tpr)
    for f, t in zip(fpr, tpr):
        roc_data.append({"class": cls, "fpr": f, "tpr": t, "auc": auc_score})

#save roc
roc_df = pd.DataFrame(roc_data)
roc_df.to_csv(snakemake.output.roc, index = False)

#calibration curve per class (OvR)
rows = []
for cls_idx, cls in enumerate(classes):
    frac_pos, mean_pred = calibration_curve(
        (y_test == cls).astype(int),
        y_proba[:, cls_idx],
        n_bins = 10
    )
    for fp, mp in zip(frac_pos, mean_pred):
        rows.append({
            "class": cls,
            "fraction_of_positives": fp,
            "mean_predicted_value": mp
        })

df = pd.DataFrame(rows)
df.to_csv(snakemake.output.calibration, index = False)

