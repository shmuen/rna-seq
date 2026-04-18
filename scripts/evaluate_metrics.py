import pandas as pd
import json, joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

#import data
y_test = pd.read_csv(snakemake.input.y_test)['Class']
y_pred = pd.read_csv(snakemake.input.y_pred)
y_proba = pd.read_csv(snakemake.input.y_proba).values
model = joblib.load(snakemake.input.model)

#create and save report of metrics
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
report.to_csv(snakemake.output.report)

#calculate confusion matrix
cm =confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)
cm_df.to_csv(snakemake.output.cm, index = False)

#roc 
#bin y_test data
y_test_bin = label_binarize(y_test, classes = model.classes_)

roc_data = []
for i, cls in enumerate(model.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_proba[:,i])
    auc_score = auc(fpr, tpr)
    for f, t in zip(fpr, tpr):
        roc_data.append({"class": cls, "fpr": f, "tpr": t, "auc": auc_score})

#save roc
roc_df = pd.DataFrame(roc_data)
roc_df.to_csv(snakemake.output.roc, index = False)
