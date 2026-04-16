import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report

#import data and model
X_test = pd.read_csv(snakemake.input.X_test)
y_test = pd.read_csv(snakemake.input.y_test)['Class']
scaler = joblib.load(snakemake.input.scaler)
pca = joblib.load(snakemake.input.pca)
model = joblib.load(snakemake.input.model)

X_scaled = scaler.transform(X_test)
X_pca = pca.transform(X_scaled)

y_pred = model.predict(X_pca)

report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
report.to_csv(snakemake.output.report)
