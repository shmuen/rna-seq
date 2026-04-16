import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#import data and model
X_test = pd.read_csv(snakemake.input.X_test)
y_test = pd.read_csv(snakemake.input.y_test)['Class']
scaler = joblib.load(snakemake.input.scaler)
pca = joblib.load(snakemake.input.pca)
model = joblib.load(snakemake.input.model)

X_scaled = scaler.transform(X_test)
X_pca = pca.transform(X_scaled)

y_pred = model.predict(X_pca)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_macro": f1_score(y_test, y_pred, average="macro"),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    "auc":  roc_auc_score(y_test, model.predict_proba(X_pca),
                          multi_class='ovr', average='macro')
}

pd.DataFrame([metrics]).to_csv(snakemake.output.metrics, index=False)

