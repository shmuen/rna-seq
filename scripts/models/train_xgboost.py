import joblib
import pandas as pd
from xgboost import XGBClassifier

X_train_pca = pd.read_csv(snakemake.input.X_train_pca)
y_train = pd.read_csv(snakemake.input.y_train)['Class']

xgb = XGBClassifier(n_estimators = 200, eval_metric = 'mlogloss', random_state = snakemake.params.seed)
xgb.fit(X_train_pca.values, y_train)
joblib.dump(xgb, snakemake.output.model)