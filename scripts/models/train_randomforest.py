import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

X_train_pca = pd.read_csv(snakemake.input.X_train_pca)
y_train = pd.read_csv(snakemake.input.y_train)['Class']

rf = RandomForestClassifier(n_estimators=200, random_state=snakemake.params.seed)
rf.fit(X_train_pca, y_train)
joblib.dump(rf, snakemake.output.model)