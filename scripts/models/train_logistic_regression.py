import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

X_train = pd.read_csv(snakemake.input.X_train)
y_train = pd.read_csv(snakemake.input.y_train)['Class']

lr = LogisticRegression(max_iter = 1000, random_state = snakemake.params.seed)
lr.fit(X_train.values, y_train)
joblib.dump(lr, snakemake.output.model)