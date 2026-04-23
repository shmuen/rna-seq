import joblib
import pandas as pd
from xgboost import XGBClassifier

#load data
X_train_pca = pd.read_csv(snakemake.input.X_train_pca, index_col=0)
y_train = pd.read_csv(snakemake.input.y_train, index_col=0)['Class']

#define model and train
xgb = XGBClassifier(
    n_estimators = 200, 
    eval_metric = 'mlogloss', 
    random_state = snakemake.params.seed,
    n_jobs = snakemake.threads)
xgb.fit(X_train_pca, y_train)

#save model
joblib.dump(xgb, snakemake.output.model)

with open(snakemake.log[0], 'w') as log:
    log.write(f'Number of trees: {xgb.n_estimators}\n')
    log.write(f'Learning rate: {xgb.learning_rate}\n')
    log.write(f'Training samples: {X_train_pca.shape[0]}\n')
    log.write(f'Features: {X_train_pca.shape[1]}\n')