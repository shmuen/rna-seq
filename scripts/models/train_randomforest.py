import joblib, json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#load data and parameters
X_train_pca = pd.read_csv(snakemake.input.X_train_pca, index_col=0)
y_train = pd.read_csv(snakemake.input.y_train, index_col=0)['Class']
with open(snakemake.input.best_params) as f:
    best_params = json.load(f)

#define model and train
rf = RandomForestClassifier(
    **best_params, 
    random_state=snakemake.params.seed, 
    n_jobs = snakemake.threads
    )
rf.fit(X_train_pca, y_train)

#save model
joblib.dump(rf, snakemake.output.model)

with open(snakemake.log[0], 'w') as log:
    log.write(f'Number of trees: {rf.n_estimators}\n')
    log.write(f'Max depth: {rf.max_depth}\n')
    log.write(f'Feature importances (Top 5): {sorted(zip(X_train_pca.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)[:5]}\n')
    log.write(f'Training samples: {X_train_pca.shape[0]}\n')
    log.write(f'Features: {X_train_pca.shape[1]}\n')