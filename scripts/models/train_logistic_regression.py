import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


#load data
X_train_pca = pd.read_csv(snakemake.input.X_train_pca, index_col=0)
y_train = pd.read_csv(snakemake.input.y_train, index_col=0)['Class']

#define model and train
lr = LogisticRegression(
    max_iter = 1000, 
    random_state = snakemake.params.seed,
    )
lr.fit(X_train_pca, y_train)

#save model
joblib.dump(lr, snakemake.output.model)

with open(snakemake.log[0], 'w') as log:
    log.write(f'Convergence: {lr.n_iter_[0]} iterations (max: {lr.max_iter})\n')
    log.write(f'Classes: {list(lr.classes_)}\n')
    log.write(f'Training samples: {X_train_pca.shape[0]}\n')
    log.write(f'Features: {X_train_pca.shape[1]}\n')