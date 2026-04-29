import joblib, json
import pandas as pd
from sklearn.svm import SVC

#load data and parameters
X_train_pca = pd.read_csv(snakemake.input.X_train_pca, index_col=0)
y_train = pd.read_csv(snakemake.input.y_train, index_col=0)['Class']
with open(snakemake.input.best_params) as f:
    best_params = json.load(f)

#define model and train
svm = SVC(
    **best_params,
    probability=True, 
    random_state = snakemake.params.seed
    )
svm.fit(X_train_pca, y_train)

#save model
joblib.dump(svm, snakemake.output.model)

with open(snakemake.log[0], 'w') as log:
    log.write(f'Kernel: {svm.kernel}\n')
    log.write(f'Support vectors per class: {dict(zip(svm.classes_, svm.n_support_))}\n')
    log.write(f'Training samples: {X_train_pca.shape[0]}\n')
    log.write(f'Features: {X_train_pca.shape[1]}\n')