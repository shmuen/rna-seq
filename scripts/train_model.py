import joblib, json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

#load data and parameters
X_train_pca = pd.read_csv(snakemake.input.X_train_pca, index_col=0)
y_train = pd.read_csv(snakemake.input.y_train, index_col=0)['Class']
with open(snakemake.input.best_params) as f:
    best_params = json.load(f)
best_params["random_state"] = snakemake.params.seed

MODELS = {
    "randomforest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "xgboost": XGBClassifier
}

# define current model and train with best_params either from tuning or 
# predefined parameters if tuning is disabled
model_name = snakemake.wildcards.model
ModelClass = MODELS[model_name]

model = ModelClass(**best_params)

model.fit(X_train_pca, y_train)

#save model
joblib.dump(model, snakemake.output.model)

#create log with individual output per model
with open(snakemake.log[0], 'w') as log:
    log.write(f'Training samples: {X_train_pca.shape[0]}\n')
    log.write(f'Features: {X_train_pca.shape[1]}\n')
    if model_name == 'logistic_regression':
       log.write(f'Convergence: {model.n_iter_[0]} iterations (max: {model.max_iter})\n')
       log.write(f'Classes: {list(model.classes_)}\n')
    elif model_name == 'randomforest':
        log.write(f'Number of trees: {model.n_estimators}\n')
        log.write(f'Max depth: {model.max_depth}\n')
        log.write(f'Feature importances (Top 5): {sorted(zip(X_train_pca.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]}\n')
    elif model_name == 'svm':
        log.write(f'Kernel: {model.kernel}\n')
        log.write(f'Support vectors per class: {dict(zip(model.classes_, model.n_support_))}\n')
    elif model_name == 'xgboost':
        log.write(f'Number of trees: {model.n_estimators}\n')
        log.write(f'Learning rate: {model.learning_rate}\n')
  