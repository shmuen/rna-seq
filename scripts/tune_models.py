import pandas as pd
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

if snakemake.params.tune:
    #load data
    X_train = pd.read_csv(snakemake.input.X_train, index_col=0)
    y_train = pd.read_csv(snakemake.input.y_train, index_col=0)['Class']

    #arrange dictionary with models
    MODELS = {
        "randomforest": RandomForestClassifier(random_state=snakemake.params.seed),
        "logistic_regression": LogisticRegression(random_state=snakemake.params.seed),
        "svm": SVC(probability=True, random_state=snakemake.params.seed),
        "xgboost": XGBClassifier(random_state=snakemake.params.seed)
    }

    #get current model and use respective model for grid search
    model_name = snakemake.wildcards.model
    estimator = MODELS[model_name]

    #get grid parameters and add clf prefix
    raw_grid = snakemake.params.param_grid
    param_grid = {f"clf__{k}": v for k, v in raw_grid.items()}

    #create pipeline with scaling and PCA to avoid data leakage in grid search
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=snakemake.params.n_components)),
        ('clf', estimator)
    ])

    #perform grid search and fit model
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc_ovr",
        n_jobs=snakemake.threads)
    search.fit(X_train, y_train)

    with open(snakemake.log[0], 'w') as log:
        log.write(f'Best parameters{search.best_params_}\n')
        log.write(f'Mean ROC-AUC for all 5 folds {search.best_score_}\n')

    #remove prefix
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

else: 
    best_params = snakemake.params.default_grid

#store best parameters as json
with open(snakemake.output.best_params, 'w') as f:
    json.dump(best_params, f)