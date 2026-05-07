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
from sklearn.model_selection import StratifiedKFold, cross_val_score

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=snakemake.params.seed)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=snakemake.params.seed)


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
inner = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="f1_macro",
    n_jobs=1
    )

outer_scores = cross_val_score(
    inner, 
    X_train, 
    y_train, 
    cv=outer_cv, 
    scoring="f1_macro",
    n_jobs=snakemake.threads)

with open(snakemake.log[0], 'w') as log:
    log.write(f'Mean f1-macro: {outer_scores.mean():.3f}\n')
    log.write(f'Std f1-macro {outer_scores.std():.3f}\n')

pd.DataFrame({
    "mean": [outer_scores.mean()],
    "std": [outer_scores.std()]
}).to_csv(snakemake.output.nested_cv_score, index=False)