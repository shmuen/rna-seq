"""
SHAP analysis on original (scaled) features, without PCA, so that SHAP
values map back to gene names.
"""
import pandas as pd
import json, shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import sys
sys.stdout = open(snakemake.log[0], "w")
sys.stderr = sys.stdout

X_train_scaled = pd.read_csv(snakemake.input.X_train_scaled, index_col=0)
X_test_scaled = pd.read_csv(snakemake.input.X_test_scaled, index_col=0)
y_train = pd.read_csv(snakemake.input.y_train, index_col=0).squeeze()

# load mapping and decode
with open(snakemake.input.label_mapping) as f:
    label_mapping = json.load(f)
inv_map = {int(k): v for k, v in label_mapping.items()}

# load parameters
with open(snakemake.input.best_params) as f:
    best_params = json.load(f)
best_params["random_state"] = snakemake.params.seed

MODELS = {
    "randomforest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "xgboost": XGBClassifier
}

# instantiate and train model
model_name = snakemake.wildcards.model

model = MODELS[model_name](**best_params)

# retrain model with scaled data (without PCA) so that SHAP values map to gene names
model.fit(X_train_scaled, y_train)

# choose correct SHAP explainer
if snakemake.wildcards.model in ["randomforest", "xgboost"]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

elif snakemake.wildcards.model == "logistic_regression":
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

elif snakemake.wildcards.model == "svm":
    # SVM has no native SHAP support; KernelExplainer approximates values
    # by masking features (model-agnostic). background = reference baseline,
    # nsamples = feature combinations sampled per observation.    
    background = shap.sample(X_train_scaled, 50)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test_scaled, nsamples=100)

# create list with shap_values (different formats for different Explainer)
# normalize shap_values to a list of 2D arrays, one per class.
# different explainers return different formats (list or 3D array)
if isinstance(shap_values, list):
    shap_values_list = shap_values
elif shap_values.ndim == 3:
    shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
else:
    raise ValueError(f"Unexpected shap_values shape: {shap_values.shape} for model {model_name}")

np.save(snakemake.output.shap_values, np.stack(shap_values_list))

# save top 20 genes
stacked = np.stack(shap_values_list, axis=0)   
mean_abs = np.abs(stacked).mean(axis=(0, 1))
pd.DataFrame({
    "gene":          X_test_scaled.columns,
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False).head(20)\
  .to_csv(snakemake.output.top_genes, index=False)