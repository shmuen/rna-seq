"""
SHAP analysis on original (scaled) features, without PCA, so that SHAP
values mpa back to gene names.
"""
import pandas as pd
import json, shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
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

# get gene names
gene_names = X_train_scaled.columns.tolist()

MODELS = {
    "randomforest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "xgboost": XGBClassifier
}

# instantiate and train model
model_name = snakemake.wildcards.model

model = MODELS[model_name](**best_params)

# retrain model with scaled data (without PCA)
model.fit(X_train_scaled, y_train)

# chose right SHAP explainer
if snakemake.wildcards.model in ["randomforest", "xgboost"]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

elif snakemake.wildcards.model == "logistic_regression":
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

elif snakemake.wildcards.model == "svm":
    background = shap.sample(X_train_scaled, 25)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test_scaled, nsamples=100)


shap_mean = np.abs(np.array(shap_values)).mean(axis=0)

if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
else:
    shap_values_list = shap_values

# summary plot
shap.summary_plot(
    shap_values_list,          
    X_test_scaled,
    plot_type="bar",
    class_names=list(label_mapping.values()),
    show=False
)
plt.savefig(snakemake.output.shap_summary, dpi=150, bbox_inches="tight")
plt.close()

for i in range(len(inv_map)):
    name = inv_map[i]
    shap.summary_plot(shap_values_list[i], X_test_scaled,
                      max_display=20, show=False)
    plt.title(f"SHAP – {name}")
    plt.savefig(f"results/shap/{model_name}_{name}_beeswarm.png",
                dpi=150, bbox_inches="tight")
    plt.close()

for path, i in zip(snakemake.output.beeswarm, inv_map):
    name = inv_map[i]
    shap.summary_plot(
        shap_values_list[i], 
        X_test_scaled, 
        max_display=20, 
        show=False
        )
    plt.title(f"SHAP - {name}")
    plt.savefig(path, dpi=150, bbox_inches="tight")  
    plt.close()

# top Gene
stacked = np.stack(shap_values_list, axis=0)   
mean_abs = np.abs(stacked).mean(axis=(0, 1))
pd.DataFrame({
    "gene":          X_test_scaled.columns,
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False).head(20)\
  .to_csv(snakemake.output.top_genes, index=False)

for i, name in zip(range(len(inv_map)), inv_map.values()):
    mean_per_gene = np.abs(shap_values_list[i]).mean(axis=0)  # (n_features,)
    top_idx = np.argsort(mean_per_gene)[-20:]                 # Top 20

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [gene_names[j] for j in top_idx],
        mean_per_gene[top_idx]
    )
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Top Genes - {name}")
    plt.tight_layout()
    plt.savefig(f"results/shap/{model_name}_{name}_barplot.png", dpi=150)
    plt.close()