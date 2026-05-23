import pandas as pd
import json, shap
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.stdout = open(snakemake.log[0], "w")
sys.stderr = sys.stdout

X_test_scaled = pd.read_csv(snakemake.input.X_test_scaled, index_col=0)
shap_values_arr = np.load(snakemake.input.shap_values)
shap_values_list = [shap_values_arr[i] for i in range(shap_values_arr.shape[0])]

# load mapping and decode
with open(snakemake.input.label_mapping) as f:
    label_mapping = json.load(f)
inv_map = {int(k): v for k, v in label_mapping.items()}

# summary plot
shap.summary_plot(
    shap_values_list,          
    X_test_scaled,
    plot_type="bar",
    class_names=[inv_map[i] for i in sorted(inv_map)],
    show=False
)
plt.savefig(snakemake.output.shap_summary, dpi=150, bbox_inches="tight")
plt.close()

# beeswarm plots
for path, i in zip(snakemake.output.beeswarm, sorted(inv_map)):
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

# get gene names
gene_names = X_test_scaled.columns.tolist()
model_name = snakemake.wildcards.model

# top Gene barplot
for path, i in zip(snakemake.output.barplot, sorted(inv_map)):
    name = inv_map[i]
    mean_per_gene = np.abs(shap_values_list[i]).mean(axis=0)  
    top_idx = np.argsort(mean_per_gene)[-20:]    

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [gene_names[j] for j in top_idx],
        mean_per_gene[top_idx]
    )
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Top Genes - {name}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()