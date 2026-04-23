import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#list with names of metrics to compare
metrics = ["accuracy", "macro avg", "mcc", "kappa"]

#iterate over all models, get metrics, the time benchmark and macro AUC
rows = []
for model, bench, roc in zip(snakemake.input.reports, snakemake.input.bench, snakemake.input.roc):
    model_name = Path(model).stem.replace("_report", "")
    tmp = pd.read_csv(model, index_col =0)
    row = tmp.loc[metrics,["f1-score"]].T
    row.index = [model_name]
    #add training time from benchmark file
    row["train time"] = pd.read_csv(bench,sep="\t")["s"].item()
    #compute macro AUC by averaging AUC across classes
    row["auc_macro"] = pd.read_csv(roc).groupby("class")["auc"].first().mean()
    rows.append(row)
    
#create summary dataframe with all models and save to file
df_summary = pd.concat(rows).rename(
    columns={
        "accuracy": "Accuracy",
        "macro avg": "Macro F1",
        "mcc": "MCC",
        "auc_macro": "Macro AUC"
    },
    index={
        "randomforest": "Random Forest",
        "svm": "SVM",
        "logistic_regression": "Logistic Regression",
        "xgboost": "XGBoost"    
    }
)

df_summary.round(3).to_csv(snakemake.output.summary)

#dataframes for plots without kappa and train time
df_barplot = df_summary.drop(columns = ["kappa", "train time"])

#create long format for grouped barplot
df_long = df_barplot.reset_index(names="model").melt(id_vars="model", var_name="metric", value_name="value")

#plot and save grouped barplot with models and metrics
sns.barplot(data=df_long, x="model", y="value", hue="metric", gap=.1)
plt.ylim([0,1.35])
plt.tight_layout()
plt.savefig(snakemake.output.plot, dpi= 150)
plt.close()

#plot and save heatmap with metrics annotated with values
plt.figure()
ax = sns.heatmap(
    df_summary.drop(columns=["train time"]),
    annot=True,          
    fmt=".3f",
    cmap="RdYlGn",
    vmin=df_summary.drop(columns=["train time"]).min().min() * 0.99,
    vmax=1.0   
)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.savefig(snakemake.output.heat, dpi = 150)
plt.close()