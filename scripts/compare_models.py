import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import reports of all models
mod = []
acc, macro, mcc, kappa = [], [], [], []
train_time = []

for model in snakemake.input.reports:
    tmp = pd.read_csv(model)
    mod.append(model.split("/")[-1].split("_")[0])
    acc.append(tmp[tmp["Unnamed: 0"]=="accuracy"]["f1-score"].item())
    macro.append(tmp[tmp["Unnamed: 0"]=="macro avg"]["f1-score"].item())
    mcc.append(tmp[tmp["Unnamed: 0"]=="mcc"]["f1-score"].item())
    kappa.append(tmp[tmp["Unnamed: 0"]=="kappa"]["f1-score"].item())

for model in snakemake.input.bench:
    train_time.append(pd.read_csv(model,sep="\t")["s"].item())

#summary
df_summary = pd.DataFrame({
    "model": mod,
    "accuracy": acc,
    "macro": macro,
    "mcc": mcc,
    "kappa": kappa,
    "train time": train_time
})

#save summary
df_summary.round(3).to_csv(snakemake.output.summary, index=False)

#dataframes for plots
df_barplot = pd.DataFrame({
    "model": mod,
    "accuracy": acc,
    "macro": macro,
    "mcc": mcc,
})

df_heatmap = pd.DataFrame({
    "accuracy": acc,
    "macro": macro,
    "mcc": mcc,
    "kappa": kappa
}).transpose()

df_heatmap.columns = mod

df_long = df_barplot.melt(id_vars="model", var_name="metric", value_name="value")

sns.barplot(data=df_long, x="model", y="value", hue="metric", gap=.1)
plt.ylim([0,1.35])
plt.savefig(snakemake.output.plot, dpi= 150)

#heatmap with metrics
plt.figure()
ax = sns.heatmap(
    df_heatmap,          
    annot=True,          
    fmt=".3f",
    cmap="RdYlGn",
    vmin=0.9, vmax=1.0   
)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

plt.savefig(snakemake.output.heat, dpi = 150)


# a=["results/metrics/randomforest_report.csv",
# "results/metrics/svm_report.csv",
# "results/metrics/logistic_regression_report.csv",
# "results/metrics/xgboost_report.csv"]