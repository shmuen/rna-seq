import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

#import preprocessed data, parameter and mapping
df = pd.read_csv(snakemake.input.data)
X = df.drop(columns=['Class'])
y = df['Class']
n = snakemake.params.n_components

with open(snakemake.input.mapping) as f:
    label_mapping = json.load(f)

inverse_mapping = {v: k for k, v in label_mapping.items()}

#scaling of data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA
pca = PCA(n_components=n)
components = pca.fit_transform(X_scaled)

#save compoments
df_components = pd.DataFrame(
    components,
    columns=[f'PC{i+1}' for i in range(n)]
)
df_components['Class'] = y.values
df_components.to_csv(snakemake.output.components, index = False)

#plot
plt.figure(figsize=(10,8))
for label in y.unique():
    mask = y == label
    plt.scatter(
        components[mask, 0],
        components[mask, 1],
        label = inverse_mapping[label],
        alpha = 0.6
    )

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA - Gene Expression Cancer')
plt.legend()
plt.tight_layout()
plt.savefig(snakemake.output.plot, dpi=150)


