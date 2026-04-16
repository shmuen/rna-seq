import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

#import preprocessed data, parameter, mapping and scaler
X_scaled = pd.read_csv(snakemake.input.scaled)
y = pd.read_csv(snakemake.input.y_train)['Class']
n = snakemake.params.n_components

with open(snakemake.input.mapping) as f:
    label_mapping = json.load(f)

inverse_mapping = {v: k for k, v in label_mapping.items()}

#PCA on training data and save PCA
pca = PCA(n_components=n)
pca.fit(X_scaled.values)
components = pca.transform(X_scaled)
joblib.dump(pca, snakemake.output.pca)

#full PCA
pca_full = PCA()
pca_full.fit(X_scaled)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components_80 = np.argmax(cumsum >= 0.8) + 1
print(f'components for 80% variance: {n_components_80}')

#save compoments
df_components = pd.DataFrame(
    components,
    columns=[f'PC{i+1}' for i in range(n)]
)
# df_components['Class'] = y.values
df_components.to_csv(snakemake.output.components, index = False)

#plot PCA
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

#plot PCA 80% variance
plt.figure(figsize=(10,8))
plt.plot(cumsum)
plt.xlabel("amount components")
plt.ylabel("cumulutive variance")
plt.axhline(y=0.8, color ='r', linestyle = '--')
plt.title('Explained variance')
plt.savefig(snakemake.output.variance_plot, dpi= 150)


