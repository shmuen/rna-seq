import pandas as pd
import numpy as np
import joblib, json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#import preprocessed data, parameter, mapping and scaler
X_train_scaled = pd.read_csv(snakemake.input.X_train_scaled)
X_test_scaled = pd.read_csv(snakemake.input.X_test_scaled)
y = pd.read_csv(snakemake.input.y_train)['Class']
n = snakemake.params.n_components

with open(snakemake.input.mapping) as f:
    label_mapping = json.load(f)

inverse_mapping = {v: k for k, v in label_mapping.items()}

#PCA on training data and save PCA
pca = PCA(n_components=n)
X_train_pca = pca.fit_transform(X_train_scaled)
joblib.dump(pca, snakemake.output.pca)

#transform scaled test data
X_test_pca = pca.transform(X_test_scaled)

#save compoments
components_cols = [f'PC{i+1}' for i in range(n)]
pd.DataFrame(X_train_pca, columns=components_cols).to_csv(snakemake.output.X_train_pca, index = False)
pd.DataFrame(X_test_pca, columns=components_cols).to_csv(snakemake.output.X_test_pca, index = False)

#full PCA
pca_full = PCA()
pca_full.fit(X_train_scaled)

#get amount of compoments to reach 80% variance
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components_80 = np.argmax(cumsum >= 0.8) + 1
print(f'components for 80% variance: {n_components_80}')

#visualize PCA and variance
plt.figure(figsize=(10,8))
for label in y.unique():
    mask = y == label
    plt.scatter(
        X_train_pca[mask, 0],
        X_train_pca[mask, 1],
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


