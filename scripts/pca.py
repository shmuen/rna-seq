import pandas as pd
import numpy as np
import joblib, json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#import preprocessed data, PCA parameters and label mapping
X_train_scaled = pd.read_csv(snakemake.input.X_train_scaled, index_col=0)
X_test_scaled = pd.read_csv(snakemake.input.X_test_scaled, index_col=0)
y = pd.read_csv(snakemake.input.y_train, index_col=0)['Class']
n = snakemake.params.n_components

#load mapping and create inv_map for PCA plot
with open(snakemake.input.label_mapping) as f:
    label_mapping = json.load(f)
inv_map = {int(k): v for k, v in label_mapping.items()}

#filter for genes with high variance to reduce noise and dimensionality
variance = X_train_scaled.var()
high_var_genes = variance[variance > variance.median()].index
X_train_scaled_filtered = X_train_scaled[high_var_genes]
X_test_scaled_filtered = X_test_scaled[high_var_genes]
with open(snakemake.log[0], 'w') as log:
    log.write(f'Filtering leaves {len(X_train_scaled_filtered.columns)} genes\n')

#fit PCA on training data only, save PCA and transform train and test data
pca = PCA(n_components=n)
X_train_pca = pca.fit_transform(X_train_scaled_filtered)
joblib.dump(pca, snakemake.output.pca)
X_test_pca = pca.transform(X_test_scaled_filtered)

#save compoments
components_cols = [f'PC{i+1}' for i in range(n)]
pd.DataFrame(X_train_pca, columns=components_cols, index=X_train_scaled.index).to_csv(snakemake.output.X_train_pca)
pd.DataFrame(X_test_pca, columns=components_cols, index=X_test_scaled.index).to_csv(snakemake.output.X_test_pca)

#fit full PCA (all compoments) to determine variance explained
pca_full = PCA()
pca_full.fit(X_train_scaled_filtered)

#get and log number of compoments to reach 80% variance
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components_80 = np.argmax(cumsum >= 0.8) + 1
with open(snakemake.log[0], 'a') as log:
    log.write(f'components for 80% variance: {n_components_80}\n')
    log.write(f'Variance explained by {n} components: {cumsum[n-1]:.3f}\n')

#visualize PCA and variance
plt.figure(figsize=(10,8))
for label in y.unique():
    mask = y == label
    plt.scatter(
        X_train_pca[mask, 0],
        X_train_pca[mask, 1],
        label = inv_map[int(label)],
        alpha = 0.6
    )

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA - Gene Expression Cancer')
plt.legend()
plt.tight_layout()
plt.savefig(snakemake.output.plot, dpi=150)
plt.close()

#plot PCA 80% variance
plt.figure(figsize=(10,8))
plt.plot(cumsum)
plt.xlabel("Number of components")
plt.ylabel("Cumulative variance")
plt.axhline(y=0.8, color ='r', linestyle = '--')
plt.title('Explained variance')
plt.tight_layout()
plt.savefig(snakemake.output.variance_plot, dpi= 150)
plt.close()


