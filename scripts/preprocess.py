import pandas as pd
import numpy as np

#import of data
rna_seq = pd.read_csv(snakemake.input.data, index_col = 0)
labels = pd.read_csv(snakemake.input.labels, index_col = 0)
df = rna_seq.join(labels)

#first inspection of data, write to log
with open(snakemake.log[0], 'w') as log:
    log.write(f'Shape: {df.shape}\n')
    #check for missing values
    log.write(f'Missing: {df.isnull().sum().sum()}\n')
    #check for duplicates
    log.write(f'Duplicates: {df.duplicated().sum()}\n')

#log1p-transformation to reduce skewness of RNA-seq data
X = df.drop(columns=['Class'])
X_log = np.log1p(X)

#filter for genes with high variance to reduce noise and dimensionality
variance = X_log.var()
X_filtered = X_log[variance[variance > variance.median()].index]
with open(snakemake.log[0], 'a') as log:
    log.write(f'Filtering leaves {len(X_filtered.columns)} genes\n')

#combine X and y for output and save to file
X_filtered = pd.concat([X_filtered, labels['Class']], axis=1)  #pd.Series(y, name='Class', index=X_filtered.index)], axis=1)
X_filtered.to_csv(snakemake.output.data)

