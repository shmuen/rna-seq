import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#import of data
rna_seq = pd.read_csv(snakemake.input.data)
labels = pd.read_csv(snakemake.input.labels)

df = pd.merge(rna_seq, labels, on="Unnamed: 0")

#first inspection of data
print(f'Shape: {df.shape}')
# print(f'Label: {labels.columns.tolist()}')

#check for missing values
print(f'Missing: {df.isnull().sum().sum()}')

#check for duplicates
print(f'Duplicates: {df.duplicated().sum()}')

#encoding labels
le = LabelEncoder()
y = le.fit_transform(labels["Class"])


#save encoding
mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
with open(snakemake.output.label_mapping, "w") as f:
    json.dump(mapping, f)

#log1p-transformation
X = df.drop(columns=['Class', 'Unnamed: 0'])
x_log = np.log1p(X)

#combine X and y for output
out = x_log.copy()
out['Class'] = y

#save output
output_path = snakemake.output.data
out.to_csv(output_path)

