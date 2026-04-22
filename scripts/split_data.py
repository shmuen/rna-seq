import pandas as pd
from sklearn.model_selection import train_test_split

#import data
df = pd.read_csv(snakemake.input.data, index_col=0)

#separate X and y
X = df.drop(columns=['Class'])
y = df['Class']

#split data into train and test datasets with config parameters
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=snakemake.params.test_size,
    stratify=y,
    random_state=snakemake.params.seed
)

#log split results
with open(snakemake.log[0], 'w') as log:
    log.write(f'X_train: {X_train.shape}, X_test: {X_test.shape}\n')
    log.write(f'y_train distribution:\n{y_train.value_counts().to_string()}\n')
    log.write(f'y_test distribution:\n{y_test.value_counts().to_string()}\n')

#save train and test data separately for X and y
X_train.to_csv(snakemake.output.X_train)
X_test.to_csv(snakemake.output.X_test)
y_train.to_csv(snakemake.output.y_train)
y_test.to_csv(snakemake.output.y_test)
