import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(snakemake.input.data)

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=snakemake.params.test_size,
    stratify=y,
    random_state=snakemake.params.seed
)

X_train.to_csv(snakemake.output.X_train)
X_test.to_csv(snakemake.output.X_test)
y_train.to_csv(snakemake.output.y_train)
y_test.to_csv(snakemake.output.y_test)
