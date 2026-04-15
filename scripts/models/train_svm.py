import joblib
import pandas as pd
from sklearn.svm import SVC

X_train = pd.read_csv(snakemake.input.X_train)
y_train = pd.read_csv(snakemake.input.y_train)['Class']

svm = SVC(kernel="rbf", probability=True, random_state = snakemake.params.seed)
svm.fit(X_train, y_train)
joblib.dump(svm, snakemake.output.model)