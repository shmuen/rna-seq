import pandas as pd
import joblib

#import test data
X_test_pca = pd.read_csv(snakemake.input.X_test_pca)
model = joblib.load(snakemake.input.model)

#calculate prediction and probability for test data 
y_pred = model.predict(X_test_pca)
y_proba = model.predict_proba(X_test_pca)

#save predictoin and probability
pd.DataFrame(y_pred).to_csv(snakemake.output.y_pred, index =False)
pd.DataFrame(y_proba).to_csv(snakemake.output.y_proba, index =False)


