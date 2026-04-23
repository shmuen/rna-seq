import pandas as pd
import joblib

#import PCA-transformed test data and trained model
X_test_pca = pd.read_csv(snakemake.input.X_test_pca, index_col=0)
model = joblib.load(snakemake.input.model)

#calculate class prediction and probability for test data 
y_pred = model.predict(X_test_pca)
y_proba = model.predict_proba(X_test_pca)

#save prediction and probability
pd.DataFrame(y_pred, index=X_test_pca.index, columns=["y_pred"]).to_csv(snakemake.output.y_pred)
pd.DataFrame(y_proba,  index=X_test_pca.index, columns=model.classes_).to_csv(snakemake.output.y_proba)