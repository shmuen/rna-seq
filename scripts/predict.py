import pandas as pd
import joblib

#import PCA-transformed test data and trained model
X_test_pca = pd.read_csv(snakemake.input.X_test_pca, index_col=0)
#check if model contains label encoder and transform
loaded = joblib.load(snakemake.input.model)
if isinstance(loaded, tuple):
    model, le = loaded
    classes = le.classes_
    y_pred = le.inverse_transform(model.predict(X_test_pca))
else:
    model = loaded
    y_pred = model.predict(X_test_pca)
    classes = model.classes_

#calculate class prediction and probability for test data 
y_proba = model.predict_proba(X_test_pca)

#save prediction and probability
pd.DataFrame(y_pred, index=X_test_pca.index, columns=["y_pred"]).to_csv(snakemake.output.y_pred)
pd.DataFrame(y_proba,  index=X_test_pca.index, columns=classes).to_csv(snakemake.output.y_proba)