import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

#import train data
X_train = pd.read_csv(snakemake.input.X_train)
X_test = pd.read_csv(snakemake.input.X_test)

#fit scaler on train data and save scaler 
scaler = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
joblib.dump(scaler, snakemake.output.scaler)

#transform test data
X_test_scaled = scaler.transform(X_test)

#save scaled data
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(snakemake.output.X_train_scaled, index =False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(snakemake.output.X_test_scaled, index =False)

