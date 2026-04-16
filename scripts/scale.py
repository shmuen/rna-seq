import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

#import data
X_train = pd.read_csv(snakemake.input.data)

#scale data and save scaler
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, snakemake.output.scaler)

#transform X_train
X_scaled = scaler.transform(X_train)

#save scaled data
pd.DataFrame(X_scaled, columns=X_train.columns).to_csv(snakemake.output.scaled, index =False)

