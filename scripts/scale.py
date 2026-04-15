import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

#import data
X = pd.read_csv(snakemake.input.data)

#scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#save scaled data
pd.DataFrame(X_scaled, columns=X.columns).to_csv(snakemake.output.scaled, index =False)

#save scaler
joblib.dump(X_scaled, snakemake.output.scaler)