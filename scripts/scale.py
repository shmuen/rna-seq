import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

#import train data
X_train = pd.read_csv(snakemake.input.X_train, index_col=0)
X_test = pd.read_csv(snakemake.input.X_test, index_col=0)

#fit scaler on train data and save scaler 
scaler = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
joblib.dump(scaler, snakemake.output.scaler)

#transform test data
X_test_scaled = scaler.transform(X_test)

#write mean and std of scaler for first 5 entries to log
with open(snakemake.log[0], 'w') as log:
    log.write(f'Scaler mean: {scaler.mean_[:5]}\n')
    log.write(f'Scaler std: {scaler.scale_[:5]}\n')

#save scaled data
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(snakemake.output.X_train_scaled)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(snakemake.output.X_test_scaled)

