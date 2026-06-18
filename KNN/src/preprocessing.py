from sklearn.preprocessing import StandardScaler
import pandas as pd
scaler = StandardScaler()

def preprocessing(csv_path, predictor):
    df = pd.read_csv(csv_path)

    df[predictor] = scaler.fit_transform(df[predictor])

    return df

