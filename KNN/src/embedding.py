import pandas as pd
import numpy as np
from preprocessing import preprocessing
def embedding(predictor, csv_path, embedding_length):
    df = pd.read_csv(csv_path)
    df_scaled = preprocessing(csv_path, predictor)
    #X[i]をembedding_length個の[,,]成分を持つベクトル
    #Y[i]を次の観測値
    dependent = "target_sales"
    X,Y = [],[]
    for i in range(len(df_scaled) - embedding_length):
        X.append(
            df_scaled[predictor].iloc[i:i + embedding_length ]
            )
        Y.append(
            df_scaled[dependent].iloc[i+embedding_length]
            )

    return X, Y



