import pandas as pd
from src.embedding import embedding
from src.preprocessing import preprocessing
df = pd.read_csv("data/knn_time_series_sample.csv")

predictor = ["lag_1_sales", "rolling_7_sales_avg", "promo_index"]

df_scaled = preprocessing("data/knn_time_series_sample.csv", predictor)
print(df_scaled)