# KNN による時系列予測

ラグ特徴と移動平均を窓に埋め込み、k 近傍法で次の値を予測します。

## ディレクトリ

コードは `KNN/` にあります。

| パス | 内容 |
| --- | --- |
| `KNN/src/preprocessing.py` | 特徴量の標準化 |
| `KNN/src/embedding.py` | 時系列の窓埋め込み |
| `KNN/src/model.py` | ユークリッド距離の k 近傍 |
| `KNN/src/predict.py` | 予測の入口 |
| `KNN/try.py` | 前処理結果の確認 |
| `KNN/config.toml` | `k`、埋め込み長、使う列 |
| `KNN/data/knn_time_series_sample.csv` | サンプルデータ |

## 実行

```bash
pip install numpy pandas
cd KNN/src
python predict.py \
  --csv_path ../data/knn_time_series_sample.csv \
  --predictor "lag_1_sales,rolling_7_sales_avg,promo_index" \
  --embedding_length 4 \
  --k 5
```

距離の逆数で重み付けする場合は `--weighted` を付けます。

前処理だけ確認する場合:

```bash
cd KNN
python try.py
```
