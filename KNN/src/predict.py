import argparse
import numpy as np
import pandas as pd

from preprocessing import preprocessing
from embedding import embedding
from model import KNNModel


def predict_knn(
    csv_path,
    predictor,
    embedding_length=7,
    k=5,
    weighted=False
):
    
    # 学習用の埋め込みデータを作成
    X, Y = embedding(
        predictor=predictor,
        csv_path=csv_path,
        embedding_length=embedding_length
    )

    # モデル作成
    model = KNNModel(X=X, y=Y, k=k)

    # CSVを標準化して、最後の embedding_length 個を予測用入力にする
    df_scaled = preprocessing(csv_path, predictor)

    x_pred = df_scaled[predictor].iloc[-embedding_length:]

    # 近傍点を取得
    nearest_distances, nearest_indices = model.select_points(x_pred)

    # 近傍点に対応するYを取り出す
    Y = np.array(Y)
    nearest_values = Y[nearest_indices]

    # 予測値を計算
    if weighted:
        distances = np.array(nearest_distances)

        # 距離0の場合に割り算エラーを防ぐ
        weights = 1 / (distances + 1e-8)
        pred = np.sum(weights * nearest_values) / np.sum(weights)
    else:
        pred = np.mean(nearest_values)

    return pred, nearest_indices, nearest_distances, nearest_values


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--predictor", type=str, required=True)
    parser.add_argument("--embedding_length", type=int, default=7)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--weighted", action="store_true")

    args = parser.parse_args()

    # カンマ区切りで複数列に対応
    predictor = [col.strip() for col in args.predictor.split(",")]

    pred, nearest_indices, nearest_distances, nearest_values = predict_knn(
        csv_path=args.csv_path,
        predictor=predictor,
        embedding_length=args.embedding_length,
        k=args.k,
        weighted=args.weighted
    )

    print("===== KNN Time Series Prediction =====")
    print(f"prediction: {pred}")
    print()
    print("nearest indices:")
    print(nearest_indices)
    print()
    print("nearest distances:")
    print(nearest_distances)
    print()
    print("nearest target values:")
    print(nearest_values)


if __name__ == "__main__":
    main()