import numpy as np
import heapq


class KNNModel:
    def __init__(self, X, y, k=5):
        self.X = np.array(X)
        self.y = np.array(y)
        self.k = k

    def distance(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.linalg.norm(x1 - x2)

    def select_points(self, x):
        distances = []

        for i in range(len(self.X)):
            dist = self.distance(x, self.X[i])
            distances.append((dist, i))

        nearest = heapq.nsmallest(self.k, distances)

        nearest_distances = [dist for dist, i in nearest]
        nearest_indices = [i for dist, i in nearest]

        return nearest_distances, nearest_indices