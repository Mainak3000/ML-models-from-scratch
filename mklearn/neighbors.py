import numpy as np
import pandas as pd
import statistics as stats
import sys
from collections import Counter
from exception import CustomException



class KNeighborsClassifier:
    def __init__(self, metric='euclidean', n_neighbors=5):  ## metric is type of distance metric (euclidean or manhattan)
        self.metric = metric
        self.n_neighbors = n_neighbors

    def get_distance_metric(self, p1, p2):

        p1 = np.array(p1)
        p2 = np.array(p2)

        distance = 0

        if self.metric == 'euclidean':
            distance = np.sqrt(np.sum((p1 - p2) ** 2))
        elif self.metric == 'manhattan':
            distance = np.sum(np.abs(p1 - p2))
        else:
            raise ValueError("Input metric name is incorrect")

        return distance

    def nearest_neighbors(self, test_data):

        distance_list = []

        for i, data in enumerate(self.X):
            distance = self.get_distance_metric(data, test_data)
            distance_list.append((i, distance))

        distance_list.sort(key = lambda x: x[1])

        neighbors_index = [distance_list[j][0] for j in range(self.n_neighbors)]

        return neighbors_index

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y).flatten()

    def predict(self, X):
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []

        for test_sample in X:
            neighbors_index = self.nearest_neighbors(test_sample)
            labels = [self.y[i] for i in neighbors_index]

            # Handle tie-breaking using Counter
            label_counts = Counter(labels)
            max_count = max(label_counts.values())
            mode_candidates = [label for label, count in label_counts.items() if count == max_count]
            prediction = min(mode_candidates)  # Tiebreak: choose the smallest label

            predictions.append(prediction)

        return predictions if len(predictions) > 1 else predictions[0]



