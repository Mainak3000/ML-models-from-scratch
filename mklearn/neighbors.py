import numpy as np
import sys
from collections import Counter
from exception import CustomException

class BaseKNN:
    def __init__(self, metric='euclidean', n_neighbors=5):
        self.metric = metric
        self.n_neighbors = n_neighbors

    def get_distance_metric(self, p1, p2):
        try:
            p1 = np.array(p1)
            p2 = np.array(p2)

            if self.metric == 'euclidean':
                return np.sqrt(np.sum((p1 - p2) ** 2))
            elif self.metric == 'manhattan':
                return np.sum(np.abs(p1 - p2))
            else:
                raise ValueError("Input metric name is incorrect")
        except Exception as e:
            raise CustomException(e, sys)

    def nearest_neighbors(self, test_data):
        try:
            distance_list = []

            for i, data in enumerate(self.X):
                distance = self.get_distance_metric(data, test_data)
                distance_list.append((i, distance))

            distance_list.sort(key=lambda x: x[1])
            neighbors_index = [distance_list[j][0] for j in range(self.n_neighbors)]
            return neighbors_index
        except Exception as e:
            raise CustomException(e, sys)

    def fit(self, X, y):
        try:
            self.X = np.array(X)
            self.y = np.array(y).flatten()
        except Exception as e:
            raise CustomException(e, sys)


class KNeighborsClassifier(BaseKNN):
    def predict(self, X):
        try:
            X = np.array(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            predictions = []

            for test_sample in X:
                neighbors_index = self.nearest_neighbors(test_sample)
                labels = [self.y[i] for i in neighbors_index]

                label_counts = Counter(labels)
                max_count = max(label_counts.values())
                mode_candidates = [label for label, count in label_counts.items() if count == max_count]
                prediction = min(mode_candidates)  # Tie-breaker
                predictions.append(prediction)

            return predictions if len(predictions) > 1 else predictions[0]
        except Exception as e:
            raise CustomException(e, sys)


class KNeighborsRegressor(BaseKNN):
    def predict(self, X):
        try:
            X = np.array(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            predictions = []

            for test_sample in X:
                neighbors_index = self.nearest_neighbors(test_sample)
                neighbor_values = [self.y[i] for i in neighbors_index]
                prediction = np.mean(neighbor_values)
                predictions.append(prediction)

            return predictions if len(predictions) > 1 else predictions[0]
        except Exception as e:
            raise CustomException(e, sys)