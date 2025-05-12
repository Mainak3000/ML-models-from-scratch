import numpy as np
import sys
from collections import Counter
from exception import CustomException

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=100, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        try:
            self.X = np.array(X)
            self.y = np.array(y).flatten()
            self.max_features = self.X.shape[1] if self.max_features is None else min(self.X.shape[1], self.max_features)
            indices = np.arange(self.X.shape[0])
            self.root = self._grow_tree(indices)
        except Exception as e:
            raise CustomException(e, sys)

    def _grow_tree(self, indices, depth=0):
        X = self.X[indices]
        y = self.y[indices]
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split):
            return Node(value=self._most_common_label(y))

        feat_idxs = np.random.choice(n_features, self.max_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            return Node(value=self._most_common_label(y))

        # Split using boolean masks, then map back to global indices
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        if len(left_indices) == 0 or len(right_indices) == 0:
            return Node(value=self._most_common_label(y))

        left = self._grow_tree(left_indices, depth + 1)
        right = self._grow_tree(right_indices, depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs = np.where(X_column <= threshold)[0]
        right_idxs = np.where(X_column > threshold)[0]

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l = self._entropy(y[left_idxs])
        e_r = self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    