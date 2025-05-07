import numpy as np
import pandas as pd
import sys
from exception import CustomException


def train_test_split(X, y, test_size=0.33, random_state=42):

    try:

        X = np.array(X)
        y = np.array(y)

        np.random.seed(random_state)

        indices = np.arange(len(X))
        np.random.shuffle(indices)

        test_count = int(len(X)*test_size)

        test_indices = indices[ : test_count]
        train_indices = indices[ test_count : ]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        raise CustomException(e, sys)