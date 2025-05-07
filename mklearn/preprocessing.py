import numpy as np
import pandas as pd
import sys
from exception import CustomException

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.
        """
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1

    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        """
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        self.fit(X)
        return self.transform(X)
