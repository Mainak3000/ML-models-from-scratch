import numpy as np
import pandas as pd
import sys
from exception import CustomException

class LinearRegression:
    def __init__(self, learning_rate, n_iterations, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept


    def fit(self, X, y):
        try:
            # no of training examples & features
            # no of rows = no of training example || no of columns = no of features
            self.rows, self.columns = X.shape 

            # initiating weight & bias
            self.w = np.zeros(self.columns)
            if self.fit_intercept:
                self.c = 0

            self.X = np.array(X)
            self.y = np.array(y)

            #implementing gradient descent
            for i in range(self.n_iterations):
                self.update_weights()


        except Exception as e:
            raise CustomException(e, sys)


    def update_weights(self):
        try:

            y_pred = self.predict(self.X)
            
            # calculating gradient descent
            dw = - (2 * (self.X.T).dot(self.y - y_pred)) / self.rows
            if self.fit_intercept:
                dc = - (2 * np.sum(self.y - y_pred)) / self.rows

            #updating weights
            self.w = self.w - self.learning_rate*dw
            if self.fit_intercept:
                self.c = self.c - self.learning_rate*dc

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X):
        try:
            if self.fit_intercept:
                return X.dot(self.w) + self.c
            else:
                return X.dot(self.w)

        except Exception as e:
            raise CustomException(e, sys)


class LogisticRegression:
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        try:
            # no of training examples & features
            # no of rows = no of training example || no of columns = no of features
            self.rows, self.columns = X.shape 

            # initiating weight & bias
            self.w = np.zeros(self.columns)
            self.b = 0

            self.X = np.array(X)
            self.y = np.array(y).flatten()

            #implementing gradient descent
            for i in range(self.n_iterations):
                self.update_weights()


        except Exception as e:
            raise CustomException(e, sys)

    def update_weights(self):
        
        try:
            y_pred = self.predict_proba(self.X)

            dw = (1/self.rows) * (self.X.T).dot(y_pred - self.y)
            db = (1/self.rows) * np.sum(y_pred - self.y)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X):
        try:
            y_pred = self.predict_proba(X)
            y_pred = np.where(y_pred > 0.5, 1, 0)

            return y_pred
        
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, X):
        try:
            # segmoid function
            X = np.array(X)
            y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
            return y_pred
        except Exception as e:
            raise CustomException(e, sys) 

