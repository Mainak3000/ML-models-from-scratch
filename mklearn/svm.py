import numpy as np
import pandas as pd
import sys
from exception import CustomException

## Equation of the hyperplane: y=wx-b

class SVC:
    def __init__(self, learning_rate, n_iterations, C=1):  #C is regularization parameter (lambda) 
        self.learing_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C

    def fit(self, X, y):
        try:
            # no of training examples & features
            # no of rows = no of training example || no of columns = no of features
            self.rows, self.columns = X.shape 

            self.w = np.zeros(self.columns)
            self.b = 0

            self.X = np.array(X)
            self.y = np.array(y).flatten()

            for i in range(self.n_iterations):
                self.update_weights()

        except Exception as e:
            raise CustomException(e, sys)

    def update_weights(self):

        try:
            #label encoding
            y_label = np.where(self.y<=0, -1, 1)

            # Gradients (dw, db)
            for i, x_i in enumerate(self.X):
                condition = y_label[i] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    dw = 2*self.C*self.w
                    db = 0
                else:
                    dw = 2*self.C*self.w - np.dot(y_label[i], x_i)
                    db = y_label[i]

                self.w = self.w - self.learing_rate*dw
                self.b = self.b - self.learing_rate*db

        except Exception as e:
            raise CustomException(e, sys)
        

    def predict(self, X):
        
        try:
            output = np.dot(X, self.w) - self.b
            y_hat = np.where(np.sign(output)<=-1, 0, 1)

            return y_hat

        except Exception as e:
            raise CustomException(e, sys)



class SVR:
    def __init__(self, learning_rate=0.001, n_iterations=1000, C=1.0, epsilon=0.1): #C is regularization parameter (lambda) 
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C 
        self.epsilon = epsilon

    def fit(self, X, y):
        try:
            self.X = np.array(X)
            self.y = np.array(y).flatten()
            self.rows, self.columns = self.X.shape

            self.w = np.zeros(self.columns)
            self.b = 0

            for _ in range(self.n_iterations):
                self.update_weights()

        except Exception as e:
            raise CustomException(e, sys)

    def update_weights(self):
        try:
            dw = np.zeros(self.columns)
            db = 0

            for i in range(self.rows):
                x_i = self.X[i]
                y_i = self.y[i]
                y_pred = np.dot(x_i, self.w) + self.b
                error = y_pred - y_i

                if error > self.epsilon:
                    dw = dw + self.C * x_i
                    db = db + self.C
                elif error < -self.epsilon:
                    dw = dw - self.C * x_i
                    db = db - self.C
                # else: within epsilon tube; no penalty

            dw = dw + self.w  # Add derivative of regularization term
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X):
        try:
            X = np.array(X)
            return np.dot(X, self.w) + self.b
        except Exception as e:
            raise CustomException(e, sys)