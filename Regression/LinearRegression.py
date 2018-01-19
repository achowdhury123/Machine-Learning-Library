import numpy as np
import math

class LinearRegression:
    def __init__(self, n_features, polynomial = False, degree = 0):
        self.n_features = n_features
        self.gradient = np.zeros((n_features, 1))
        self.polynomial = polynomial
        self.degree = degree
        if polynomial == True:
            self.params = np.random.randn(int((math.factorial(self.n_features + self.degree) / (math.factorial(self.n_features) * math.factorial(self.degree)))), int(1.0)) * 0.001
        else:
            self.params = np.random.randn(self.n_features+1, 1)

    def polyfit(self, X):
        for i in range(0, self.n_features):
            for j in range(0, self.n_features):
                if j >= i:
                    X = np.c_[X, X[:, i] * X[:, j]]
        return X

    def train(self, X, y, iterations, learning_rate, batch_size):
        if self.polynomial == True:
            X = self.polyfit(X)

        X = np.c_[np.ones((X.shape[0], 1)), X]

        for i in range(iterations):

            self.gradient = np.zeros((X.shape[1], 1))

            random_indices = np.random.permutation(X.shape[0])[:batch_size]

            X_subset = X[random_indices]

            y_subset = y[random_indices]

            for j in range(batch_size):

                self.gradient = self.gradient + 2 * (np.dot(X_subset[j], self.params) - y_subset[j]) * X_subset[j]

            self.params = self.params - learning_rate * self.gradient.T / batch_size

        return self

    def predict(self, X):

        if self.polynomial == True:
            X = self.polyfit(X)

        X = np.c_[np.ones((X.shape[0], 1)), X]

        prediction = np.dot(X, self.params)

        return prediction
