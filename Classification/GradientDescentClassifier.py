import numpy as np
import matplotlib.pyplot as pl

class GradientDescentClassifier:
    def __init__(self, C=0):
        self.C = C
        self.params = None
        self.intercept = None
        self.t = None
        self.gradient = None

    def train(self, X, y, iterations, learning_rate, batch_size):

        X = np.c_[np.ones((X.shape[0], 1)), X]

        self.params = np.random.randn(X.shape[1], 1)

        self.t = np.zeros(y.shape[0])

        for i in range(y.shape[0]):
            if y[i] == 1:
                self.t[i] = 1
            else:
                self.t[i] = -1

        for i in range(iterations):

            random_indices = np.random.permutation(X.shape[0])[:batch_size]

            X_subset = X[random_indices]

            y_subset = y[random_indices]

            t_subset = self.t[random_indices]

            for j in range(batch_size):

                var = t_subset[j] * (np.dot(np.array(X_subset[j], ndmin = 2), self.params))
                if var > 1:
                    hinge_loss = 0
                elif var == 1:
                    hinge_loss = 0
                else:
                    hinge_loss = np.array(-t_subset[j] * X_subset[j], ndmin = 2).T
                self.gradient = 0.1 * self.params + self.C * hinge_loss
                self.params = self.params - (1/(0.1*(i + 1))) * self.gradient
                self.gradient = 0

        self.intercept = self.params[0]
        self.params = self.params[1:]

        return self

    def predict(self, X):
        class_values = np.sign(np.dot(X, self.params) + self.intercept)
        zero_indices = class_values == -1
        class_values[zero_indices] = 0
        return class_values
