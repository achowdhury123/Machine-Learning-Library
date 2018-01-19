import numpy as np
import math
from sklearn import preprocessing

class LogisticClassifier:
    def __init__(self, n_features, multi_class = False, classes=2):
        self.n_features = n_features + 1
        self.classes = classes
        self.multi_class = multi_class

        if multi_class == False:
            self.params = np.random.randn(self.n_features, 1)*0.001
            self.gradient = np.zeros((self.n_features, 1))

        else:
            self.params = np.random.randn(self.classes, self.n_features)*0.001
            self.gradient = np.zeros((self.classes, self.n_features))

    def logit(self, X):
        X = X * -1
        X_logistic = 1/(1+np.exp(X))
        return X_logistic

    def softmax(self, X):

        if X.shape[0] == 1:
            return np.exp(np.dot(self.params, X.T))/np.sum(np.exp(np.dot(self.params, X.T)))

        softmax_score = np.dot(self.params, X.T)
        probability = np.zeros((self.classes, 1))
        softmax_score = np.array(softmax_score, ndmin = 2)
        for i in range(softmax_score.shape[1]):
            instance = np.array((np.exp(softmax_score[:, i])/np.sum(np.exp(softmax_score[:, i]))), ndmin = 2).T
            probability = np.c_[probability, instance]
        return probability[:, 1:].T

    def train(self, X, y, iterations, learning_rate, batch_size):

        X = np.c_[np.ones((X.shape[0], 1)), X]

        if self.multi_class == False:

            counter = 0

            for i in range(iterations):

                self.gradient = np.zeros((self.n_features, 1)).T

                random_indices = np.random.permutation(X.shape[0])[:batch_size]

                X_subset = X[random_indices]

                y_subset = y[random_indices]

                for j in range(batch_size):

                    self.gradient = self.gradient +  (self.logit(np.dot(X_subset[j], self.params))-y_subset[j]) * X_subset[j]

                self.params = self.params - learning_rate * self.gradient.T / batch_size

            return self

        else:

            labelbinarizer = preprocessing.LabelBinarizer()
            y_encoded = labelbinarizer.fit_transform(y)

            for i in range(iterations):

                self.gradient = np.zeros((self.classes, self.n_features))

                random_indices = np.random.permutation(X.shape[0])[:batch_size]

                X_subset = X[random_indices]

                y_subset = y_encoded[random_indices]

                for j in range(batch_size):

                    x = np.array(X_subset[j], ndmin = 2)

                    y = np.array(y_subset[j], ndmin = 2).T

                    self.gradient = self.gradient + (self.softmax(x)-y) * x

                self.params = self.params - learning_rate * self.gradient / batch_size

            return self


    def predict(self, X):

        X = np.c_[np.ones((X.shape[0], 1)), X]

        if self.multi_class == False:

            probability = self.logit(np.dot(X, self.params))

            if probability < 0.5:
                y = 0
            else:
                y = 1

            return y

        else:

            probability = self.softmax(X)
            correct_class = np.zeros((X.shape[0], 1))
            for i in range(X.shape[0]):
                correct_class[i] = np.where(probability[i] == np.amax(probability[i]))[0]

            return correct_class

    def predict_proba(self, X):

        X = np.c_[np.ones((X.shape[0], 1)), X]

        if self.multi_class == False:

            probability = self.logit(np.dot(X, self.params))

            return probability

        else:

            probability = self.softmax(X)

            return probability
