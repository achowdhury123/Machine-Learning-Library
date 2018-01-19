import numpy as np

class DecisionTreeRegressor:

    class Node:

        def __init__(self, average = None, feature = None, feature_value = None):
            self.average = average
            self.feature = feature
            self.feature_value = feature_value
            self.right = None
            self.left = None
            self.root = False
            self.leaf = False

    def __init__(self, max_depth = 0, min_samples_per_leaf = 0, random = False, random_features = None):
        self.max_depth = max_depth
        self.root = None
        self.min_samples_per_leaf = min_samples_per_leaf
        self.random = random
        self.random_features = random_features

    def get_average(self, y):
        if y.shape[0] == 0:
            return None
        average = 1/y.shape[0] * np.sum(y)
        return average


    def MSE(self, y):
        average = self.get_average(y)
        mse = np.sum((average - y)**2)
        return mse

    def CARTCost(self, X, y, counter = 0, node = None):

        if counter == 0:
            node = self.Node()
            node.root = True
        value = np.inf

        set_size = X.shape
        average = self.get_average(y)

        if counter == self.max_depth or X.shape[0] <= self.min_samples_per_leaf:
            node_temp = self.Node(average = average)
            node = node_temp
            node.leaf = True

            return node

        for k in self.random_features:
            feature_array = np.arange(min(X[:, k]), max(X[:, k]), (max(X[:, k]) - min(X[:, k])) / 100)
            for t_k in range(feature_array.shape[0]):
                sub_indices_left = X[:, k] > feature_array[t_k]
                sub_indices_right = X[:, k] <= feature_array[t_k]
                sub_group_left = np.c_[X[sub_indices_left], y[sub_indices_left]]
                sub_group_right = np.c_[X[sub_indices_right], y[sub_indices_right]]
                value_test = sub_group_left.shape[0] / X.shape[0] * self.MSE(y[sub_indices_left]) + sub_group_right.shape[0] / X.shape[0] * self.MSE(y[sub_indices_right])
                if value_test < value:
                    value = value_test
                    feature = k
                    decision_boundary = feature_array[t_k]
                    X_left = X[sub_indices_left]
                    X_right = X[sub_indices_right]
                    y_left = y[sub_indices_left]
                    y_right = y[sub_indices_right]

        node_temp = self.Node(average = average, feature = feature, feature_value = decision_boundary)
        node = node_temp
        node.left = self.CARTCost(X_left, y_left, counter = counter + 1)
        node.right = self.CARTCost(X_right, y_right, counter = counter + 1)

        if counter == 0:
            self.root = node
            return self

        return node

    def train(self, X, y, q = None):

        self.random_features = []

        if self.random == True:

            random_indices = np.random.permutation(X.shape[1])

            self.random_features = random_indices[:int(X.shape[0] * 0.8 + 1)]

        else:

            for i in range(X.shape[1]):

                self.random_features.append(i)

        self.CARTCost(X, y)
        if q != None:
            q.put(self)
        return self

    def predict_value(self, X, node = None):

        if node == None:
            node = self.root

        if node.leaf == True:
            y = node.average
            return y
        elif X[node.feature] >= node.feature_value:
            node = node.left
        else:
            node = node.right

        y = self.predict_value(X, node)

        return y

    def predict(self, X, node = None):

        y = np.zeros((X.shape[0]))
        counter = 0

        for x in X:
            y[counter] = self.predict_value(x)
            counter = counter + 1

        return y
