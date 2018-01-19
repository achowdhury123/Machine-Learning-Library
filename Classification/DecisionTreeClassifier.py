import numpy as np
from multiprocessing import Queue

class DecisionTreeClassifier:

    class Node:

        def __init__(self, feature=None, feature_value=None, class_value=None):
            self.feature = feature
            self.feature_value = feature_value
            self.class_value = class_value
            self.proba = None
            self.right = None
            self.left = None
            self.root = False
            self.leaf = False

    def __init__(self, max_depth = 0, min_samples_per_leaf = 0, random = False, random_features = None):
        self.max_depth = max_depth
        self.min_samples_per_leaf = min_samples_per_leaf
        self.root = None
        self.random = random
        self.random_features = random_features
        self.classes = None

    def get_class_value(self, y):

        max_array = np.bincount(y)

        max_value = np.argwhere(max_array == max(max_array))[0]

        return max_value

    def get_probabilities(self, y):
        probabilities = np.zeros(self.classes.shape)
        for i in range(self.classes.shape[0]):
            for j in range(y.shape[0]):
                if y[j] == self.classes[i]:
                    probabilities[i] = probabilities[i] + 1
        probabilities = probabilities / y.shape
        return probabilities

    def GINI(self, X, y):
        classes = np.unique(y)
        GINI_score = 0
        for i in range(classes.shape[0]):
            class_indices = y == classes[i]
            GINI_class_score = (np.bincount(class_indices)[1] / X.shape[0]) ** 2
            GINI_score = GINI_score + GINI_class_score
        return 1 - GINI_score

    def Entropy(self, X, y):
        classes = np.unique(y)
        Entropy_score = 0
        for i in range(classes.shape[0]):
            class_indices = y == classes[i]
            Entropy_class_score = (np.bincount(class_indices)[1] / X.shape[0]) * np.log(np.bincount(class_indices)[1] / X.shape[0])
            Entropy_score = Entropy_score + Entropy_class_score
        return Entropy_score * -1

    def CARTCost(self, X, y, counter = 0, node = None):
        if counter == 0:
            node = self.Node()
            node.root = True
        value = np.inf
        class_value = self.get_class_value(y)
        score = self.GINI(X, y)
        set_size = X.shape
        if self.GINI(X, y) == 0 or counter == self.max_depth or X.shape[0] <= self.min_samples_per_leaf:
            node_temp = self.Node(class_value = class_value)
            node = node_temp
            node.proba = self.get_probabilities(y)
            node.leaf = True
            return node
        for k in self.random_features:
            feature_array = np.arange(min(X[:, k]), max(X[:, k]), (max(X[:, k]) - min(X[:, k])) / 100)

            for t_k in range(feature_array.shape[0]):
                sub_indices_left = X[:, k] > feature_array[t_k]
                sub_indices_right = X[:, k] <= feature_array[t_k]
                sub_group_left = np.c_[X[sub_indices_left], y[sub_indices_left]]
                sub_group_right = np.c_[X[sub_indices_right], y[sub_indices_right]]
                value_test = sub_group_left.shape[0] / X.shape[0] * self.GINI(X[sub_indices_left], y[sub_indices_left]) + sub_group_right.shape[0] / X.shape[0] * self.GINI(X[sub_indices_right], y[sub_indices_right])
                if value_test < value:
                    value = value_test
                    feature = k
                    decision_boundary = feature_array[t_k]
                    X_left = X[sub_indices_left]
                    X_right = X[sub_indices_right]
                    y_left = y[sub_indices_left]
                    y_right = y[sub_indices_right]

        node_temp = self.Node(feature = feature, feature_value = decision_boundary)
        node = node_temp
        node.left = self.CARTCost(X_left, y_left, counter = counter + 1)
        node.right = self.CARTCost(X_right, y_right, counter = counter + 1)

        if counter == 0:
            self.root = node
            return self

        return node

    def train(self, X, y, q = None):

        self.random_features = []

        self.classes = np.unique(y)

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

    def predict_proba_value(self, X, node = None):
        if node == None:
            node = self.root

        if node.leaf == True:
            proba = node.proba
            return proba
        elif X[node.feature] >= node.feature_value:
            node = node.left
        else:
            node = node.right

        proba = self.predict_proba_value(X, node)

        return proba

    def predict_proba(self, X):

        proba = np.zeros((X.shape[0], self.classes.shape[0]))
        counter = 0

        for x in X:
            proba[counter] = self.predict_proba_value(x)
            counter = counter + 1

        return proba

    def predict_value(self, X, node = None):

        if node == None:
            node = self.root

        if node.leaf == True:
            y = node.class_value
            return y
        elif X[node.feature] >= node.feature_value:
            node = node.left
        else:
            node = node.right

        y = self.predict_value(X, node)

        return y

    def predict(self, X):

        y = np.zeros((X.shape[0]))
        counter = 0

        for x in X:
            y[counter] = self.predict_value(x)
            counter = counter + 1

        return y
