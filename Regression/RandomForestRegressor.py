import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
from DecisionTreeRegressor import DecisionTreeRegressor
import matplotlib.pyplot as plt

class RandomForestRegressor:

    def __init__(self, decision_trees = 0, max_depth = 0, min_samples_per_leaf = 0):
        self.max_depth = max_depth
        self.min_samples_per_leaf = min_samples_per_leaf
        self.decision_trees = decision_trees
        self.random_forest = []
        for i in range(self.decision_trees):
            tree = DecisionTreeRegressor(self.max_depth, self.min_samples_per_leaf, random = True)
            self.random_forest.append(tree)

    def train(self, X, y):

        q = Queue()

        processes = []

        for tree in self.random_forest:

            random_indices = np.random.permutation(X.shape[0])

            random_indices = random_indices[:int(X.shape[0] * 0.8)]

            X_random = X[random_indices]

            y_random = y[random_indices]

            P = Process(target = tree.fit, args = (X_random, y_random, q))
            P.start()
            processes.append(P)

        for P in processes:
            P.join()

        self.random_forest = []

        for i in range(self.decision_trees):
            self.random_forest.append(q.get())

        return self

    def predict(self, X):

        y = np.zeros(X.shape[0])

        counter = 0

        for x in X:

            predictions = np.zeros(self.decision_trees)

            prediction_counter = 0

            for tree in self.random_forest:
                predictions[prediction_counter] = tree.predict_value(x)
                prediction_counter = prediction_counter + 1

            y[counter] = np.average(predictions)
            counter = counter + 1

        return y
