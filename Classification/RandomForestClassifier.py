import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
from DecisionTreeClassifier import DecisionTreeClassifier
import scipy.stats

class RandomForestClassifier:

    def __init__(self, decision_trees = 0, max_depth = 0, min_samples_per_leaf = 0):
        self.max_depth = max_depth
        self.min_samples_per_leaf = min_samples_per_leaf
        self.decision_trees = decision_trees
        self.random_forest = []
        for i in range(self.decision_trees):
            tree = DecisionTreeClassifier(self.max_depth, self.min_samples_per_leaf, random = True)
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

    def predict_proba(self, X):

        proba = np.zeros((X.shape[0], self.random_forest[0].classes.shape[0]))

        counter = 0

        for x in X:

            predictions = np.zeros(self.random_forest[0].classes.shape[0])

            for tree in self.random_forest:
                predictions = predictions + tree.predict_proba_value(x)

            proba[counter] = predictions

            counter = counter + 1

        proba = proba / self.decision_trees

        return proba



    def predict(self, X):

        y = np.zeros(X.shape[0])

        counter = 0

        for x in X:

            prediction_counter = 0

            predictions = np.zeros(self.decision_trees)

            for tree in self.random_forest:
                predictions[prediction_counter] = tree.predict_value(x)
                prediction_counter = prediction_counter + 1

            mode = scipy.stats.mode(predictions)
            y[counter] = mode[0]
            counter = counter + 1

        return y
