import numpy as np
import cvxopt

class SVM:
    def __init__(self, C=0):
        self.C = C
        self.params = None
        self.intercept = None
        self.t = None
        self.slack_variables = None

    def get_params(self, X, y):
        for i in range(y.shape[0]):
            if y[i] == 1:
                self.t[i] = 1
            else:
                self.t[i] = -1

        self.t = np.array(self.t, ndmin = 2).T
        if self.C == 0:
            P = cvxopt.matrix(np.identity(X.shape[1]))
            P[0, 0] = 0
            q = cvxopt.matrix(np.zeros(X.shape[1]))
            G = cvxopt.matrix(-1 * self.t * X)
            h = cvxopt.matrix(np.ones(X.shape[0]) * -1)
        else:
            P_topleft = np.identity(X.shape[1])
            P_topleft[0, 0] = 0
            P_topright = np.zeros((X.shape[1], X.shape[0]))
            P_top = np.hstack((P_topleft, P_topright))
            P_bottom = np.zeros((X.shape[0], X.shape[0] + X.shape[1]))
            P = np.vstack((P_top, P_bottom))
            P = cvxopt.matrix(P)
            q = np.zeros(X.shape[1])
            q_new = np.ones(X.shape[0]) * self.C
            q = np.hstack((q, q_new))
            q = cvxopt.matrix(q)
            G_topleft = -1 * self.t * X
            G_topright = np.identity(X.shape[0]) * -1
            G_top = np.hstack((G_topleft, G_topright))
            G_bottom_left = np.zeros((X.shape))
            G_bottom_right = np.identity(X.shape[0]) * -1
            G_bottom = np.hstack((G_bottom_left, G_bottom_right))
            G = np.vstack((G_top, G_bottom))
            G = cvxopt.matrix(G)
            h = np.ones(X.shape[0]) * -1
            h_new = np.zeros(X.shape[0])
            h = np.hstack((h, h_new))
            h = cvxopt.matrix(h)
        params = cvxopt.solvers.qp(P, q, G, h)
        params = np.ravel(params['x'])
        return params

    def train(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.t = np.zeros(y.shape[0])
        self.params = self.get_params(X, y)
        self.intercept = self.params[0]
        self.slack_variables = self.params[3:]
        self.params = self.params[1:3]
        return self

    def predict(self, X):
        class_values = np.sign(np.dot(X, self.params) + self.intercept)
        zero_indices = class_values == -1
        class_values[zero_indices] = 0
        return class_values
