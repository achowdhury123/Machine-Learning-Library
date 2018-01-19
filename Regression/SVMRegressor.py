import numpy as np
import cvxopt

class SVMRegressor:

        def __init__(self, C = 0, sigma = 0):
            self.C = C
            self.sigma = sigma
            self.params = None
            self.intercept = None
            self.t = None
            self.slack_variables = None

        def get_params(self, X, y):
            y = np.array(y, ndmin = 2).T
            if self.C == 0:
                P = np.identity(X.shape[1])
                P[0, 0] = 0
                P = cvxopt.matrix(P)
                q = np.zeros(X.shape[1])
                q = cvxopt.matrix(q)
                G_top = -1 * X
                G_bottom = X
                G = np.vstack((G_top, G_bottom))
                G = cvxopt.matrix(G)
                h_top = np.ones(X.shape[0]) * (self.sigma - y)
                h_bottom = np.ones(X.shape[0]) * (self.sigma + y)
                h = np.hstack((h_top, h_bottom)).T
                h = cvxopt.matrix(h)
            else:
                P_topleft = np.identity(X.shape[1])
                P_topleft[0, 0] = 0
                P_topright = np.zeros((X.shape[1], X.shape[0] * 2))
                P_top = np.hstack((P_topleft, P_topright))
                P_bottom = np.zeros((X.shape[0] * 2, X.shape[0] * 2 + X.shape[1]))
                P = np.vstack((P_top, P_bottom))
                P = cvxopt.matrix(P)
                q = np.zeros(X.shape[1])
                q_new = np.ones(X.shape[0] * 2) * self.C
                q = np.hstack((q, q_new))
                q = cvxopt.matrix(q)
                G_topleft = X * -1
                G_nextright = np.identity(X.shape[0]) * -1
                G_topright = np.zeros((X.shape[0], X.shape[0]))
                G_top = np.hstack((G_topleft, G_nextright, G_topright))
                G_topnext = X
                G_nextrightnext = np.zeros((X.shape[0], X.shape[0]))
                G_toprightnext = np.identity(X.shape[0]) * -1
                G_next = np.hstack((G_topnext, G_nextrightnext, G_toprightnext))
                G_bottom_left = np.zeros((X.shape[0] * 2, X.shape[1]))
                G_bottom_right = np.identity(X.shape[0] * 2) * -1
                G_bottom = np.hstack((G_bottom_left, G_bottom_right))
                G = np.vstack((G_top, G_next, G_bottom))
                G = cvxopt.matrix(G)
                h = np.ones(X.shape[0]) * (self.sigma - y)
                h_other = np.ones(X.shape[0]) * (self.sigma + y)
                h_new = np.array(np.zeros(X.shape[0] * 2), ndmin = 2)
                h = np.hstack((h, h_other, h_new)).T
                h = cvxopt.matrix(h)
            params = cvxopt.solvers.qp(P, q, G, h)
            params = np.ravel(params['x'])
            return params

        def train(self, X, y):
            X = np.c_[np.ones((X.shape[0], 1)), X]
            self.params = self.get_params(X, y)
            self.intercept = self.params[0]
            self.slack_variables = self.params[X.shape[1]:]
            self.params = self.params[1:X.shape[1]]
            return self

        def predict(self, X):
            return np.dot(X, self.params) + self.intercept
