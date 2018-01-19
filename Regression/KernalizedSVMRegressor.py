import numpy as np
import cvxopt

class KernalizedSVMRegressor:

    def linear_kernel(self, x1, x2, degree=0):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x, y, degree):
        return (np.dot(x, y) + 1) ** degree

    def __init__(self, C=0, sigma = 0, kernel = "linear", degree=0):
        self.params = None
        self.intercept = None
        self.kernel_matrix = None
        self.t = None
        self.lagrangevalues = None
        self.supportvectors = None
        self.supportvectorclasses = None
        self.C = C
        self.sigma = 0
        if kernel == "linear":
            self.kernel = self.linear_kernel
        else:
            self.kernel = self.polynomial_kernel
        self.degree = degree

    def get_params(self, X, y):
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.kernel_matrix[i, j] = self.kernel(X[i, :], X[j, :], self.degree)
        P_topleft = self.kernel_matrix
        P_topright = self.kernel_matrix * -1
        P_top = np.hstack((P_topleft, P_topright))
        P_bottomleft = self.kernel_matrix * -1
        P_bottomright = self.kernel_matrix
        P_bottom = np.hstack((P_bottomleft, P_bottomright))
        P = np.vstack((P_top, P_bottom))
        P = cvxopt.matrix(P)
        q_upper = (self.sigma - y).T
        q_lower = (self.sigma + y).T
        q = np.hstack((q_upper, q_lower)).T
        q = cvxopt.matrix(q)
        A_upper = np.ones(X.shape[0])
        A_lower = np.ones(X.shape[0]) * -1
        A = np.array(np.hstack((A_upper, A_lower)), ndmin = 2)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(0.0)
        if self.C == 0:
            G = cvxopt.matrix(np.diag(np.ones(X.shape[0] * 2) * -1))
            h = cvxopt.matrix(np.zeros(X.shape[0] * 2))
        else:
            G_upper = np.diag(np.ones(X.shape[0] * 2) * -1)
            G_lower = np.identity(X.shape[0] * 2)
            h_upper = np.zeros(X.shape[0] * 2)
            h_lower = np.ones(X.shape[0] * 2) * 0.5
            G = np.vstack((G_upper, G_lower))
            h = np.hstack((h_upper, h_lower))
            G = cvxopt.matrix(G)
            h = cvxopt.matrix(h)
        params = cvxopt.solvers.qp(P, q, G, h, A, b)
        params = np.ravel(params['x'])
        return params

    def train(self, X, y):
        self.kernel_matrix = np.zeros((X.shape[0], X.shape[0]))
        self.t = np.zeros(y.shape[0])
        self.params = np.zeros(X.shape[1])
        self.intercept = 0
        params = self.get_params(X, y)
        self.lagrangevalues = params
        self.supportvectors = X
        self.supportvectorclasses = y

        intercept_term = 0
        shape = self.lagrangevalues.shape[0]/2
        for i in range(X.shape[0]):
            intercept_term = 0
            for j in range(shape):
                intercept_term = intercept_term + (self.lagrangevalues[j] - self.lagrangevalues[j + shape]) * self.kernel(X[i], self.supportvectors[j], self.degree)
            self.intercept = self.intercept + y[i] - intercept_term
        self.intercept = self.intercept / X.shape[0]

        return self

    def predict(self, X):
        y_predict = np.zeros(X.shape[0])
        self.intercept = 2
        shape = self.lagrangevalues.shape[0]/2
        for i in range(X.shape[0]):
            y = 0
            for j in range(shape):
                y = y + (self.lagrangevalues[j] - self.lagrangevalues[j + shape]) * self.kernel(X[i], self.supportvectors[j], self.degree)
            y_predict[i] = y
        return (y_predict + self.intercept)
