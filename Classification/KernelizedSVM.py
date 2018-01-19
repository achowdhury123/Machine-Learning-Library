import numpy as np
import cvxopt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

class KernelizedSVM:
    def linear_kernel(self, x1, x2, degree=0):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x, y, degree):
        return (np.dot(x, y) + 1) ** degree

    def __init__(self, C=0, kernel = "linear", degree=0):
        self.params = None
        self.intercept = None
        self.kernel_matrix = None
        self.t = None
        self.lagrangevalues = None
        self.supportvectors = None
        self.supportvectorclasses = None
        self.C = C
        if kernel == "linear":
            self.kernel = self.linear_kernel
        else:
            self.kernel = self.polynomial_kernel
        self.degree = degree

    def get_params(self, X, y):
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.kernel_matrix[i, j] = self.kernel(X[i, :], X[j, :], self.degree)
        for i in range(y.shape[0]):
            if y[i] == 1:
                self.t[i] = 1
            else:
                self.t[i] = -1
        P = cvxopt.matrix(np.outer(self.t, self.t) * self.kernel_matrix)
        q = cvxopt.matrix(np.ones(X.shape[0]) * -1)
        A = cvxopt.matrix(self.t, (1, X.shape[0]))
        b = cvxopt.matrix(0.0)
        if self.C == 0:
            G = cvxopt.matrix(np.diag(np.ones(X.shape[0]) * -1))
            h = cvxopt.matrix(np.zeros(X.shape[0]))
        else:
            G_upper = np.diag(np.ones(X.shape[0]) * -1)
            G_lower = np.identity(X.shape[0])
            h_upper = np.zeros(X.shape[0])
            h_lower = np.ones(X.shape[0]) * self.C
            G = cvxopt.matrix(np.vstack((G_upper, G_lower)))
            h = cvxopt.matrix(np.hstack((h_upper, h_lower)))
        params = cvxopt.solvers.qp(P, q, G, h, A, b)
        params = np.ravel(params['x'])
        return params

    def train(self, X, y):
        self.kernel_matrix = np.zeros((X.shape[0], X.shape[0]))
        self.t = np.zeros(y.shape[0])
        self.params = np.zeros(X.shape[1])
        self.intercept = 0
        params = self.get_params(X, y)
        supportvectors = params > 0.0001
        self.lagrangevalues = params[supportvectors]
        self.supportvectors = X[supportvectors]
        self.supportvectorclasses = self.t[supportvectors]

        for i in range(self.supportvectors.shape[0]):
            self.params = self.params + self.lagrangevalues[i] * self.supportvectorclasses[i] * self.supportvectors[i]

        intercept_term = 0
        for i in range(self.supportvectors.shape[0]):
            intercept_term = 0
            for j in range(self.supportvectors.shape[0]):
                intercept_term = intercept_term + self.lagrangevalues[j] * self.supportvectorclasses[j] * self.kernel(self.supportvectors[i], self.supportvectors[j], self.degree)
            self.intercept = self.intercept + self.supportvectorclasses[i] - intercept_term
        self.intercept = self.intercept / self.supportvectors.shape[0]

        return self

    def predict(self, X):
        y_predict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y = 0
            for j in range(self.supportvectors.shape[0]):
                y = y + self.lagrangevalues[j] * self.supportvectorclasses[j] * self.kernel(X[i], self.supportvectors[j], self.degree)
            y_predict[i] = y
        class_values = np.sign(y_predict + self.intercept)
        zero_indices = class_values == -1
        class_values[zero_indices] = 0
        return (class_values)
