import numpy as np

from ml.base import BaseEstimator
from ml.supervised.svm.kernals.linear import Linear
from ml.supervised.svm.kernals.rbf import RadialBasisFunction


class SVM(BaseEstimator):
    """
    Support Vector Machine
    Simplified Sequential Minimum Optimization
    Reference:
    Platt, John. Fast Training of Support Vector Machines using Sequential Minimal Optimization
    http://research.microsoft.com/˜jplatt/smo.html

     Parameters:
    -----------
    kernal  : function
    Penalty : float
        1 / Lambda
    """

    def __init__(self, kernal=None, C=.6, tolerance=.001, max_iter=1000):
        self.kernal = kernal if kernal is not None else RadialBasisFunction()
        self.intercept = 0
        self.C = C
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.alpha = None  # Langrangian multiplier
        self.kernal_matrix = None

    def fit(self, X, y, verbose=False):
        self.X = X
        self.y = y

        self.m = X.shape[0]  # no of samples
        self.n = X.shape[1]  # no of features

        self.alpha = np.zeros(self.m)  # init 1-->n alpha_i

        self.kernal_matrix = self.init_kernal_matrix(self.m, self.n)

        itr = 0
        while itr < self.max_iter:
            itr += 1
            alpha_prev = np.copy(self.alpha)

            for j in range(self.m):

                # Select i s.t j != i randomly.
                i = j
                while i == j:
                    i = np.random.randint(0, self.m - 1)

                # Calculate E_j = f(x(j)) − y(j)
                err_i = self.calculate_error(i)
                err_j = self.calculate_error(j)

                if (self.y[j] * err_j < -self.tolerance and self.alpha[j] < self.C) or (
                        self.y[j] * err_j > self.tolerance and self.alpha[j] > 0):

                    lower_bound, higher_bound = self.get_bounds_foe_alpha(i, j)
                    old_alpha_j, old_alpha_i = self.alpha[j], self.alpha[i]

                    # eta = 2 K( x(i), x(j) )  − K( x(i), x(i)) − K( x(j), x(j))
                    eta = 2.0 * self.kernal_matrix[i, j] - self.kernal_matrix[i, i] - self.kernal_matrix[j, j]

                    if eta >= 0:
                        # continue to next i
                        continue
                    # If not > 0, update alpha by eqn 12
                    self.alpha[j] -= (self.y[j] * (err_i - err_j)) / eta
                    # and eqn 15
                    self.alpha[j] = self.bound_alpha(self.alpha[j], lower_bound, higher_bound)

                    # eqn 16
                    self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (old_alpha_j - self.alpha[j])

                    # eqn 17
                    b1 = self.intercept - err_i - self.y[i] * (self.alpha[i] - old_alpha_j) * self.kernal_matrix[i, i] - \
                         self.y[j] * (self.alpha[j] - old_alpha_j) * self.kernal_matrix[i, j]
                    # eqn 18
                    b2 = self.intercept - err_j - self.y[j] * (self.alpha[j] - old_alpha_j) * self.kernal_matrix[j, j] - \
                         self.y[i] * (self.alpha[i] - old_alpha_i) * self.kernal_matrix[i, j]

                    # eqn 19
                    if 0 < self.alpha[i] < self.C:
                        self.intercept = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.intercept = b2
                    else:
                        self.intercept = 0.5 * (b1 + b2)

                delta = np.linalg.norm(self.alpha - alpha_prev)
            if delta < self.tolerance:
                print("Converged.")
                print(self.alpha)
                break

    def init_kernal_matrix(self, m, n):
        kernal_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(n):
                kernal_matrix[i, j] = self.kernal(self.X[i], self.X[j])

        return kernal_matrix

    def calculate_error(self, row_index):
        # E_j = f(x_i) - y_j
        return self.predict_row(self.X[row_index]) - self.y[row_index]

    def predict_row(self, X):
        # f(x) = Sigma_1:m [ alpha_i y_i X.T ] + b
        k_v = self.kernal(self.X, X)
        return np.dot((self.alpha * self.y).T, k_v.T) + self.intercept

    def get_bounds_foe_alpha(self, i, j):
        # Find constraints on alpha_j
        # L ≤ alpha_j ≤ H and 0 ≤ alpha_j ≤ C
        # Simplified SMO Equations (10) and (11)

        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H

    def bound_alpha(self, alpha, lower_bound, higher_bound):
        if alpha > higher_bound:
            alpha = higher_bound
        elif alpha < lower_bound:
            alpha = lower_bound
        return alpha

    def predict(self, X):
        n = X.shape[0]
        res = np.zeros(n)
        for i in range(n):
            res[i] = np.sign(self.predict_row(X[i, :]))
        return res
