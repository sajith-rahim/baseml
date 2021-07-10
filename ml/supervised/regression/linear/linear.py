from ml.regularizers.L2 import L2
from ml.supervised.regression.regression import Regression

import numpy as np

class LinearRegression(Regression):
    """Linear model.
    Parameters:
    -----------
    n_iter: float
        max number of iterations
    learning_rate: float
        step size
    use_gd: boolean
        If false Use:  (X.TX)^-1 X.T y => Moore Penrose Inverse
        else use gradient descent.
    """

    def __init__(self, n_iter=100, lr=0.01, use_gd=True, reg_coeff = None):
        self.use_gd = use_gd
        if reg_coeff is None:
            # No regularization
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = L2(_lambda=reg_coeff)
        super(LinearRegression, self).__init__(n_iter=n_iter, lr=lr)

    def fit(self, X, y, verbose=True):
        if not self.use_gd:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)
            # Moore-Penrose pseudo-inverse
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            # (X.TX)^-1
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass
