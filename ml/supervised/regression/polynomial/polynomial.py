from abc import ABC

from ml.regularizers.L2 import L2
from ml.regularizers.NoRegularization import NoRegularization

from itertools import combinations_with_replacement

import numpy as np

from ml.supervised.regression.regression import Regression


class PolynomialRegression(Regression):
    """
    convert original feature space into transformed space
    allows us to find linearity in transformed space
    """

    def __init__(self, degree, n_iter=1000, reg_coeff=None, lr=0.001):
        self.degree = degree
        if reg_coeff is None:
            self.regularization = NoRegularization();
        else:
            self.regularization = L2(_lambda=reg_coeff)

        super(PolynomialRegression, self).__init__(n_iter=n_iter,
                                                   lr=lr)

    def fit(self, X, y):
        X = self.transform(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        X = self.transform(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)

    def transform(self, X, degree):
        n_samples, n_features = np.shape(X)

        def index_combinations():
            combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
            flat_combs = [item for sublist in combs for item in sublist]
            return flat_combs

        combinations = index_combinations()
        n_output_features = len(combinations)
        X_transformed = np.empty((n_samples, n_output_features))

        for i, index_combs in enumerate(combinations):
            X_transformed[:, i] = np.prod(X[:, index_combs], axis=1)

        return X_transformed

    def newton_fit(self, X, y, verbose=True):
        pass  # base class method -irrelevant
