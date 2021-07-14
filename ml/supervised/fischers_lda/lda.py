import numpy as np

from ml.base import BaseEstimator
from ml.utils.data_utils import calculate_covariance_matrix


class LDA(BaseEstimator):
    """
    Fischer's Linear Discriminant Analysis
    maximize mean sep. of projections and minimize variance of each class
    J(w) = m2-m1 / s1^2 + s2^2
    """

    def __init__(self):
        self.w = None

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise ValueError("Implementation is for 2 Class")

        X_Class1 = X[y == 0]
        X_Class2 = X[y == 1]

        s1_square = calculate_covariance_matrix(X_Class1)
        s2_square = calculate_covariance_matrix(X_Class2)
        s = s1_square + s2_square

        m1 = X_Class1.mean(0)
        m2 = X_Class2.mean(0)

        mean_diff = np.atleast_1d(m1 - m2)

        # w = (m1 - mu2) / (sigma1 + sigma2)
        self.w = np.linalg.pinv(s).dot(mean_diff)

    def transform(self, X, y):
        self.fit(X, y)
        # Project data onto vector w
        X_transform = X.dot(self.w)
        return X_transform

    def predict(self, X):
        y_pred = []
        for row in X:
            h = row.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred
