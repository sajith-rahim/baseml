import math
import numpy as np

from ml.base import BaseEstimator


class Regression(BaseEstimator):
    """
    Base Regression Model
    """

    def __init__(self, n_iter=1000, learning_rate=0.001):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.training_errors = []
        self.tolerance = 0.0001
        #self.regularization = lambda x: 0
        #self.regularization.grad = lambda x: 0

    def initialize_weights(self, n_features):
        """
        Initialize weights:  [-1/N, 1/N]
        Refer readme.
        """
        _range = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-_range, _range, (n_features,))

    def fit(self, X, y, verbose=True):
        # w0 is added to X (as bias) at index 0 along first axis
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        for i in range(self.n_iter):
            y_pred = X.dot(self.w)
            # sq. loss
            loss = np.mean(0.5 * (y_pred - y) ** 2 + self.regularization(self.w))
            self.training_errors.append(loss)
            if verbose:
                print("Iteration %s, error %s" % (i, loss))
            # sq.loss gradient
            grad_w = -(y_pred - y).dot(X) + self.regularization.grad(self.w)
            # Update the weights
            self.w = self.w - self.learning_rate * grad_w

            # case
            """
            lossdiff = np.linalg.norm(self.training_errors[i - 1] - self.training_errors[i])
            if lossdiff < self.tolerance and i > 100:
                print("Converged.")
                break
            """

            if loss == float('inf') or loss == float('-inf'):
                print("Underflow/Overflow.")
                self.training_errors.pop()
                break



    def predict(self, X):
        # insert w0
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass
