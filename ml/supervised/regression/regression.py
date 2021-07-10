import math
import numpy as np

from ml.base import BaseEstimator


class Regression(BaseEstimator):
    """
    Base Regression Model
    """

    def __init__(self, n_iter=1000, lr=0.0001, squash = False):
        self.n_iter = n_iter
        self.learning_rate = lr
        self.training_loss = []
        self.tolerance = 0.0001

        self.squash = squash
        #self.regularization = lambda x: 0
        #self.regularization.grad = lambda x: 0

    def initialize_weights(self, n_features):
        """
        Initialize weights:  [-1/N, 1/N]
        Refer readme.
        """
        _range = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-_range, _range, (n_features,))

    def newton_fit(self, X, y, verbose = True):

        """


        # Make a diagonal matrix of the sigmoid gradient column vector
        i in range(n_iter)
            diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
            # Batch optimize as
            \theta_{n+1} = \theta_{n} + H_{\ell(\hat{\theta})}^{-1}\nabla \ell(\theta)
        """
        raise NotImplementedError('yet to be implemented')

    def fit(self, X, y, verbose=True):
        # w0 is added to X (as bias) at index 0 along first axis
        X = np.insert(X, 0, 1, axis=1)
        self.training_loss = []
        self.initialize_weights(n_features=X.shape[1])

        for i in range(self.n_iter):
            y_cap = X.dot(self.w)
            if not self.squash:
                # sq. loss
                _loss = np.mean(0.5 * np.square(y - y_cap) + self.regularization(self.w))
                if verbose:
                    print("Iteration %s, error %s" % (i, _loss))
                self.training_loss.append(_loss)
                # sq.loss gradient = -2(y - y_cap) X.T
                grad_w = -(y - y_cap).dot(X) + self.regularization.gradient(self.w)
                # Update
                self.w -= self.learning_rate * grad_w

                # case
                """
                lossdiff = np.linalg.norm(self.training_loss[i - 1] - self.training_loss[i])
                if lossdiff < self.tolerance and i > 100:
                    print("Converged.")
                    break
                """
                if _loss == float('inf') or _loss == float('-inf'):
                    print("Underflow/Overflow.")
                    self.training_loss.pop()
                    break

            else:
                y_cap = 1 / (1 + np.exp(-y_cap))
                grad_w = -(y - y_cap).dot(X)
                self.w -= self.learning_rate * grad_w





    def predict(self, X):
        # insert w0
        X = np.insert(X, 0, 1, axis=1)
        y_cap = X.dot(self.w)

        if self.squash:
            y_cap = 1 / (1 + np.exp(-y_cap))
            y_cap = np.round(y_cap).astype(int)

        return y_cap

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass
