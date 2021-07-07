import numpy as np


class BaseEstimator:
    """Base class for all estimators."""
    is_supervised = True

    def _validate_input(self, X, y=None):
        """
        Validate whether inputs are in required format.

        Parameters
        ----------
        X : array-like
            features.
        y : array-like
            target.
        """
        # validate feature_matrix
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("received an empty feature matrix!")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        # validate target values

        if self.is_supervised:
            if y is None:
                raise ValueError("target array is missing.")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("received an empty target array.")

        self.y = y

    def fit(self, X, y=None, verbose=False):
        self._validate_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        raise NotImplementedError()

    def set_is_supervised(self, is_supervised):
        self.is_supervised = is_supervised

    def get_is_supervised(self):
        return self.is_supervised

    def get_params(self, deep=True):
        raise NotImplementedError()

    def set_params(self, **params):
        raise NotImplementedError()
