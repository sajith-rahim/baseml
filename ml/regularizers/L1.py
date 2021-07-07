import numpy as np


class L1:
    """
    Lasso Regularizer
    |w|
    """

    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, w):
        return self._lambda * np.linalg.norm(w)

    def grad(self, w):
        return self._lambda * np.sign(w)
