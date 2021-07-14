import numpy as np


class Linear:
    """
    Linear Kernal
    """

    def __init__(self):
        self.name = 'Linear'

    def __call__(self, x, y):
        return np.dot(x, y.T)

    def __repr__(self):
        return self.name
