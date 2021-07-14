import numpy as np


class Polynomial:
    """
    Polynomial Kernal
    """

    def __init__(self, degree = 2):
        self.name = 'Polynomial'
        self.degree = degree

    def __call__(self, x, y):
        return np.dot(x, y.T) ** self.degree

    def __repr__(self):
        return self.name
