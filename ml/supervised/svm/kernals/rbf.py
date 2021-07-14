import numpy as np
from scipy.spatial import distance


class RadialBasisFunction:
    """
    Radial Basis Function Kernal
    """

    def __init__(self, gamma=.1):
        self.name = 'Radial Basis Function'
        self.gamma = gamma

    def __call__(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        return np.exp(-self.gamma * distance.cdist(x, y, 'euclidean') ** 2).flatten()

    def __repr__(self):
        return self.name
