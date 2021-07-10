import numpy as np


class NoRegularization:
    """
    No Regualrization
    Note: Class made for consistency
    """

    def __init__(self):
        pass

    def __call__(self, w):
        return 0.0*w

    @staticmethod
    def gradient(w):
        return 0.0*w
