import numpy as np

class ElasticNetL1L2:
    """
    Elastic Net
    lambda1 * |w| + lambda2 * wTw
    """

    def __init__(self, lambda1, lamda2):
        self.lambda1 = lambda1;
        self.lambda2= lamda2;

    def __call__(self, w):
        l1 = self.lambda1 * np.linalg.norm(w)
        l2 = self.lambda2 * w.T.dot(w)
        return l1 + l2

    def gradient(self, w):
        l1 = self.lambda1 * np.sign(w)
        l2 = self.lambda2 * w
        return l1+l2