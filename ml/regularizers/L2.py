class L2:
    """
    Ridge Regularizer
    ||w||^2_2
    """

    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, w):
        return self._lambda * w.T.dot(w)

    def grad(self, w):
        return self._lambda * w