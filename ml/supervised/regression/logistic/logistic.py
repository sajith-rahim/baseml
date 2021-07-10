from ml.regularizers.L2 import L2
from ml.regularizers.NoRegularization import NoRegularization
from ml.supervised.regression.regression import Regression

import numpy as np


class LogisticRegression(Regression):

    def __init__(self, n_iter=1000, use_gd=True, reg_coeff=None, lr=.01):
        self.n_iter = n_iter
        self.lr = lr
        self.use_gd = use_gd
        if reg_coeff is None:
            self.regularization = NoRegularization();
        else:
            self.regularization = L2(_lambda=reg_coeff)

        super(LogisticRegression, self).__init__(lr=lr, squash=True, n_iter=n_iter)

    def fit(self, X, y, verbose=True):
        if not self.use_gd:
            # raise NotImplementedError('Newtons method not implemented!')
            super(LogisticRegression, self).newton_fit(X, y, True)
        else:
            super(LogisticRegression, self).fit(X, y, True)

    #def predict(self, X):
    #    return super(LogisticRegression, self).predict(X, squash=True)
