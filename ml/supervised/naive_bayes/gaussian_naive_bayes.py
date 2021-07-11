import numpy as np
from ml.base import BaseEstimator


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive Bayes
    - numerical features
    """
    def __init__(self):
        self.parameters = []
        #self.n_classes = np.unique(y)
        #super(NaiveBayes,self).__init__(X,y)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = []
        # Calculate the mean and variance of each feature for each class
        for i, _class in enumerate(self.classes):
            # rows with class = current class
            X_subset = X[np.where(y == _class)]
            self.parameters.append([])
            # Add the mean and variance for each feature (column)
            for col in X_subset.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)

        #print(parameters)

    def calculate_class_prob(self, c):
        """
        P(c) : prior probability
        = # of points with class c / total # of points
        """
        #length = self.y.shape[0]
        #prob = np.where(self.y == c)[0].shape[0]/ length
        return np.mean(self.y == c)

    def calculate_gaussian_ll(self, mu, sigma, point):
        """ Gaussian likelihood of the data x given mean and var """
        _eps = 1e-4  # Laplace Smoothing
        _coeff = 1.0 / np.sqrt(2.0 * np.pi * sigma + _eps)
        _exponent = np.exp(-(np.square(point - mu) / (2 * sigma + 2 * _eps)))
        return _coeff * _exponent

    def classify(self, sample):
        """
        posterior = likelihood * prior / evidence
        """
        _posteriors = []
        # Go through list of classes
        for i, c in enumerate(self.classes):
            # Initialize posterior = prior
            posterior = self.calculate_class_prob(c)
            for feature_value, params in zip(sample, self.parameters[i]):
                likelihood = self.calculate_gaussian_ll(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            _posteriors.append(posterior)
        # MAP estimate
        return self.classes[np.argmax(_posteriors)]

    def predict(self, X):
        return [self.classify(q) for q in X]