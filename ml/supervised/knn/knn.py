import warnings
from collections import Counter

import numpy as np

from ml.base import BaseEstimator
from ml.utils.distances import euclidean_distance, cosine_dist


class KNearestNeighbours(BaseEstimator):

    def __init__(self, is_classifier, k=3, measure="mean", distance_measure = 'l2'):
        self.is_classifier = is_classifier
        if k % 3 == 0 and is_classifier:
            warnings.warn("set k as odd to avoid ties")
        self.k = k
        self.measure = measure if not is_classifier else None
        self.distance_measure = cosine_dist if distance_measure!= 'l2' else euclidean_distance

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def predict(self, X_test):
        n = X_test.shape[0];
        y_cap = np.empty(n)

        for i, query_point in enumerate(X_test):
            # find the l2 distance between query point and dataset points
            _distances = [self.distance_measure(query_point, x) for x in self.X]
            # get the indexes of k nearest neighbours
            idx = np.argsort(_distances)[:self.k]
            # get the labels of k nearest neighbors
            knn = np.array([self.y[i] for i in idx])
            # Vote or Avg
            if self.is_classifier:
                y_cap[i] = self.__vote(knn)
            else:
                y_cap[i] = self.__measure(knn)

        return y_cap

    @staticmethod
    def __vote(nearest_neighbors):
        most_common_label = Counter(nearest_neighbors).most_common(1)  # [('label', 2)]
        return most_common_label[0][0]

    def __measure(self, nearest_neighbors):
        if self.measure == "mean":
            y_cap = np.mean(nearest_neighbors)
        else:
            y_cap = np.median(nearest_neighbors)

        return y_cap

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass
