import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

from ml.supervised.fischers_lda.lda import LDA
from ml.supervised.regression.bayesian.bayesian import BayesianRegression
from ml.utils.viz import plot_in_2d

if __name__ == '__main__':
    print('baseml')

    X, y = make_classification(n_samples=1000, n_features=9, n_informative=5,
                               random_state=999, n_classes=2, class_sep=1.50, )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

    lda = LDA()
    lda.fit(X_train, y_train)
    ypred = lda.predict(X_test)
    accuracy = accuracy_score(y_test, ypred)
    print("Accuracy:", accuracy)

    plot_in_2d(X_test, ypred, title="2 Class Fischer's LDA", accuracy=accuracy)
