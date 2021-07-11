import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

from ml.supervised.regression.logistic.logistic import LogisticRegression
from ml.utils.data_utils import normalize
from ml.utils.viz import plot_in_2d


def run():
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    # 2class
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)


if __name__ == '__main__':
    print('baseml')
    run()
