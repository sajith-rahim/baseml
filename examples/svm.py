import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml.supervised.regression.logistic.logistic import LogisticRegression
from ml.supervised.svm.support_vector_machine import SVM
from ml.utils.data_utils import normalize
from ml.utils.viz import plot_in_2d


def run():
    X, y = make_classification(n_samples=1000, n_features=9, n_informative=5,
                               random_state=999, n_classes=2, class_sep= 1.0, )

    two_class = np.vectorize(lambda x: -1 if x == 0 else 1)
    y = two_class(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

    clf = SVM()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    plot_in_2d(X_test, y_pred, title="SVM RBF", accuracy=accuracy)


if __name__ == '__main__':
    print('baseml')
    run()
