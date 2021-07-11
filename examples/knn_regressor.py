import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error

from ml.supervised.knn.knn import KNearestNeighbours


def run():
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    model = KNearestNeighbours(is_classifier=False, measure='mean')

    model.fit(diabetes_X_train, diabetes_y_train)

    y_pred = model.predict(diabetes_X_test)
    print(y_pred)
    mse = mean_squared_error(diabetes_y_test, y_pred)
    print("Mean squared error:", mse)

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.scatter(diabetes_X_test, y_pred, color='blue', linewidth=1)

    plt.xticks(())
    plt.yticks(())

    plt.show()


if __name__ == '__main__':
    print('baseml')
    run()
