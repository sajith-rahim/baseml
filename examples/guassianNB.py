import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml.supervised.naive_bayes.gaussian_naive_bayes import GaussianNaiveBayes
from ml.utils.viz import plot_in_2d

def run():
    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    model = GaussianNaiveBayes()

    model.fit(X_train, y_train)

    y_cap = model.predict(X_test)
    print(y_cap)
    y_cap = np.reshape(y_cap, y_test.shape)

    accuracy = accuracy_score(y_test, y_cap)
    print("Accuracy:", accuracy)

    # Plot outputs
    # plot_in_2d(X_test, y_test, title="KNN", accuracy=accuracy, legend_labels=data.target_names)
    plot_in_2d(X_test, y_cap, title="GNB", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == '__main__':
    print('baseml')

    run()