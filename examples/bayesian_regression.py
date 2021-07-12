import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error

from ml.supervised.regression.bayesian.bayesian import BayesianRegression


def run():
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]
    diabetes_X = np.insert(diabetes_X, 0, 1, axis=1)

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    n_samples, n_features = np.shape(diabetes_X_train)

    mu_0 = np.array([0.1] * n_features)
    omega_0 = np.diag([.001] * n_features)
    nu_0 = 1
    sig = 10
    cred_int = 100

    model = BayesianRegression(n_draws=100, mu_0=mu_0, nu_0=nu_0, omega_0=omega_0, scale_param_sigma=sig)

    model.fit(diabetes_X_train, diabetes_y_train)

    y_pred = model.predict(diabetes_X_test, True)

    mse = mean_squared_error(diabetes_y_test, y_pred['pred'])
    print("Mean squared error: %s" % (mse))

    # Plot outputs
    plt.scatter(np.delete(diabetes_X_test, 0, 1), diabetes_y_test, color='black')
    plt.plot(np.delete(diabetes_X_test, 0, 1), y_pred['pred'], color='blue', linewidth=1)
    plt.plot(np.delete(diabetes_X_test, 0, 1), y_pred['lower'], color='red', linewidth=1)
    plt.plot(np.delete(diabetes_X_test, 0, 1), y_pred['higher'], color='red', linewidth=1)

    plt.xticks(())
    plt.yticks(())

    plt.show()

if __name__ == '__main__':
    print('baseml')

    run()
