import matplotlib.pyplot as plt
import numpy as np

from ml.utils.data_utils import calculate_covariance_matrix


def project_on_eigen(X, dim):
    """
    PrincipalComponentAnalysis()
    pca.fit(X, 2)
    X_transformed = pca.transform(X)
    """

    covariance = calculate_covariance_matrix(X)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    # Sort eigenvalues and eigenvector by largest eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx][:dim]
    eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :dim]
    # Project the data onto principal components
    X_transformed = X.dot(eigenvectors)

    return X_transformed


def plot_in_2d(X, y=None, title=None, accuracy=None, legend_labels=None):
    X_transformed = project_on_eigen(X, dim=2)
    x1 = X_transformed[:, 0]
    x2 = X_transformed[:, 1]
    class_distr = []

    cmap = plt.get_cmap('viridis')

    y = np.array(y).astype(int)

    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    # Plot the different class distributions
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    # Plot legend
    if legend_labels is not None:
        plt.legend(class_distr, legend_labels, loc=1)

    # Plot title
    if title:
        if accuracy:
            percentage = 100 * accuracy
            plt.suptitle(title)
            plt.title("Accuracy: %.1f%%" % percentage, fontsize=10)
        else:
            plt.title(title)

    # Axis labels
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.show()
