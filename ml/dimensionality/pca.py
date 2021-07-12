import numpy as np

from ml.utils.data_utils import calculate_covariance_matrix


class PrincipalComponentAnalysis:
    """
    Principal Component Analysis
    """

    def __init__(self):
        self.components = None
        self.eigenvectors = self.eigenvalues = None

    def fit(self, X, n_components):

        covariance_matrix = calculate_covariance_matrix(X)
        """
        if not all (len (row) == len (X) for row in X):
            raise ValueError("Not square matrix, use SVD for rectangular matrices.")
        """
        self.eigenvalues, self.eigenvectors = np.linalg.eig(covariance_matrix)

        idx = self.eigenvalues.argsort()[::-1]

        eigenvectors = self.eigenvectors[:, idx]
        self.eigenvectors = eigenvectors[:, :n_components]

        eigenvalues_squared = self.eigenvalues ** 2
        variance_ratio = eigenvalues_squared / eigenvalues_squared.sum()

        return {'explained variance': variance_ratio}

    def transform(self, X):
        if self.eigenvalues is None:
            raise AttributeError('Please fit before transforming!')

        return np.matmul(X, self.eigenvectors)

    def get_params(self):
        return {"eigval": self.eigenvalues, "eigvec": self.eigenvectors}
