import numpy as np
from scipy.stats import multivariate_normal

from ml.base import BaseEstimator
from ml.utils.distributions import scaled_inv_chi_sq_rvs


class BayesianRegression(BaseEstimator):

    def __init__(self, n_draws, mu_0, nu_0, omega_0, scale_param_sigma, credible_interval=95):
        self.n_draws = n_draws
        self.mu_0 = mu_0
        self.nu_0 = nu_0
        self.omega_0 = omega_0
        self.scale_param_sigma = scale_param_sigma
        self.credible_interval = credible_interval

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        XT_X = X.T.dot(X)

        # Moore Penrose : (X.TX)^-1 X.Ty)
        beta_hat = np.linalg.pinv(XT_X).dot(X.T).dot(y)

        # Assuming Congugate Prior
        # If prior and posterior belongs to same class then prior is congugate to likelihood

        # Posterior mean mu_n : Normal
        # mu_n = (Omega_n)^-1 . (X.TX (X.TX)^-1 X.Ty) + Omega0.Mu0)

        omega_n = XT_X + self.omega_0
        omega_n_inv = np.linalg.pinv(omega_n)
        mu_n = omega_n_inv.dot(XT_X.dot(beta_hat) + self.omega_0.dot(self.mu_0))

        # Posterior nu_n :Scaled inverse chi-squared
        nu_n = self.nu_0 + n_samples



        term_0 = self.check(self.mu_0.T.dot(self.omega_0))
        term_1 = self.check(term_0.dot(self.mu_0))

        interim_term = y.T.dot(y) + term_1 - mu_n.T.dot(omega_n.dot(mu_n))
        sigma_n_square = (1.0 / nu_n) * (self.nu_0 * self.scale_param_sigma + (interim_term))

        # Simulate parameter values for n_draws
        beta_draws = np.empty((self.n_draws, n_features))
        for i in range(self.n_draws):
            sigma_sq = scaled_inv_chi_sq_rvs(n=1, df=nu_n, scale=sigma_n_square)
            beta = multivariate_normal.rvs(size=1, mean=mu_n[:], cov=sigma_sq * np.linalg.pinv(omega_n))
            beta_draws[i, :] = beta

        # Select the mean of the simulated variables as the ones used to make predictions
        self.w = np.mean(beta_draws, axis=0)

        # Lower and upper boundary of the credible interval
        lower_eti = 50 - self.credible_interval / 2
        upper_eti = 50 + self.credible_interval / 2
        self.equal_tail_interval = np.array(
            [[np.percentile(beta_draws[:, i], q=lower_eti), np.percentile(beta_draws[:, i], q=upper_eti)] \
             for i in range(n_features)])

    def predict(self, X, equal_tail_interval=False):

        y_pred = X.dot(self.w)
        # If the lower and upper boundaries for the 95%
        # equal tail interval should be returned
        if equal_tail_interval:
            lower_w = self.equal_tail_interval[:, 0]
            upper_w = self.equal_tail_interval[:, 1]
            y_lower_pred = X.dot(lower_w)
            y_upper_pred = X.dot(upper_w)
            return { 'pred': y_pred, 'lower': y_lower_pred, 'higher': y_upper_pred}

        return y_pred

    def check(self, x):
        #1D Case
        # check = lambda x: x if isinstance(x, np.ndarray) else np.array(x)
        #x = x if isinstance(x, np.ndarray) else np.array(x)
        return x

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass

