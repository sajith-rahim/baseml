from scipy.stats import chi2


def scaled_inv_chi_sq_rvs( n, df, scale):
    """
    geoR::rinvchisq, the
    return ((df * scale) / rchisq(n, df=df))
    """
    X = chi2.rvs(size=n, df=df)
    return df * scale / X