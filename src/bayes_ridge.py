import numpy as np

def sse(x, y):
    return np.sum((x - y)**2)

def fit_bayes_ridge(X, y, tol=1e-2, max_iter=100, pen_init=10):
    n_samp, n_col = np.shape(X)

    mu = np.zeros(n_col)

    I_n = np.eye(n_col)

    XX = X.T @ X 

    Xy = X.T @ y 

    w = np.linalg.lstsq(XX + I_n * pen_init, Xy, rcond=None)[0]

    sig2y = see(y, X @ w)

    alpha = pen_init / sig2y

    covar = np.linalg.pinv( 1 / sig2y * XX+ I_n * alpha)

    mu = 1 / sig2y * covar @ Xy

    gammas = 1 - alpha * np.diag(covar)

    alpha = (n_col - alpha * np.trace(covar)) / np.sum(mu**2)

    sig2y = sse(y, X @ mu) / (n_samp - np.sum(gammas))

    lml = np.full(max_iter, np.nan)

    lml[0] = -np.log(np.linalg.det(covar)) - n_samp * np.log(1/sig2y) - n_samp *  np.log(alpha)

    params = np.array([alpha, sig2y])

    for iter_j in range(1, max_iter):
        covar = np.linalg.pinv( 1 / sig2y * XX + I_n * alpha)

        mu = 1 / sig2y * covar @ Xy

        gammas = 1 - alpha * np.diag(covar)

        alpha = (n_col - alpha * np.trace(covar)) / np.sum(mu**2)

        sig2y = sse(y, X @ mu) / (n_samp - np.sum(gammas))
 
        lml[iter_j] = -np.log(np.linalg.det(covar)) - n_samp * np.log(1/sig2y) - n_samp *  np.log(alpha)

        if np.linalg.norm( params - np.array([alpha, sig2y])) < tol:
            print(f"converged in {iter_j} iterations")
            break

        params = np.array([alpha, sig2y])

    estimates =  {
        "sig2y" : sig2y, 
        "w" : mu,
        "covar" : covar, 
        "alpha" : alpha,
        "lml" : lml[~np.isnan(lml)]
    }

    return estimates
