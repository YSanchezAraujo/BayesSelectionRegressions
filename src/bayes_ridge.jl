using LinearAlgebra, Distributions
using StatsBase;

sse(x, y) = sum((x .- y).^2)

function fit_bayes_ridge(X, y; tol=1e-2,  max_iter=100)
    n_samp, n_col = size(X)

    mu = zeros(n_col)

    lml = fill(NaN, max_iter)

    I_n = I(n_col)
    
    XX = X'X 
    
    Xy = X'y

    w = (XX + I_n) \ Xy

    sig2y = sse(y, X*w)

    alpha = sig2y

    S = pinv(1 / sig2y * XX .+ I_n * alpha)

    mu = 1 / sig2y * S * Xy

    gammas = 1 .- alpha .* diag(S)

    alpha = (n_col - alpha * tr(S)) ./ sum(mu.^2)

    sig2y = sse(y, X*mu) / (n_samp - sum(gammas))

    params = [alpha, sig2y]

    lml[1] = -log(det(S)) - n_samp * log(1/sig2y) - n_samp * log(alpha)

    for iter in 2:max_iter

        S = pinv(1 / sig2y * XX .+ I_n * alpha)

        mu = 1 / sig2y * S * Xy
    
        gammas = 1 .- alpha .* diag(S)

        alpha = (n_col - alpha * tr(S)) ./ sum(mu.^2)

        sig2y = sse(y, X*mu)  / (n_samp - sum(gammas))

        lml[iter] = -log(det(S)) - n_samp * log(1/sig2y) - n_samp * log(alpha)

        if norm(params .- [alpha, sig2y]) < tol
            println(string("converged in $iter", " iterations"))
            break
        end

        params = [alpha, sig2y]

    end

    return (
        sig2y = sig2y,
        w = mu, 
        alpha = alpha,
        covar = S, 
        lml = lml[.!isnan.(lml)]
    )

end
