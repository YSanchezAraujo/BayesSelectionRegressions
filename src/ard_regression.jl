using LinearAlgebra, Distributions
using StatsBase;

sse(x, y) = sum((x .- y).^2)

function fit_ard(X, y; tol=1e-2,  max_iter=50, max_alpha=100)
    n_samp, n_col = size(X)

    alphas = ones(n_col)

    keep_coefs = ones(Bool, n_col)

    a, b, c, d = ones(4) .* 1e-4

    sig2y = 1.

    mu = zeros(n_col)

    S = pinv(1 / sig2y * X[:, keep_coefs]'X[:, keep_coefs] .+ diagm(alphas))

    mu[keep_coefs] = 1 / sig2y * S * X[:, keep_coefs]' * y

    gammas = 1 .- alphas[keep_coefs] .* diag(S)

    alphas[keep_coefs] = (gammas .+ 2*a) ./ (mu[keep_coefs].^2 .+ 2*b)

    sig2y = (sse(y, X*mu) + 2*d) / (n_samp - sum(gammas) + 2*c)

    lml = fill(NaN, max_iter)

    lml[1] = -log(det(S)) - n_samp * log(1/sig2y) - sum(log.(alphas[keep_coefs]))

    for iter in 2:max_iter

        S[keep_coefs, keep_coefs] = pinv(1 / sig2y * X[:, keep_coefs]'X[:, keep_coefs] .+ diagm(alphas[keep_coefs]))

        mu[keep_coefs] = 1 / sig2y * S[keep_coefs, keep_coefs] * X[:, keep_coefs]' * y
    
        lml[iter] = -log(det(S[keep_coefs, keep_coefs])) - n_samp * log(1/sig2y) - sum(log.(alphas[keep_coefs]))

        gammas = 1 .- alphas[keep_coefs] .* diag(S[keep_coefs, keep_coefs])

        alphas[keep_coefs] = (gammas .+ 2*a) ./ (mu[keep_coefs].^2 .+ 2*b)

        sig2y = (sse(y, X[:, keep_coefs]*mu[keep_coefs]) + 2*d) / (n_samp - sum(gammas) + 2*c)

        keep_coefs[alphas .> max_alpha] .= false

        mu[.!keep_coefs] .= 0

        if abs(lml[iter] - lml[iter-1]) < tol
            println(string("converged in $iter", " iterations"))
            break
        end

    end

    return (
        sig2y = sig2y,
        w = mu, 
        alpha = alphas,
        lml = lml[.!isnan.(lml)],
        C = S
    )

end
