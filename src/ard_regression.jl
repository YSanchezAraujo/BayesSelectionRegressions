using LinearAlgebra, Convex, SCS


"""
y: vector of target values
X: design matrix
sig2y: assumed noise in y
gammas: diagonal of regularization matrix
n_samp: number of samples
"""
function weighted_x(y, X, sig2y, gammas, n_samp)
    return sig2y * I(n_samp) .+ X * (gammas .* X')
end

"""
X: design matrix
y: vector of target values
"""
function fit_ard(X, y;  max_iter=100, tol = 1e-2, verbose = false)
    n_samp, n_col = size(X)

    z = rand(n_col)

    sig2y = 1.

    w = Variable(n_col)

    problem = minimize(
        norm(y - X*w, 2) + 2 * sig2y * sum(sqrt.(z) .* abs.(w))
    )

    solve!(problem, SCS.Optimizer; silent_solver = !verbose)

    gammas_prev = ones(n_col)

    gammas = 1 ./ sqrt.(z) .* abs.(evaluate(w))

    C = pinv(sig2y .* X'X .+ diagm(gammas))

    S = weighted_x(y, X, sig2y, gammas, n_samp)

    S_inv = pinv(S)

    z = diag(X' * S_inv * X)

    for iter in 1:max_iter

        problem = minimize(
            norm(y - X*w, 2) + 2 * sig2y * sum(sqrt.(z) .* abs.(w))
        )

        solve!(problem, SCS.Optimizer; silent_solver = !verbose)

        gammas = 1 ./ sqrt.(z) .* abs.(evaluate(w))

        C = pinv(sig2y .* X'X .+ diagm(gammas))

        sig2y = norm(y - X*evaluate(w), 2) / (n_samp - sum(1 .- gammas .* diag(C)))
    
        S = weighted_x(y, X, sig2y, gammas, n_samp)

        S_inv = pinv(S)

        z = diag(X' * S_inv * X)

        if norm(gammas_prev .- gammas) < tol 
            println(string("converged in $iter", " iterations"))
            break 
        end

        gammas_prev = gammas
    end

    opt_S_inv = pinv(weighted_x(y, X, sig2y, gammas, n_samp))

    w_ard = diagm(gammas) * X' * opt_S_inv * y

    C = pinv(sig2y .* X'X .+ diagm(gammas))

    sig2y = norm(y - X*w_ard, 2) / (n_samp - sum(1 .- gammas .* diag(C)))

    return (
        w = w_ard, 
        gamma = gammas, 
        C = C,
        sig2y = sig2y
    )

end

sse(x, y) = sum((x .- y).^2)

function fit_ard_mackay(X, y; tol=1e-2,  max_iter=50, max_alpha=100)
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

    ml = fill(NaN, max_iter)

    ml[1] = -log(det(S)) - n_samp * log(sig2y) - sum(log.(alphas[keep_coefs]))

    for iter in 2:max_iter

        S = pinv(1 / sig2y * X[:, keep_coefs]'X[:, keep_coefs] .+ diagm(alphas[keep_coefs]))

        mu[keep_coefs] = 1 / sig2y * S * X[:, keep_coefs]' * y
    
        ml[iter] = -log(det(S)) - n_samp * log(sig2y) - sum(log.(alphas[keep_coefs]))

        gammas = 1 .- alphas[keep_coefs] .* diag(S)

        alphas[keep_coefs] = (gammas .+ 2*a) ./ (mu[keep_coefs].^2 .+ 2*b)

        sig2y = (sse(y, X[:, keep_coefs]*mu[keep_coefs]) + 2*d) / (n_samp - sum(gammas) + 2*c)

        keep_coefs[alphas .> max_alpha] .= false

        mu[.!keep_coefs] .= 0

        if abs(ml[iter] - ml[iter-1]) < tol
            println(string("converged in $iter", " iterations"))
            break
        end

    end

    return (
        sig2y = sig2y,
        w = mu, 
        alpha = alphas,
        ml = ml[.!isnan.(ml)],
        C = S
    )

end
