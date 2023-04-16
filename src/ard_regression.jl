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

function woodbury_inversion(X, sig2y, gamma, n_samp, n_col)
    inv_gamma = 1 ./ gamma 

    S = pinv(I(n_samp) .* sig2y .+ X * (inv_gamma .* X'))

    S = S * (X .* inv_gamma')

    S = -((inv_gamma' .* X)' * S)

    S = S .+ (I(n_col) .* inv_gamma )
    
    return S
end

function update_covar(X, sig2y, gammas, n_samp, n_col)
    if n_samp > n_col
        S = pinv((1 / sig2y) * X'X .+ diagm(gammas))
    else
        S = woodbury_inversion(X, sig2y, gammas, n_samp, n_col)
    end

    return S
end

function fit_ard_mackay(X, y; tol=1e-2,  max_iter=50, max_gamma=100)
    n_samp, n_col = size(X)

    gammas =  ones(n_col)

    a, b, c, d = ones(4) .* 1e-4

    sig2y = 1.

    S = update_covar(X, sig2y, gammas, n_samp, n_col)

    mu = (1 / sig2y) * S * X' * y

    g = 1 .- gammas .* diag(S)

    gammas = (g .+ 2*a) ./ (mu.^2 .+ 2*b)

    sig2y = (n_samp - sum(g) + 2*c) / (norm(y - X*mu, 2) + 2*d)

    keep_coefs = ones(Bool, n_col)

    prune_coefs = .!keep_coefs

    mu_prev = ones(n_col) .* 10000

    ml = fill(NaN, max_iter)

    for iter in 1:max_iter

        S = update_covar(X[:, keep_coefs], sig2y, gammas[keep_coefs], n_samp, size(X[:, keep_coefs], 2))

        mu[keep_coefs] = (1 / sig2y) * S * X[:, keep_coefs]' * y
    
        ml[iter] = -log(det(S)) - n_samp * log(1/sig2y) - sum(log.(gammas[keep_coefs]))

        g = 1 .- gammas[keep_coefs] .* diag(S)

        gammas[keep_coefs] = (g .+ 2*a) ./ (mu[keep_coefs].^2 .+ 2*b)

        sig2y = (n_samp - sum(g) + 2*c) / (norm(y - X*mu, 2) + 2*d)

        prune_coefs[gammas .>= max_gamma] .= true

        keep_coefs = .!prune_coefs

        mu[prune_coefs] .= 0

        if sum(abs.(mu .- mu_prev)) < tol
            println(string("converged in $iter", " iterations"))
            break
        end

        mu_prev = mu
    end

    S = update_covar(X, sig2y, gammas, n_samp, size(X, 2))

    mu = (1 / sig2y) * S * X' * y

    g = 1 .- gammas .* diag(S)

    sig2y = (n_samp - sum(g) + 2*c) / (norm(y - X*mu, 2) + 2*d)

    return (
        sig2y = sig2y,
        w = mu, 
        gamma = gammas,
        ml = ml[.!isnan.(ml)],
        S = S
    )

end
