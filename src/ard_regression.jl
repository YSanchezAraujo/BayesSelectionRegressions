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
function fit_ard(X, y; sig2y = nothing, max_iter=100, tol = 1e-2, verbose = false)
    n_samp, n_col = size(X)

    z = rand(n_col)

    sig2y = isnothing(sig2y) ? 1 : sig2y

    gammas = ones(n_col)

    w = Variable(n_col)

    problem = minimize(
        norm(y - X*w, 2) + 2 * sig2y * sum(sqrt.(z) .* abs.(w))
    )

    solve!(problem, SCS.Optimizer; silent_solver = !verbose)

    gammas_prev = gammas

    gammas = 1 ./ sqrt.(z) .* abs.(vec(w.value))

    S = weighted_x(y, X, sig2y, gammas, n_samp)

    S_inv = inv(S)

    ySy = y' * S_inv * y

    z = diag(X' * S_inv * X)

    for iter in 1:max_iter

        if norm(gammas_prev .- gammas) < tol 
            println(string("converged in $iter", " iterations"))
            break 
        end

        problem = minimize(
            norm(y - X*w, 2) + 2 * sig2y * sum(sqrt.(z) .* abs.(w))
        )

        solve!(problem, SCS.Optimizer; silent_solver = !verbose)

        gammas = 1 ./ sqrt.(z) .* abs.(vec(w.value))
    
        S = weighted_x(y, X, sig2y, gammas, n_samp)

        S_inv = inv(S)

        ySy = y' * S_inv * y

        z = diag(X' * S_inv * X)

        gammas_prev = gammas
    end

    opt_S_inv = inv(weighted_x(y, X, sig2y, gammas, n_samp))

    w_ard = diagm(gammas) * X' * opt_S_inv * y

    return (w = w_ard, 
            gamma = gammas)

end
