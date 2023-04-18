using LinearAlgebra, Optim;
import LogExpFunctions: logistic, softplus;
using StatsBase;
import Random: MersenneTwister;
using Distributions;



function nll(w, X, y, alphas)
    mu = X*w

    A = diagm(alphas)

    prior = 0.5 * w'*A*w

    return -y'mu + sum(softplus.(mu)) + prior
end

function grad_w(w, X, y, alphas)
    p = logistic.(X*w)

    return X' * (y .- p) - diagm(alphas)*w
end

function hess_w(w, X, y, alphas)
    p = logistic.(X * w)
    
    B = diagm(p .* (1 .- p))

   return  -(X' * B * X .+ diagm(alphas))
end

function newtons_method(w, X, y, alphas; tol=1e-2, max_iter=100)
    w_old = ones(length(w))

    for iter in 1:max_iter
        w = w .- pinv(hess_w(w, X, y, alphas)) * grad_w(w, X, y, alphas)

        if sum(abs.(w .- w_old)) < tol 
            break
        end

        w_old = w
    end

    return w
end


function ard_laplace(y, X; rng_state = nothing, tol=1e-3, max_iter=100, max_alpha=500)
    n_samp, n_col = size(X)

    a, b, c, d = ones(4) .* 1e-4

    rng_state = isnothing(rng_state) ? MersenneTwister(43) : rng_state

    alphas = ones(n_col)

    w_mp = optimize(w -> nll(w, X, y, alphas), rand(rng_state, n_col), Newton())

    w_mp = w_mp.minimizer

    keep_coefs = ones(Bool, n_col)

    p = logistic.(X * w_mp)
    
    B = diagm(p .* (1 .- p))

    hess = -(X[:, keep_coefs]' * B * X[:, keep_coefs] .+ diagm(alphas[keep_coefs]))

    Sigma = -inv(hess)

    gammas = 1 .- alphas[keep_coefs] .* diag(Sigma[keep_coefs, keep_coefs])

    alphas[keep_coefs] = (gammas .+ 2*a) ./ (w_mp[keep_coefs].^2 .+ 2*b)

    ll = fill(NaN, max_iter)

    ll[1] = -nll(w_mp, X, y, alphas)

    for iter in 2:max_iter
        w_mp = newtons_method(w_mp, X, y, alphas)
    
        p = logistic.(X * w_mp)
    
        B = diagm(p .* (1 .- p))

        hess[keep_coefs, keep_coefs] = -(X[:, keep_coefs]' * B * X[:, keep_coefs] .+ diagm(alphas[keep_coefs]))

        Sigma = inv(-hess)

        gammas = 1 .- alphas[keep_coefs] .* diag(Sigma[keep_coefs, keep_coefs])

        alphas[keep_coefs] = (gammas .+ 2*a) ./ (w_mp[keep_coefs].^2 .+ 2*b)

        keep_coefs[alphas .> max_alpha] .= false

        w_mp[.!keep_coefs] .= 0
    
        ll[iter] = -nll(w_mp, X, y, alphas)

        if abs(ll[iter] - ll[iter-1]) < tol
            println(string("converged in $iter", " iterations"))
            break
        end

    end

    Sigma = (Sigma .+ conj(Sigma)') ./ 2

    probs = logistic.(X*w_mp)

    return (w = w_mp, 
           covar = Sigma,
           p = probs,
           alpha = alphas, 
           ll = ll[.!isnan.(ll)],
           acc = mean(round.(probs) .== y)
    )

end

#X_b, y_b = skd.make_classification(
#    n_samples=1000, n_features=20, n_informative=2, n_redundant=2, random_state=42
#)


#X_b, y_b = skd.load_iris(return_X_y=true)
#use_rows = (y_b .== 1) .| (y_b .== 0)
#X = X_b[use_rows, :]
#y = y_b[use_rows]


#a=ard_laplace(y, X; max_iter=100, max_alpha=1000, tol=1e-2)
