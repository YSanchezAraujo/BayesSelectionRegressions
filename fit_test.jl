include("src/ard_regression.jl");
include("src/ard_logistic.jl");
using PyCall, PyPlot;

sklm = pyimport("sklearn.linear_model");

skd = pyimport("sklearn.datasets");

X, y, true_weights = skd.make_regression(
    n_samples=100,
    n_features=100,
    n_informative=10,
    noise=5,
    coef=true,
    random_state=42,
)


res_ard = sklm.ARDRegression(compute_score=true, n_iter=30).fit(X, y)

xb0 = [ones(size(X, 1)) X]

w_ard = fit_ard(xb0, y; max_iter=50)

w_ols = coef(lm(xb0, y))

rmse(x, y) = sqrt(mean((x .- y).^2))

println(string("error between true weights and sklearn ARD: ", rmse(true_weights, res_ard.coef_)))
println(string("error between true weights and OLS: ", rmse(true_weights, w_ols[2:end])))
println(string("error between true weights and this ARD: ", rmse(true_weights, w_ard.w[2:end]))) 

plt.plot(true_weights, label = "true weights", lw = 3)
plt.plot(res_ard.coef_, linestyle = "--", color = "black", label = "sklearn", lw = 3)
plt.plot(w_ols[2:end], linestyle = "--", color = "grey", label = "ols", lw = 3)
plt.plot(w_ard.w[2:end], linestyle = "--", color = "tab:orange", label = "this ard", lw = 3)
plt.legend()

    
# testingn classification
X_b, y_b = skd.make_classification(
    n_samples=1000, n_features=20, n_informative=2, n_redundant=2, random_state=42
)


#X_b, y_b = skd.load_iris(return_X_y=true)
#use_rows = (y_b .== 1) .| (y_b .== 0)
#X = X_b[use_rows, :]
#y = y_b[use_rows]


a=ard_laplace(y_b, X_b; max_iter=100, max_alpha=100, tol=1e-2)
