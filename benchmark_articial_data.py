import numpy as np
import scipy
import logging
from sklearn.linear_model import LinearRegression
from ada_hub.huber_regressor import HuberRegressor as AdHuberRegressor
from ada_hub.huber_regressor import HuberLoss
from sklearn.linear_model import HuberRegressor as SkHuberRegressor


def generate_data(noise, n, d):

    beta = np.array([5, -2, 0, 0, 3])
    beta = np.concatenate((beta, np.zeros((max(d - 5, 0),))))

    x = scipy.stats.multivariate_normal(np.zeros((d,)), np.identity(d)).rvs(size=n)
    epsilon = noise.rvs(size=n)
    y = x @ beta + epsilon
    return x, y, beta


import matplotlib.pyplot as plt


def scatter(x, y, var=0, name="scatter.png"):
    plt.scatter(x[:, var], y)
    plt.savefig(name)


# In the original paper l2 errors are averaged over 100 simulations.add()

if __name__ == "__main__":
    np.random.seed(42)

    ds = [5, 100, 500, 1000]
    n = 100
    tau = 2
    loss = HuberLoss(tau=tau)
    noises = [
        (scipy.stats.norm(0, 4), "normal"),
        (scipy.stats.t(df=1.5), "t-distrib"),
        (scipy.stats.lognorm(s=4, scale=np.exp(0)), "log-norm"),
    ]
    for d in ds:
        for noise, name in noises:
            print()
            print(d, name)

            x, y, beta_opt = generate_data(noise, n, d)
            beta_opt = np.concatenate([[0], beta_opt])
            scatter(x, y, 0, "{}_{}.png".format(d, name))
            c_tau = 0.5  # cross-val between {.5, 1, 1.5} in original paper
            c_lambda = 1e-2  # cross-val between {.5, 1, 1.5} in original paper

            t = np.log(n)
            y_hat = np.mean(y)
            sigma_hat = np.sqrt(np.mean((y - y_hat) ** 2))
            # for simplicity

            if d >= n:
                n_eff = n / np.log(d)  # for simplicity
                lambda_reg = c_lambda * sigma_hat * np.sqrt(n_eff / t)
            else:
                n_eff = n
                lambda_reg = 0

            tau = c_tau * sigma_hat * np.sqrt(n_eff / t)

            for name, regressor, args, kwargs in [
                ("Linear Regression", LinearRegression(), list(), dict()),
                (
                    "Adaptative Huber Regression",
                    AdHuberRegressor(tau=tau, lambda_reg=lambda_reg, verbose="DEBUG"),
                    list(),
                    dict(
                        beta_0=np.random.random(d + 1) * 2 * sigma_hat,
                        phi_0=1e-6,
                        convergence_threshold=1e-8,
                    ),
                ),
                (
                    "Huber Regression",
                    SkHuberRegressor(epsilon=tau, alpha=lambda_reg, max_iter=1000),
                    list(),
                    dict(),
                ),
            ]:
                regressor.fit(x, y, *args, **kwargs)
                y_pred = regressor.predict(x)
                beta_hat = np.concatenate([[regressor.intercept_], regressor.coef_])
                print(
                    name,
                    beta_hat,
                    np.sum((y - y_pred) ** 2),
                    loss(y, y_pred),
                    np.sum((beta_opt - beta_hat) ** 2),
                    sep="\n\t",
                )
