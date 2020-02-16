import numpy as np
import scipy
import logging
from sklearn.linear_model import LinearRegression
from ada_hub.huber_regressor import HuberRegressor as AdHuberRegressor
from ada_hub.huber_regressor import HuberLoss
from sklearn.linear_model import HuberRegressor as SkHuberRegressor
from collections import defaultdict
import json
from tqdm import tqdm


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


# We do a benchmark of different methods over different types of data

if __name__ == "__main__":
    np.random.seed(42)

    ds = [5, 100, 500, 1000]
    n = 100

    noises = [
        (scipy.stats.norm(0, 4), "normal"),
        (scipy.stats.t(df=1.5), "t-distrib"),
        (scipy.stats.lognorm(s=4, scale=np.exp(0)), "log-norm"),
    ]

    results = defaultdict(  # name
        lambda: defaultdict(  # dimension
            lambda: defaultdict(lambda: list())  # regressor
        )
    )

    for d in ds:
        for noise, name in noises:
            print()
            print(d, name)
            for _ in tqdm(range(100)):

                x, y, beta_opt = generate_data(noise, n, d)
                beta_opt = np.concatenate([[0], beta_opt])
                scatter(x, y, 0, "{}_{}.png".format(d, name))

                t = np.log(n)
                y_hat = np.mean(y)
                sigma_hat = np.sqrt(np.mean((y - y_hat) ** 2))

                c_tau = .5  # cross-val between {.5, 1, 1.5} in original paper
                c_lambda = .5  # cross-val between {.5, 1, 1.5} in original paper
                if d >= n:
                    n_eff = n / np.log(d)  # for simplicity
                    # TODO: CHECK THIS
                    lambda_reg = c_lambda * sigma_hat * np.sqrt(t / n)
                else:
                    n_eff = n
                    lambda_reg = 0

                # n_eff / t in the paper > but n / t in theoretical part
                tau = c_tau * sigma_hat * np.sqrt(n / t)
                loss = HuberLoss(tau=tau)

                for reg_name, regressor, args, kwargs in [
                    ("Linear Regression", LinearRegression(), list(), dict()),
                    (
                        "Adaptative Huber Regression",
                        SkHuberRegressor(epsilon=tau, alpha=lambda_reg, max_iter=10000),
                        list(),
                        dict(),
                    ),
                ]:
                    regressor.fit(x, y, *args, **kwargs)
                    y_pred = regressor.predict(x)
                    beta_hat = np.concatenate([[regressor.intercept_], regressor.coef_])
                    if False:
                        print(
                            reg_name,
                            beta_hat,
                            np.sum((y - y_pred) ** 2),
                            loss(y, y_pred),
                            np.sum((beta_opt - beta_hat) ** 2),
                            sep="\n\t",
                        )
                    results[name][d][reg_name].append(np.sum((beta_opt - beta_hat) ** 2))

    with open("../results/results_sklearn.json", "w") as file:
        file.write(json.dumps(results))
