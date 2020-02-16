import numpy as np
import scipy
import logging
from sklearn.linear_model import LinearRegression
from ada_hub.huber_regressor import AdaptativeHuberRegressor as AdHuberRegressor
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


def scatter(x, y, var=0, name="./images/scatter.png"):
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

                for reg_name, regressor, args, kwargs in [
                    ("Linear Regression", LinearRegression(), list(), dict()),
                    ("Adaptative Huber Regression", AdHuberRegressor(c_tau=.5, c_lambda=.5), list(), dict(),),
                ]:
                    regressor.fit(x, y, *args, **kwargs)
                    y_pred = regressor.predict(x)
                    beta_hat = np.concatenate([[regressor.intercept_], regressor.coef_])
                    results[name][d][reg_name].append(np.sum((beta_opt - beta_hat) ** 2))

    with open("./results/results.json", "w") as file:
        file.write(json.dumps(results))
