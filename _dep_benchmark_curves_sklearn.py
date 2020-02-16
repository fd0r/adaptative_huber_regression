import os
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
import matplotlib.pyplot as plt
import json


def generate_data(noise, n, d):

    beta = np.array([5, -2, 0, 0, 3])
    beta = np.concatenate((beta, np.zeros((max(d - 5, 0),))))

    x = scipy.stats.multivariate_normal(np.zeros((d,)), np.identity(d)).rvs(size=n)
    epsilon = noise.rvs(size=n)
    y = x @ beta + epsilon
    return x, y, beta


def scatter(x, y, var=0, name="../images/scatter.png"):
    plt.scatter(x[:, var], y)
    plt.savefig(name)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# The goal is to generate the figures 3 and 4 from the original paper

if __name__ == "__main__":
    np.random.seed(42)
    params = [(500, 5), (500, 1000)]
    for n, d in params:
        dfs = np.arange(1, 3, 0.1) + 0.1
        deltas = list()
        beta_hats_linear = list()
        beta_hats = list()
        for df in tqdm(dfs):
            temp = list()
            temp_linear = list()
            for _ in range(100):
                delta = df - 1 - 0.05
                noise = scipy.stats.t(df=df)

                x, y, beta_opt = generate_data(noise, n, d)

                beta_opt = np.concatenate([[0], beta_opt])

                c_tau = 0.5  # cross-val between {.5, 1, 1.5} in original paper
                c_lambda = 1e-2  # cross-val between {.5, 1, 1.5} in original paper

                t = np.log(n)
                y_hat = np.mean(y)
                sigma_hat = np.sqrt(np.mean((y - y_hat) ** 2))
                # for simplicity

                if d >= n:
                    n_eff = n / np.log(d)  # for simplicity
                else:
                    n_eff = n

                lambda_reg = c_lambda * sigma_hat * np.sqrt(n_eff / t)

                tau = c_tau * sigma_hat * np.sqrt(n_eff / t)
                loss = HuberLoss(tau=tau)
                regressor = SkHuberRegressor(
                    epsilon=tau, alpha=lambda_reg, max_iter=10000
                )
                regressor.fit(x, y)
                beta_hat = np.concatenate([[regressor.intercept_], regressor.coef_])
                temp.append(beta_hat)
                regressor_linear = LinearRegression()
                regressor_linear.fit(x, y)
                beta_hat_linear = np.concatenate(
                    [[regressor_linear.intercept_], regressor_linear.coef_]
                )
                temp_linear.append(beta_hat_linear)

            deltas.append(delta)
            beta_hats.append(temp)
            beta_hats_linear.append(temp_linear)

        plt.plot(
            deltas,
            [
                np.mean(
                    [-np.log(np.sqrt(np.sum((beta - beta_opt) ** 2))) for beta in elt]
                )
                for elt in beta_hats
            ],
        )
        plt.savefig("../images/log_errors_{}_{}_sklearn.png".format(n, d))
        plt.close()

        plt.plot(
            deltas,
            [
                np.mean([np.sqrt(np.sum((beta - beta_opt) ** 2)) for beta in elt])
                for elt in beta_hats
            ],
            label="huber",
        )
        plt.plot(
            deltas,
            [
                np.mean([np.sqrt(np.sum((beta - beta_opt) ** 2)) for beta in elt])
                for elt in beta_hats_linear
            ],
            label="linear",
        )
        plt.legend()
        plt.savefig("../images/errors_{}_{}_sklearn.png".format(n, d))
        plt.close()

        with open("../results/errors_{}_{}_sklearn.json".format(n, d), "w") as file:
            file.write(
                json.dumps(
                    {"beta_hats": beta_hats, "beta_hats_linear": beta_hats_linear},
                    cls=NumpyEncoder,
                )
            )

