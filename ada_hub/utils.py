import numpy as np
import scipy

def generate_data(noise, n, d):
    beta = np.array([5, -2, 0, 0, 3])
    beta = np.concatenate((beta, np.zeros((max(d - 5, 0),))))

    x = scipy.stats.multivariate_normal(np.zeros((d,)), np.identity(d)).rvs(size=n)
    epsilon = noise.rvs(size=n)
    y = x @ beta + epsilon
    return x, y, beta

def find_parameters(x, y, c_tau, c_lambda):
    # Find hyper_parameters
    assert x.shape[0] == len(y)
    n = len(y)
    t = np.log(n)
    d = x.shape[1]
    y_hat = np.mean(y)
    # Estimate 2nd order moment
    sigma_hat = np.sqrt(np.mean((y - y_hat) ** 2))
    # We only consider \delta = 1
    # TODO: Implement this with arbitrary \delta
    if n <= d:  # High dimension
        tau = c_tau * sigma_hat * np.sqrt(n/np.log(d)*t)
        lambda_reg = c_lambda * sigma_hat * np.sqrt(t*np.log(d) / n)
    else:
        lambda_reg = 0
        tau = c_tau * sigma_hat * np.sqrt(n/t)

    return tau, lambda_reg