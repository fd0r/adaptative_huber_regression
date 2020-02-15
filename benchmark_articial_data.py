import numpy as np
import scipy
import logging
from sklearn.linear_model import LinearRegression
from ada_hub.huber_regressor import HuberRegressor

def generate_data(noise, n, d):
    
    beta = np.array([5, -2, 0, 0, 3])
    beta = np.concatenate((beta, np.zeros((max(d-5, 0),))))

    x = scipy.stats.multivariate_normal(
        np.zeros((d,)), np.identity(d)).rvs(size=n)
    epsilon = noise.rvs(size=n)
    y = x @ beta + epsilon
    return x, y, beta


# In the original paper l2 errors are averaged over 100 simulations.add()

if __name__ == "__main__":
    np.random.seed(42)
    
    ds = [5, 100, 500, 1000]
    n = 100
    tau = 2
  
    noises = [
        (scipy.stats.norm(0, 4), 'normal'),
        (scipy.stats.t(df=1.5), 't-distrib'),
        (scipy.stats.lognorm(s=4, scale=np.exp(0)), 'log-norm'),
    ]
    for d in ds:
        for noise, name in noises:
            logging.info(d, name)
            
            x, y, beta_opt = generate_data(noise, n, d)

            c_tau = .5    # cross-val between {.5, 1, 1.5} in original paper
            c_lambda = .5 # cross-val between {.5, 1, 1.5} in original paper

            t = np.log(n) 
            y_hat = np.mean(y)
            sigma_hat = np.sqrt(np.mean((y - y_hat)**2))
                      # for simplicity
            
            if d >= n:
                n_eff = n / np.log(d)   # for simplicity
                lambda_reg = c_lambda*sigma_hat*np.sqrt(n_eff/t)
            else:
                n_eff = n
                lambda_reg = 0

            tau = c_tau*sigma_hat*np.sqrt(n_eff/t)


            lin_reg = LinearRegression()
            hub_reg = HuberRegressor(tau=tau, lambda_reg=lambda_reg)
            hub_reg.fit(x, y, 
                beta_0=(np.random.random(d+1,)-.5)*sigma_hat, 
                phi_0=1e-6, convergence_threshold=1e-8)
            lin_reg.fit(x, y)
            print(lin_reg.coef_)
            print(hub_reg.beta)
            print(beta_opt)
            # raise Exception