from ada_hub.huber_regressor import HuberRegressor
import numpy as np

n = 1000
d = 5 

x = np.random.random((n, d))



y = np.random.random((n, )) + 5*x[:,0] + 10*x[:,1] + 4

optimal_beta = np.array([4, 5, 10, 0, 0, 0])

"""
For numerical studies and real data analysis, in the case where the actual order
of moments is unspecified, we presume the variance is finite and therefore choose
robustification and regularization parameters as follows:
"""

c_tau = .5 # cross-val between {.5, 1, 1.5}
c_lambda = .5 # cross-val between {.5, 1, 1.5}

y_hat = np.mean(y)
sigma_hat = np.sqrt(np.mean((y - y_hat)**2))
n_eff = n / np.log(d)  # for simplicity
t = np.log(n)  # for simplicity
tau = c_tau*sigma_hat*np.sqrt(n_eff/t)
lambda_reg = c_lambda*sigma_hat*np.sqrt(n_eff/t)
lambda_reg = 1e-2

print('Tau={}\nLambda={}'.format(tau, lambda_reg))

regressor = HuberRegressor(
    tau=tau, lambda_reg=lambda_reg, verbose='DEBUG')

regressor.fit(
    x, y, beta_0=np.random.random(d+1,)*2*sigma_hat, 
    phi_0=1e-2)

y_pred = regressor.predict(x)

print(np.mean((y-y_pred)**2))

print(np.sum((regressor.beta - optimal_beta)**2))

import matplotlib.pyplot as plt

plt.scatter(x[:,0], y)
plt.scatter(x[:,0], y_pred)
plt.savefig('scatter.png')
