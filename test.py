from ada_hub.huber_regressor import HuberRegressor
import numpy as np

n = 1000
d = 5 

x = np.random.random((n, d))
y = np.random.random((n, )) + 5*x[:,0] + 10*x[:,1]

regressor = HuberRegressor(
    tau=2, lambda_reg=1, verbose='INFO')

regressor.fit(
    x, y, beta_0=np.random.random(d+1,), phi_0=10)

y_pred = regressor.predict(x)

print(np.mean((y-y_pred)**2))
