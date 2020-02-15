import numpy as np
import logging
from sklearn.base import BaseEstimator
import sys


class HuberLoss:
    def __init__(self, tau=1):
        super().__init__()
        # Parameters
        self.tau = tau
        self.tau_squared_2 = (tau**2)/2
        # Vectorized util functions
        self.loss = np.vectorize(self._huber_loss)
        self.grad = np.vectorize(self._huber_grad)

    def __call__(self, y_true: np.array, y_pred: np.array):
        return np.mean(
            self.loss(np.abs(y_true - y_pred))
        )

    def _huber_loss(self, x):
        if x <= self.tau:
            return (x**2)/2
        return self.tau*x - self.tau_squared_2

    def _huber_grad(self, x):
        if x <= self.tau:
            return x
        return self.tau * np.sign(x)

    def grad_loss(self, y_true, y_pred):
        return self.grad(np.abs(y_true - y_pred))


def S(x, lamb):
    return np.sign(x)*np.max(np.c_[np.abs(x)-lamb, np.zeros(x.shape)], axis=1)

class HuberRegressor(BaseEstimator):
    def __init__(self, tau, lambda_reg=0, gamma_u=2, verbose=0, fit_intercept=True,
                 max_iter=3000, max_phi_iter=1000):
        super().__init__()
        self.loss = HuberLoss(tau=tau)
        self.lambda_reg = lambda_reg
        self.logger = logging.getLogger('HuberRegressor')
        self.logger.setLevel(verbose)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.beta = None
        self.fit_intercept = fit_intercept
        self.gamma_u = gamma_u
        self.max_iter = max_iter
        self.max_phi_iter = max_phi_iter

    def fit(self, X, y, beta_0, phi_0=2, convergence_threshold=1e-6):
        # TODO: Implement optimal parameters as in paper


        # Add intercept if needed
        if self.fit_intercept:
            intercept = np.ones((X.shape[0],))
            X = np.c_[intercept, X]

        # Assert basic properties
        assert phi_0 != 0
        assert beta_0.shape == X.shape[1:]

        # Without regularization
        if self.lambda_reg == 0:
            raise NotImplementedError(
                "Version without regularization not implemented yet")

        # With regularization: LAMM Algorithm
        self.logger.info(
            "Fitting with LAMM algorithm and lambda={}".format(self.lambda_reg))
        
        step_counter = 0
        beta_k = beta_0

        # Optimal beta loop
        while True:
            self.logger.info("Step: {}".format(step_counter))
            # Find optimal phi
            phi_counter = 0
            phi = phi_0           
            pred_k = X @ beta_k
            loss_k = self.loss(y, pred_k)

            # TODO: CHECK THIS
            grad_k = np.mean(
                self.loss.grad_loss(y, pred_k)
                .reshape(y.shape+(1,))*X, axis=0)
            
            self.logger.debug('Loss at step {} = {} with grad={}'.format(
                step_counter, loss_k, grad_k))

            # Optimal phi loop
            while True:
                beta_k_1 = beta_k - (grad_k / self.lambda_reg)
                self.logger.debug(
                    'Beta_k before applying S: {}\nWith in S = {}'.format(
                        beta_k_1, beta_k_1 - self.lambda_reg/phi))
                beta_k_1 = S(beta_k_1, self.lambda_reg/phi)
                self.logger.debug(
                    'Beta_k after applying S:  {}\nWith lambda in S = {}'.format(
                        beta_k_1, self.lambda_reg/phi))

                diff = beta_k_1 - beta_k
                norm_diff = np.sum(diff**2)

                g_k = loss_k + grad_k @ diff + (phi/2)*norm_diff

                # TODO: check this
                self.logger.debug('G(beta_k_1 | beta_k) = {}\nLoss(beta_k_1) = {}'.format(
                    g_k, self.loss(y, X @ beta_k_1)))
                if g_k < self.loss(y, X @ beta_k_1):
                    phi *= self.gamma_u
                else:
                    self.logger.debug("Found good phi to be: {} after {} iters".format(phi, phi_counter))
                    break  # Found good phi
                phi_counter += 1
                
                # If max iter
                if phi_counter > self.max_phi_iter:
                    raise Exception("""Couldn't find phi with \gamma_u={}! 
Raising gamma_u might be a good idea!""".format(self.gamma_u))

            self.logger.debug('Converged after {} steps with phi_k={}'.format(
                phi_counter, phi
            ))
            self.logger.debug("Old Beta: {} \nNew Beta: {}".format(
                beta_k, beta_k_1
            ))
            # Update weights
            self.beta = beta_k_1
            beta_k = beta_k_1
            step_counter += 1

            # If too many iterations
            if step_counter > self.max_iter:
                self.logger.warning(
                    "Algorithm did not converge after {} iterations.".format(
                        step_counter))
                break

            # If convergence
            if np.sqrt(norm_diff) < convergence_threshold:
                break

        self.logger.info(
            "Algorithm runned {} iterations.".format(step_counter-1))
        return self

    def predict(self, X):
        if self.fit_intercept:
            intercept = np.ones((X.shape[0],))
            X = np.c_[intercept, X]
        return X@self.beta
