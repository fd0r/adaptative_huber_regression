# /!\ WORK IN PROGRESS /!\

import torch

class HuberLoss:
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def __call__(self, output, target):
        diff = torch.abs(output - target)
        return torch.mean(
            torch.where(
                diff <= self.tau,
                (diff**2)/2,
                self.tau*diff - ((self.tau**2)/2)
            )
        )

def S(x, lamb):  # TODO: Test this function
    return torch.sign(x)*torch.max(
        torch.cat((np.abs(x)-lamb, np.zeros(x.shape)), 1)

class HuberRegressor(torch.nn.Module):
    def __init__(self, tau=2, fit_intercept=True, lambda_reg=0, gamma_u=2, 
                verbose=0, max_iter=3000, max_phi_iter=1000):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.loss = HuberLoss(tau)
        self.lambda_reg = lambda_reg
        self.gamma_u = gamma_u
        self.verbose = verbose
        self.max_iter = max_iter
        self.max_phi_iter = max_phi_iter

    def __call__(self, X):
        # TODO: Add Intercept
        if self.fit_intercept:
            X = torch.cat((X,torch.ones(X.shape[0],1)),1)
        return X @ self.beta

    def fit(self, X, y, beta_0, phi_0=2, convergence_threshold=1e-6):
        if self.fit_intercept:
            X = torch.cat((X,torch.ones(X.shape[0],1)),1)

        # TODO: Implement optimal parameters as in paper
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
                norm_diff = torch.sum(diff**2)

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
            X = torch.cat((X,torch.ones(X.shape[0],1)),1)
        return X@self.beta

