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

class HuberRegressor(torch.nn.Module):
    pass
